"""知识切片的 Elasticsearch BM25 召回；只提供排名，不与向量分数直接相加。"""
import json
import os
import re

import httpx


class ElasticsearchStore:
    def __init__(self, url=None, index_name=None, timeout=10, username=None,
                 password=None, api_key=None, client=None):
        self.url = (url or os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")).rstrip("/")
        self.index_name = index_name or os.getenv("ELASTICSEARCH_INDEX", "nexo_knowledge")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.index_name):
            raise ValueError("Elasticsearch 索引名只能包含小写字母、数字、下划线和连字符")
        headers = {"Authorization": f"ApiKey {api_key}"} if api_key else {}
        auth = (username, password or "") if username else None
        self._client = client or httpx.AsyncClient(timeout=timeout, headers=headers, auth=auth)
        self._owns_client = client is None

    async def ensure_index(self):
        response = await self._client.get(f"{self.url}/{self.index_name}")
        if response.status_code == 200:
            mapping = response.json()[self.index_name]["mappings"].get("properties", {})
            for name in ("title", "content"):
                field = mapping.get(name, {})
                if field.get("analyzer") != "ik_max_word" or field.get("search_analyzer") != "ik_smart":
                    raise RuntimeError("现有 Elasticsearch 索引不是预期 IK 配置，请使用新的知识索引名重建")
            for name in ("doc_id", "doc_version", "chunk_id", "status", "section"):
                if mapping.get(name, {}).get("type") != "keyword":
                    raise RuntimeError("现有 Elasticsearch 索引元数据映射不匹配，请使用新的知识索引名重建")
            return False
        if response.status_code != 404:
            response.raise_for_status()
        response = await self._client.put(f"{self.url}/{self.index_name}", json={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"properties": {
                "title": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                "content": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
                "doc_id": {"type": "keyword"}, "doc_version": {"type": "keyword"},
                "chunk_id": {"type": "keyword"}, "chunk_index": {"type": "integer"},
                "total_chunks": {"type": "integer"}, "source": {"type": "keyword"},
                "section": {"type": "keyword"}, "status": {"type": "keyword"},
            }},
        })
        response.raise_for_status()
        return True

    async def count_ready(self):
        response = await self._client.post(f"{self.url}/{self.index_name}/_count", json={
            "query": {"term": {"status": "ready"}},
        })
        response.raise_for_status()
        result = response.json()
        if result.get("_shards", {}).get("failed", 0):
            raise RuntimeError("Elasticsearch 切片计数未完整完成")
        return result["count"]

    async def index_chunks(self, chunks):
        if not chunks:
            return
        lines = []
        for item in chunks:
            lines.append(json.dumps({"index": {"_id": item["chunk_id"]}}))
            lines.append(json.dumps(item, ensure_ascii=False))
        response = await self._client.post(
            f"{self.url}/{self.index_name}/_bulk", params={"refresh": "true"},
            content="\n".join(lines) + "\n", headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        result = response.json()
        if result.get("errors"):
            failures = [item["index"].get("error") for item in result.get("items", [])
                        if item.get("index", {}).get("error")]
            raise RuntimeError(f"Elasticsearch 切片写入失败: {failures}")

    async def delete_document_chunks(self, doc_id, doc_version=None):
        filters = [{"term": {"doc_id": doc_id}}]
        if doc_version is not None:
            filters.append({"term": {"doc_version": doc_version}})
        response = await self._client.post(
            f"{self.url}/{self.index_name}/_delete_by_query", params={"refresh": "true"},
            json={"query": {"bool": {"filter": filters}}},
        )
        response.raise_for_status()
        result = response.json()
        if result.get("failures") or result.get("timed_out") or result.get("version_conflicts"):
            raise RuntimeError("Elasticsearch 旧切片清理未完成")

    async def search(self, query, top_k=5):
        if top_k <= 0 or not query.strip():
            return []
        response = await self._client.post(f"{self.url}/{self.index_name}/_search", json={
            "size": top_k,
            "query": {"bool": {
                "filter": [{"term": {"status": "ready"}}],
                "must": [{"multi_match": {"query": query, "fields": ["title^2", "content"]}}],
            }},
        })
        response.raise_for_status()
        result = response.json()
        if result.get("timed_out") or result.get("_shards", {}).get("failed", 0):
            raise RuntimeError("Elasticsearch 检索未完整完成")
        items = []
        for hit in result["hits"]["hits"]:
            item = dict(hit["_source"])
            item.update(chunk_id=hit["_id"], chunk=item["chunk_index"],
                        score=hit.get("_score", 0.0), channel="keyword")
            items.append(item)
        return items

    async def close(self):
        if self._owns_client:
            await self._client.aclose()
