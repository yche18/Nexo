"""手动运行真实 Chroma + Elasticsearch 存储冒烟，不调用聊天/评判 LLM。

在 backend 容器中运行：
  python /app/evaluation/run_rag_store_smoke.py
  python /app/evaluation/run_rag_store_smoke.py --report /app/data/eval/store_smoke.json

使用镜像已有的 Chroma 默认 embedding。只创建 nexo_rag_smoke_<uuid> 隔离
collection/index；finally 删除本次成功创建的两个资源，不访问正式知识库或记忆。
此脚本验证存储与检索接口契约，不测回答质量或语义召回准确率。
"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

import chromadb
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from mcp.elasticsearch_store import ElasticsearchStore
from mcp.knowledge_base import KnowledgeBase


class SmokeKnowledgeBase(KnowledgeBase):
    """仅替换 collection 名称，复用实际导入、向量检索与邻居实现。"""


async def run_smoke(args):
    name = "nexo_rag_smoke_" + uuid.uuid4().hex
    SmokeKnowledgeBase.COLLECTION_NAME = name
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "collection": name, "index": name,
        "embedding": "Chroma default embedding; no chat or judge LLM",
        "status": "failed", "checks": [], "cleanup": {},
    }
    collection_created = False
    index_created = False
    client = None
    store = ElasticsearchStore(
        url=args.elasticsearch_url, index_name=name,
        username=os.getenv("ELASTICSEARCH_USERNAME"),
        password=os.getenv("ELASTICSEARCH_PASSWORD"),
        api_key=os.getenv("ELASTICSEARCH_API_KEY"),
    )
    try:
        client = chromadb.HttpClient(
            host=args.chroma_host, port=args.chroma_port,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        client.heartbeat()
        # create_collection 不复用同名集合：创建失败便不会把该名字标记为可清理。
        client.create_collection(name=name)
        collection_created = True
        kb = SmokeKnowledgeBase(
            chroma_host=args.chroma_host, chroma_port=args.chroma_port,
            keyword_store=store, allow_local_fallback=False, seed_defaults=False,
        )
        index_created = await store.ensure_index()
        assert index_created, "随机索引名已存在，停止测试且不删除该索引"
        await kb.initialize_keyword_index()
        assert kb.doc_count == 0 and await store.count_ready() == 0
        report["checks"].append({"name": "fresh_initialization", "passed": True})

        token = "vpnprobe" + uuid.uuid4().hex
        document = {
            "doc_id": name + "_vpn", "title": token,
            "source": "smoke://vpn", "section": "network",
            "content": ("VPN 连接排查：先记录客户端版本和错误提示，再检查网络、DNS 与代理配置，最后保存诊断日志。" * 40),
        }
        count = await kb.add_documents_async([document])
        original = kb._read_chunks(where={"doc_id": document["doc_id"]})
        original.sort(key=lambda item: item["chunk_index"])
        assert count == len(original) >= 3, "长文未生成至少三个切片"
        assert all(item["status"] == "ready" for item in original)
        assert all(len(item["content"]) <= 500 for item in original)
        assert await store.count_ready() == count
        old_ids = {item["chunk_id"] for item in original}
        old_version = original[0]["doc_version"]
        report["checks"].append({"name": "new_import", "passed": True, "chunks": count})

        dense = await kb.search_async("VPN 连接排查", top_k=3)
        keyword = await kb.keyword_search_handler({"query": token, "top_k": 3}, None)
        assert dense and keyword, "向量或关键词检索为空"
        assert all(item["chunk_id"] in old_ids and item["status"] == "ready"
                   for item in dense + keyword)
        assert all(item["doc_id"] == document["doc_id"] for item in keyword)
        report["checks"].append({"name": "both_retrieval_channels", "passed": True,
                                 "dense_hits": len(dense), "keyword_hits": len(keyword)})

        # 仅在隔离语料中把相邻切片设为另一章节，验证同 doc/version 仍不能跨 section。
        appendix = dict(original[2], section="appendix")
        kb._collection.update(ids=[appendix["chunk_id"]], metadatas=[kb._metadata(appendix)])
        await store.index_chunks([appendix])
        neighbors = await kb.get_neighbors_async(original[1], window=1)
        expected = {original[0]["chunk_id"], original[1]["chunk_id"]}
        assert {item["chunk_id"] for item in neighbors} == expected
        assert all(item["section"] == "network" and item["doc_version"] == old_version
                   for item in neighbors)
        report["checks"].append({"name": "section_bounded_neighbors", "passed": True,
                                 "neighbor_indexes": [item["chunk_index"] for item in neighbors]})

        revised = dict(document, content=(
            "VPN 新版排查：先检查客户端升级记录与证书状态，再记录连接时间和影响范围，最后附上新的诊断日志。" * 40
        ))
        revised_count = await kb.add_documents_async([revised])
        all_chunks = kb._read_chunks(where={"doc_id": document["doc_id"]})
        retired = [item for item in all_chunks if item["chunk_id"] in old_ids]
        ready = [item for item in all_chunks if item["status"] == "ready"]
        new_ids = {item["chunk_id"] for item in ready}
        new_versions = {item["doc_version"] for item in ready}
        assert len(retired) == count and all(item["status"] == "retired" for item in retired)
        assert len(ready) == revised_count and len(new_versions) == 1
        assert old_version not in new_versions and old_ids.isdisjoint(new_ids)
        dense = await kb.search_async("VPN 新版排查", top_k=3)
        keyword = await kb.keyword_search_handler({"query": token, "top_k": 3}, None)
        raw_keyword = await store.search(token, top_k=100)
        assert dense and keyword and raw_keyword
        assert all(item["chunk_id"] in new_ids for item in dense + keyword + raw_keyword)
        assert await store.count_ready() == revised_count
        report["checks"].append({"name": "version_replacement", "passed": True,
                                 "retired_chunks": len(retired), "ready_chunks": len(ready)})

        stored_count = kb._collection.count()
        assert await kb.add_documents_async([revised]) == revised_count
        final = kb._read_chunks(where={"status": "ready"})
        assert {item["chunk_id"] for item in final} == new_ids
        assert kb._collection.count() == stored_count
        assert await store.count_ready() == revised_count
        report["checks"].append({"name": "repeat_import_idempotency", "passed": True})
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # 名字由本函数生成，且只有创建成功后才允许精确删除；无通配符或其他集合名。
        if collection_created:
            try:
                client.delete_collection(name=name)
                report["cleanup"]["collection"] = "deleted"
            except Exception as exc:
                report["cleanup"]["collection"] = f"failed: {type(exc).__name__}: {exc}"
                report["status"] = "failed"
        if index_created:
            try:
                response = await store._client.delete(f"{store.url}/{name}")
                response.raise_for_status()
                report["cleanup"]["index"] = "deleted"
            except Exception as exc:
                report["cleanup"]["index"] = f"failed: {type(exc).__name__}: {exc}"
                report["status"] = "failed"
        await store.close()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main():
    load_dotenv(BACKEND.parent / ".env")
    parser = argparse.ArgumentParser(description="隔离的真实 Chroma + Elasticsearch 存储冒烟")
    parser.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", "localhost"))
    parser.add_argument("--chroma-port", type=int, default=int(os.getenv("CHROMA_PORT", "8000")))
    parser.add_argument("--elasticsearch-url", default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--report", type=Path, help="可选 JSON 报告路径；已存在则拒绝覆盖")
    args = parser.parse_args()
    if args.report and args.report.exists():
        parser.error("报告路径已存在，请选择新文件名")
    report = asyncio.run(run_smoke(args))
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as file:
            file.write(output + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
