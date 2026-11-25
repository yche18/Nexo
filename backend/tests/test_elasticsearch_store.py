import asyncio
import json

import httpx
import pytest

from mcp.elasticsearch_store import ElasticsearchStore


def test_index_mapping_and_idempotent_bulk_ids():
    async def scenario():
        requests = []
        def handle(request):
            requests.append(request)
            if request.method == "GET":
                return httpx.Response(404)
            return httpx.Response(200, json={"errors": False})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            store = ElasticsearchStore(url="http://es", client=client)
            await store.ensure_index()
            mapping = json.loads(requests[1].content)["mappings"]["properties"]
            assert mapping["content"]["analyzer"] == "ik_max_word"
            assert mapping["content"]["search_analyzer"] == "ik_smart"
            chunks = [{"chunk_id": "same", "content": "VPN", "status": "pending"}]
            await store.index_chunks(chunks)
            await store.index_chunks(chunks)
            assert requests[-1].content == requests[-2].content
            assert json.loads(requests[-1].content.splitlines()[0])["index"]["_id"] == "same"
    asyncio.run(scenario())


def test_keyword_query_filters_ready_and_returns_identity():
    async def scenario():
        def handle(request):
            body = json.loads(request.content)
            assert body["query"]["bool"]["filter"] == [{"term": {"status": "ready"}}]
            assert body["query"]["bool"]["must"][0]["multi_match"]["fields"] == ["title^2", "content"]
            return httpx.Response(200, json={"hits": {"hits": [{"_id": "c1", "_score": 2.4,
                "_source": {"doc_id": "d1", "doc_version": "v1", "chunk_index": 1,
                            "title": "VPN", "content": "401", "status": "ready"}}]}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await ElasticsearchStore(url="http://es", client=client).search("VPN 401")
            assert result[0]["chunk_id"] == "c1"
            assert result[0]["chunk"] == 1
            assert result[0]["channel"] == "keyword"
    asyncio.run(scenario())


@pytest.mark.parametrize("status,body", [
    (503, {}), (200, {"timed_out": True}), (200, {"_shards": {"failed": 1}}),
])
def test_search_failure_is_not_empty_success(status, body):
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json=body)
        )) as client:
            with pytest.raises((httpx.HTTPStatusError, RuntimeError)):
                await ElasticsearchStore(url="http://es", client=client).search("VPN")
    asyncio.run(scenario())


def test_bulk_partial_failure_is_reported():
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"errors": True, "items": [{"index": {"error": {"type": "mapping"}}}]}
        ))) as client:
            with pytest.raises(RuntimeError, match="写入失败"):
                await ElasticsearchStore(url="http://es", client=client).index_chunks([{"chunk_id": "a"}])
    asyncio.run(scenario())


def test_delete_scopes_document_and_version():
    async def scenario():
        def handle(request):
            filters = json.loads(request.content)["query"]["bool"]["filter"]
            assert filters == [{"term": {"doc_id": "d"}}, {"term": {"doc_version": "v"}}]
            return httpx.Response(200, json={"deleted": 1})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await ElasticsearchStore(url="http://es", client=client).delete_document_chunks("d", "v")
    asyncio.run(scenario())


def test_existing_incompatible_mapping_is_rejected():
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"nexo_knowledge": {"mappings": {"properties": {}}}}
        ))) as client:
            with pytest.raises(RuntimeError, match="IK"):
                await ElasticsearchStore(url="http://es", client=client).ensure_index()
    asyncio.run(scenario())
