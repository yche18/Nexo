import asyncio

import httpx
import pytest

from mcp.reranker import HTTPReranker


def candidates():
    return [{"chunk_id": "a", "title": "VPN", "content": "认证流程"},
            {"chunk_id": "b", "title": "账号", "content": "账号解锁"}]


class FakeClient:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return httpx.Response(200, json=self.data, request=httpx.Request("POST", url))


def test_rerank_validates_and_sorts_complete_query_document_scores():
    client = FakeClient({"results": [{"index": 0, "relevance_score": 0.1},
                                      {"index": 1, "relevance_score": 0.9}]})
    ranker = HTTPReranker("https://rerank.example/rerank", client=client)
    result = asyncio.run(ranker.rerank("账号锁定", candidates()))
    assert result["status"] == "success"
    assert [item["chunk_id"] for item in result["items"]] == ["b", "a"]
    assert client.calls[0]["json"]["top_n"] == 2
    assert client.calls[0]["json"]["documents"] == ["VPN\n认证流程", "账号\n账号解锁"]


@pytest.mark.parametrize("rows", [
    [{"index": 0, "relevance_score": 0.8}],
    [{"index": 0, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.7}],
    [{"index": 0, "relevance_score": 0.8}, {"index": 9, "relevance_score": 0.7}],
    [{"index": False, "relevance_score": 0.8}, {"index": 1, "relevance_score": 0.7}],
    [{"index": 0, "relevance_score": "0.8"}, {"index": 1, "relevance_score": 0.7}],
])
def test_invalid_reranker_result_preserves_fusion_order(rows):
    ranker = HTTPReranker("https://rerank.example/rerank", client=FakeClient({"results": rows}))
    items = candidates()
    result = asyncio.run(ranker.rerank("查询", items))
    assert result["status"] == "failed"
    assert result["items"] == items
    assert all("rerank_score" not in item for item in items)


def test_timeout_is_fallback_and_missing_config_does_not_call_service():
    client = FakeClient(error=httpx.ReadTimeout("timeout"))
    ranker = HTTPReranker("https://rerank.example/rerank", client=client)
    assert asyncio.run(ranker.rerank("查询", candidates()))["status"] == "failed"
    disabled = FakeClient()
    result = asyncio.run(HTTPReranker(client=disabled).rerank("查询", candidates()))
    assert result["status"] == "disabled"
    assert disabled.calls == []


def test_non_finite_scores_do_not_pass_validation():
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"index": 0, "relevance_score": float("nan")},
                                {"index": 1, "relevance_score": 0.9}]}

    class Client:
        async def post(self, *args, **kwargs):
            return Response()

    result = asyncio.run(HTTPReranker("https://rerank.example/rerank", client=Client()).rerank("q", candidates()))
    assert result["status"] == "failed"


def test_qwen_compatible_results_use_original_chunks_and_configured_model():
    client = FakeClient({"results": [
        {"index": 1, "relevance_score": 0.9, "document": {"text": "provider replacement text"}},
        {"index": 0, "relevance_score": 0.2},
    ], "usage": {"total_tokens": 17}})
    ranker = HTTPReranker("https://rerank.example/compatible-api/v1/reranks",
                          api_key="test-only-key", model="qwen3-rerank", client=client)
    result = asyncio.run(ranker.rerank("账号解锁", candidates()))
    assert result["status"] == "success"
    assert result["items"][0]["content"] == "账号解锁"
    assert result["items"][0]["chunk_id"] == "b"
    assert client.calls[0]["json"]["model"] == "qwen3-rerank"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer test-only-key"
