import asyncio

from mcp.tool_manager import MCPToolManager, Tool, ToolResult


def chunk(name, **kwargs):
    item = {"chunk_id": name, "title": name, "content": "步骤 " + name, "chunk": 0}
    item.update(kwargs)
    return item


def manager():
    instance = MCPToolManager("unit-test")

    async def rewrite(query, n=3):
        return [query, "改写一", "改写二", "改写三", "多余改写"]

    instance.rewrite_query = rewrite
    return instance


def test_rrf_deduplicates_by_id_not_similarity_and_dense_has_one_channel_vote():
    instance = manager()
    a, b = chunk("a"), chunk("b")
    fused = instance._rrf([[a, dict(a, score=0.8), b]], 20)
    assert len(fused) == 2
    assert fused[0]["rrf_score"] == 1 / 61
    assert fused[1]["rrf_score"] == 1 / 62
    dense = instance._rrf([[a, b]] * 4, 20)
    final = instance._rrf([dense, [b, a]], 20)
    assert final[0]["rrf_score"] == final[1]["rrf_score"]


def test_hybrid_original_keyword_query_and_capped_rewrites_with_actual_evidence_trace():
    instance = manager()
    calls = []

    async def dense(params, context):
        calls.append(("dense", params))
        return [chunk("dense"), chunk("shared")]

    async def keyword(params, context):
        calls.append(("keyword", params))
        return [chunk("keyword"), chunk("shared")]

    instance.register(Tool("knowledge_search", "dense", dense, {}))
    instance.register(Tool("keyword_search", "keyword", keyword, {}))
    instance.set_rag_components()
    result = asyncio.run(instance.search_with_rewrite("knowledge_search", "VPN 691", top_k=2))
    assert result.success
    assert result.data[0]["chunk_id"] == "shared"
    assert len(result.data) == 2
    assert [params["query"] for channel, params in calls if channel == "keyword"] == ["VPN 691"]
    assert len([1 for channel, _ in calls if channel == "dense"]) == 4
    assert all(params["top_k"] == 20 for _, params in calls)
    assert result.retrieval["evidence"] == result.data
    assert result.retrieval["rerank"]["status"] == "disabled"
    assert not result.reranked


def test_failed_dense_fallback_text_never_becomes_reranker_evidence():
    instance = manager()

    async def dense(params, context):
        raise RuntimeError("dense unavailable")

    async def keyword(params, context):
        return [chunk("real")]

    def fallback(params, context, error):
        return [chunk("fake", fallback=True)]

    class Ranker:
        async def rerank(self, query, items):
            assert [item["chunk_id"] for item in items] == ["real"]
            return {"items": items, "status": "failed", "error": "timeout"}

    instance.register(Tool("knowledge_search", "dense", dense, {}, fallback=fallback))
    instance.register(Tool("keyword_search", "keyword", keyword, {}))
    instance.set_rag_components(reranker=Ranker())
    result = asyncio.run(instance.search_with_rewrite("knowledge_search", "查询"))
    assert result.success
    assert result.retrieval["degraded"]
    assert not result.reranked
    assert [item["chunk_id"] for item in result.data] == ["real"]


def test_all_channels_fail_returns_no_evidence():
    instance = manager()
    instance.set_rag_components()
    result = asyncio.run(instance.search_with_rewrite("knowledge_search", "查询"))
    assert not result.success
    assert result.data == []
    assert result.retrieval["degraded"]


def test_cache_isolated_by_channel_and_invalidated_after_ingestion():
    instance = manager()
    calls = []

    async def dense(params, context):
        calls.append("dense")
        return [chunk("dense")]

    async def keyword(params, context):
        calls.append("keyword")
        return [chunk("keyword")]

    instance.register(Tool("knowledge_search", "dense", dense, {}, cache_ttl=300))
    instance.register(Tool("keyword_search", "keyword", keyword, {}, cache_ttl=300))
    instance.set_rag_components()

    async def run():
        params = {"query": "q", "top_k": 20}
        await instance.call("knowledge_search", params)
        result = await instance.call("knowledge_search", params)
        assert result.cached
        result = await instance.call("keyword_search", params)
        assert result.data[0]["chunk_id"] == "keyword"
        instance.invalidate_rag_cache()
        await instance.call("knowledge_search", params)

    asyncio.run(run())
    assert calls == ["dense", "keyword", "dense"]


def test_ingestion_during_retrieval_does_not_write_old_result_under_new_cache_version():
    instance = manager()
    count = 0

    async def dense(params, context):
        nonlocal count
        count += 1
        if count == 1:
            instance.invalidate_rag_cache()
        return [chunk(str(count))]

    instance.register(Tool("knowledge_search", "dense", dense, {}, cache_ttl=300))

    async def run():
        await instance.call("knowledge_search", {"query": "q"})
        result = await instance.call("knowledge_search", {"query": "q"})
        assert result.data[0]["chunk_id"] == "2"
        assert not result.cached

    asyncio.run(run())


def test_ingestion_during_hybrid_search_rejects_mixed_generation_evidence():
    instance = manager()
    count = 0

    async def dense(params, context):
        nonlocal count
        count += 1
        if count == 2:
            instance.invalidate_rag_cache()
        version = "old" if count == 1 else "new"
        return [chunk(version, doc_id="vpn", doc_version=version)]

    instance.register(Tool("knowledge_search", "dense", dense, {}))
    result = asyncio.run(instance.search_with_rewrite("knowledge_search", "VPN"))
    assert not result.success
    assert result.data == []
    assert result.retrieval["evidence"] == []
    assert result.retrieval["version_changed"]
    assert result.retrieval["degraded"]
    assert result.retrieval["started_generation"] < result.retrieval["generation"]
