import asyncio

from mcp.evidence import select_chunks, expand_neighbors, merge_windows, evidence_chars


def chunk(index, **kwargs):
    item = {"chunk_id": str(index), "doc_id": "vpn", "doc_version": "v1", "section": "连接",
            "chunk_index": index, "chunk": index, "status": "ready", "title": "VPN", "content": "内容"}
    item.update(kwargs)
    return item


def test_core_budget_preserves_whole_chunks_and_does_not_count_fallback_as_evidence():
    first, second = chunk(0), chunk(1, content="重要步骤" * 100)
    budget = evidence_chars([dict(first, evidence_role="core")])
    selected = select_chunks([dict(first, fallback=True), first, second], 5, budget)
    assert len(selected) == 1
    assert selected[0]["content"] == "内容"
    assert evidence_chars(selected) <= budget


def test_expansion_deduplicates_overlapping_windows_and_rejects_other_versions_sections():
    cores = select_chunks([chunk(1), chunk(2)], 2, 4000)
    calls = []

    async def loader(item, window=1):
        calls.append(item["chunk_id"])
        return [chunk(0), chunk(1), chunk(2), chunk(3), chunk(9),
                chunk(2, chunk_id="other-version", doc_version="v2"),
                chunk(2, chunk_id="other-section", section="账户"),
                chunk(2, chunk_id="pending", status="pending")]

    expanded = asyncio.run(expand_neighbors(cores, loader))
    result = merge_windows(cores, expanded["chunks"], 4000)
    assert [item["chunk_id"] for item in result] == ["1", "2", "0", "3"]
    assert calls == ["1", "2"]
    assert expanded["errors"] == []


def test_neighbors_cannot_displace_core_evidence_or_break_budget():
    cores = select_chunks([chunk(1), chunk(2)], 2, 4000)
    budget = evidence_chars(cores)
    assert merge_windows(cores, [dict(chunk(0), evidence_role="neighbor")], budget) == cores


def test_legacy_can_be_read_but_cannot_expand_and_failure_preserves_cores():
    cores = select_chunks([chunk(1, status="legacy"), chunk(2)], 2, 4000)
    calls = []

    async def loader(item, window=1):
        calls.append(item["chunk_id"])
        raise RuntimeError("unavailable")

    result = asyncio.run(expand_neighbors(cores, loader, max_seeds=1))
    assert calls == ["2"]
    assert result == {"chunks": [], "errors": ["RuntimeError"]}
    assert merge_windows(cores, result["chunks"], 4000) == cores
