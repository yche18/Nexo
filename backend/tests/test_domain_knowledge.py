import asyncio
import json
from pathlib import Path

from test_knowledge_base import FakeKeywordStore, make_kb


def test_domain_documents_match_current_snapshot_and_reload_idempotently():
    kb = make_kb()
    kb._load_default_docs()
    # Startup SOP revisions have their own snapshot; old eval labels stay frozen.
    snapshot_path = Path(__file__).parent / "fixtures/default_knowledge_v2.json"
    corpus = json.loads(snapshot_path.read_text(encoding="utf-8"))["corpus"]
    assert len(corpus) == len(kb._collection.rows) == 10
    for item in corpus:
        content, metadata = kb._collection.rows[item["id"]]
        assert content == item["content"]
        assert metadata["title"] == item["title"]
        assert metadata["chunk_index"] == item["chunk"]
        assert metadata["status"] == "ready"
    titles = {metadata["title"] for _, metadata in kb._collection.rows.values()}
    assert {
        "办公网络与 Wi-Fi 故障排查（演示 SOP）", "DNS 与内网访问异常（演示 SOP）",
        "打印机卡纸与打印异常（演示 SOP）", "电脑无法开机（演示 SOP）",
    } <= titles
    before = dict(kb._collection.rows)
    kb._load_default_docs()
    assert kb._collection.rows == before


def test_domain_documents_use_existing_keyword_publication_path():
    async def scenario():
        kb = make_kb()
        kb._load_default_docs()
        store = FakeKeywordStore()
        kb._keyword_store = store
        await kb.initialize_keyword_index()
        assert set(store.rows) == set(kb._collection.rows)
        assert all(item["status"] == "ready" for item in store.rows.values())
        assert any("打印机卡纸" in item["title"] for item in store.rows.values())
    asyncio.run(scenario())
