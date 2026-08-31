import asyncio
import json
from pathlib import Path

from test_knowledge_base import FakeKeywordStore, make_kb


def test_added_domain_documents_preserve_original_frozen_corpus():
    kb = make_kb()
    kb._load_default_docs()
    corpus_path = Path(__file__).parents[1] / "evaluation/datasets/rag_support_v1.json"
    old_corpus = json.loads(corpus_path.read_text(encoding="utf-8"))["corpus"]
    for old in old_corpus:
        if old["title"] == "权限申请补充规范":
            continue  # This is an evaluation-only supplement, not a startup default.
        content, metadata = kb._collection.rows[old["id"]]
        assert content == old["content"]
        assert metadata["title"] == old["title"]
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
