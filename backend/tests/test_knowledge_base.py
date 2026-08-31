import asyncio
import json
from pathlib import Path
import sys
import types

import pytest

if "chromadb" not in sys.modules:
    try:
        import chromadb  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["chromadb"] = types.ModuleType("chromadb")

from mcp.knowledge_base import KnowledgeBase


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.fail_update = False

    def upsert(self, ids, documents, metadatas):
        for chunk_id, content, meta in zip(ids, documents, metadatas):
            self.rows[chunk_id] = (content, dict(meta))

    def update(self, ids, metadatas):
        for chunk_id, meta in zip(ids, metadatas):
            content, _ = self.rows[chunk_id]
            self.rows[chunk_id] = (content, dict(meta))
            if self.fail_update:
                self.fail_update = False
                raise RuntimeError("partial Chroma publication")

    def get(self, include=None, ids=None, where=None):
        matches = []
        for chunk_id, (content, meta) in self.rows.items():
            if ids is not None and chunk_id not in ids:
                continue
            filters = where.get("$and", [where]) if where else []
            if any(any(meta.get(key) != value for key, value in condition.items()) for condition in filters):
                continue
            matches.append((chunk_id, content, dict(meta)))
        return {"ids": [item[0] for item in matches], "documents": [item[1] for item in matches],
                "metadatas": [item[2] for item in matches]}

    def query(self, query_texts, n_results, where=None):
        result = self.get(where=where)
        return {"ids": [result["ids"][:n_results]], "documents": [result["documents"][:n_results]],
                "metadatas": [result["metadatas"][:n_results]],
                "distances": [[0.2] * min(n_results, len(result["ids"]))]}

    def count(self):
        return len(self.rows)


class FakeKeywordStore:
    def __init__(self):
        self.rows = {}
        self.fail = False
        self.calls = 0
        self.on_index = None

    async def ensure_index(self):
        return False

    async def count_ready(self):
        return sum(item["status"] == "ready" for item in self.rows.values())

    async def index_chunks(self, chunks):
        self.calls += 1
        if self.fail:
            raise RuntimeError("ES unavailable")
        for item in chunks:
            self.rows[item["chunk_id"]] = dict(item)
        if self.on_index:
            await self.on_index(chunks)

    async def delete_document_chunks(self, doc_id, doc_version=None):
        self.rows = {key: item for key, item in self.rows.items()
                     if not (item["doc_id"] == doc_id and
                             (doc_version is None or item["doc_version"] == doc_version))}

    async def search(self, query, top_k):
        return [dict(item, score=1.0) for item in self.rows.values()
                if item["status"] == "ready"][:top_k]


def make_kb():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = FakeCollection()
    kb._keyword_store = None
    kb._import_lock = asyncio.Lock()
    kb._legacy_metadata = {}
    return kb


def test_repeated_document_import_is_idempotent():
    kb = make_kb()
    documents = [{"title": "权限申请规范", "content": "权限申请需说明目标系统与用途。"}]
    assert kb.add_documents(documents) == 1
    ids = set(kb._collection.rows)
    assert kb.add_documents(documents) == 1
    assert set(kb._collection.rows) == ids


def test_default_corpus_keeps_original_ids_and_content():
    kb = make_kb()
    kb._load_default_docs()
    path = Path(__file__).parents[1] / "evaluation/datasets/rag_support_v1.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))["corpus"]
    expected = {item["id"]: item for item in corpus}
    for chunk_id, (content, meta) in kb._collection.rows.items():
        assert chunk_id in expected
        item = expected[chunk_id]
        assert content == item["content"]
        assert meta["title"] == item["title"]
        assert meta["chunk_index"] == item["chunk"]


def test_single_long_sentence_is_bounded():
    chunks = make_kb()._chunk_text("网" * 1201)
    assert [len(chunk) for chunk in chunks] == [500, 500, 201]
    assert "".join(chunks) == "网" * 1201


def test_failed_new_version_preserves_old_and_retry_is_idempotent():
    async def scenario():
        kb = make_kb()
        store = FakeKeywordStore()
        kb._keyword_store = store
        await kb.add_documents_async([{"title": "VPN", "content": "旧流程"}])
        original = kb.search("VPN")[0]
        store.fail = True
        with pytest.raises(RuntimeError, match="ES unavailable"):
            await kb.add_documents_async([{"title": "VPN", "content": "新流程"}])
        assert [item["content"] for item in kb.search("VPN")] == ["旧流程"]
        assert kb.doc_count == 1
        store.fail = False
        await kb.add_documents_async([{"title": "VPN", "content": "新流程"}])
        result = kb.search("VPN")
        assert [item["content"] for item in result] == ["新流程"]
        assert original["chunk_id"] != result[0]["chunk_id"]
        count = kb._collection.count()
        await kb.add_documents_async([{"title": "VPN", "content": "新流程"}])
        assert kb._collection.count() == count
        keyword = await kb.keyword_search_handler({"query": "VPN"}, None)
        assert keyword[0]["chunk_id"] == result[0]["chunk_id"]
    asyncio.run(scenario())


def test_same_version_es_failure_does_not_hide_ready_chunk():
    async def scenario():
        kb = make_kb()
        kb._keyword_store = FakeKeywordStore()
        docs = [{"title": "VPN", "content": "已发布流程"}]
        await kb.add_documents_async(docs)
        kb._keyword_store.fail = True
        with pytest.raises(RuntimeError):
            await kb.add_documents_async(docs)
        assert kb.search("VPN")[0]["content"] == "已发布流程"
    asyncio.run(scenario())


def test_es_ready_is_hidden_until_chroma_publication():
    async def scenario():
        kb = make_kb()
        store = FakeKeywordStore()
        kb._keyword_store = store
        await kb.add_documents_async([{"title": "VPN", "content": "旧流程"}])
        async def observe(chunks):
            if chunks[0]["status"] == "ready":
                result = await kb.keyword_search_handler({"query": "VPN"}, None)
                assert [item["content"] for item in result] == ["旧流程"]
        store.on_index = observe
        await kb.add_documents_async([{"title": "VPN", "content": "新流程"}])
    asyncio.run(scenario())


def test_partial_chroma_publication_restores_old_state():
    async def scenario():
        kb = make_kb()
        kb._keyword_store = FakeKeywordStore()
        await kb.add_documents_async([{"title": "VPN", "content": "旧流程"}])
        kb._collection.fail_update = True
        with pytest.raises(RuntimeError, match="partial Chroma"):
            await kb.add_documents_async([{"title": "VPN", "content": "新流程"}])
        assert [item["content"] for item in kb.search("VPN")] == ["旧流程"]
        keyword = await kb.keyword_search_handler({"query": "VPN"}, None)
        assert [item["content"] for item in keyword] == ["旧流程"]
    asyncio.run(scenario())


def test_migration_preserves_ids_content_and_repeat_import():
    async def scenario():
        kb = make_kb()
        kb._collection.upsert(ids=["old-id"], documents=["VPN 原流程"],
                              metadatas=[{"title": "VPN", "chunk_index": 0, "total_chunks": 1}])
        kb._legacy_metadata = {item["chunk_id"]: dict(item, status="legacy")
                               for item in kb._legacy_updates(kb._read_chunks())}
        assert kb.search("VPN")[0]["status"] == "legacy"
        assert kb.migrate_legacy_metadata() == 1
        assert kb.migrate_legacy_metadata() == 0
        assert kb._collection.rows["old-id"][0] == "VPN 原流程"
        kb._keyword_store = FakeKeywordStore()
        assert await kb.rebuild_keyword_index() == 1
        await kb.add_documents_async([{"title": "VPN", "content": "VPN 原流程"}])
        assert list(kb._collection.rows) == ["old-id"]
        assert (await kb.keyword_search_handler({"query": "VPN"}, None))[0]["chunk_id"] == "old-id"
    asyncio.run(scenario())


def test_legacy_import_requires_explicit_migration():
    kb = make_kb()
    kb._legacy_metadata = {"old": {}}
    with pytest.raises(RuntimeError, match="迁移"):
        kb.add_documents([{"title": "VPN", "content": "新流程"}])


def test_ambiguous_legacy_versions_are_not_silently_merged():
    kb = make_kb()
    kb._collection.upsert(ids=["a", "b"], documents=["旧版", "新版"],
                          metadatas=[{"title": "VPN", "chunk_index": 0}] * 2)
    with pytest.raises(ValueError, match="重复"):
        kb.migrate_legacy_metadata()
    assert "status" not in kb._collection.rows["a"][1]


def test_neighbor_window_never_crosses_version_section_or_status():
    async def scenario():
        kb = make_kb()
        chunks = kb._prepare_document({"title": "VPN", "content": "网" * 1201})
        for item in chunks:
            item["status"] = "ready"
        foreign = dict(chunks[1], chunk_id="foreign", section="another")
        pending = dict(chunks[1], chunk_id="pending", status="pending")
        old = dict(chunks[1], chunk_id="old", doc_version="old")
        kb._upsert_chunks(chunks + [foreign, pending, old])
        neighbors = await kb.get_neighbors_async(chunks[0])
        assert [item["chunk_index"] for item in neighbors] == [0, 1]
        assert {item["chunk_id"] for item in neighbors} == {chunks[0]["chunk_id"], chunks[1]["chunk_id"]}
    asyncio.run(scenario())


def test_frozen_corpus_migration_keeps_every_id_and_text():
    kb = make_kb()
    path = Path(__file__).parents[1] / "evaluation/datasets/rag_support_v1.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))["corpus"]
    kb._collection.upsert(
        ids=[item["id"] for item in corpus], documents=[item["content"] for item in corpus],
        metadatas=[{"title": item["title"], "chunk_index": item["chunk"]} for item in corpus],
    )
    assert kb.migrate_legacy_metadata() == len(corpus)
    for item in corpus:
        content, meta = kb._collection.rows[item["id"]]
        assert content == item["content"]
        assert meta["title"] == item["title"]
        assert meta["chunk_index"] == item["chunk"]
        assert meta["chunk_id"] == item["id"]


def test_startup_repairs_empty_keyword_index_but_does_not_reindex_each_time():
    async def scenario():
        kb = make_kb()
        kb._load_default_docs()
        store = FakeKeywordStore()
        kb._keyword_store = store
        await kb.initialize_keyword_index()
        assert len(store.rows) == kb.doc_count
        calls = store.calls
        await kb.initialize_keyword_index()
        assert store.calls == calls
    asyncio.run(scenario())
