import json
from pathlib import Path

import pytest

from evaluation.run_rag_eval import (
    chunk_key, map_evidence, read_chat_evidence, summarize, validate, validate_judgment,
)


def test_hybrid_cases_use_unchanged_original_corpus():
    folder = Path(__file__).parents[1] / "evaluation/datasets"
    original = json.loads((folder / "rag_support_v1.json").read_text(encoding="utf-8"))
    extension = json.loads((folder / "rag_support_hybrid_v1.json").read_text(encoding="utf-8"))
    validate(original)
    validate(extension)
    assert len(original["cases"]) == 24
    assert len(extension["cases"]) == 8
    assert extension["corpus"] == original["corpus"]
    assert not {c["question"] for c in extension["cases"]} & {c["question"] for c in original["cases"]}


def test_trace_reads_actual_evidence_across_calls_without_duplicate_coverage():
    first = {"chunk_id": "a", "title": "VPN", "chunk": 0, "content": "VPN 401"}
    second = {"chunk_id": "b", "title": "Web", "chunk": 0, "content": "Web 500"}
    retrievals = [{"evidence": [first]}, {"evidence": [first, second], "degraded": True}]
    body = {"found": True, "trace": {"tool_calls": [
        {"tool_name": "lookup_error_code"},
        *[{"tool_name": "search_knowledge_base", "retrieval": r} for r in retrievals],
    ]}}
    evidence, calls = read_chat_evidence(body)
    assert evidence == [first, second]
    assert calls == retrievals


def test_old_or_missing_trace_is_not_reported_as_zero_recall():
    for body in (
        {"found": False},
        {"found": True, "trace": {"tool_calls": [{"tool_name": "search_knowledge_base"}]}},
    ):
        with pytest.raises(ValueError):
            read_chat_evidence(body)
    evidence, calls = read_chat_evidence({"found": True, "trace": {"tool_calls": []}})
    assert evidence == calls == []


def test_changed_text_cannot_match_frozen_evidence_by_id_only():
    source = {"title": "VPN", "chunk": 0, "content": "original"}
    mapping = {chunk_key(source): "a"}
    changed = {**source, "chunk_id": "a", "content": "different"}
    ids, unknown = map_evidence([source, changed], mapping)
    assert ids == ["a", None]
    assert unknown == [changed]


def test_summary_separates_answer_quality_chat_evidence_and_standalone_recall():
    rows = [
        {"answerable": True, "relevant_chunk_ids": ["a"], "judge_valid": True,
         "answer_passed": True, "evidence_available": True, "evidence_recall": 1.0,
         "retrieval_degraded": True, "chat_seconds": 2.0},
        {"answerable": True, "relevant_chunk_ids": ["b"], "judge_valid": False,
         "judge_error": "invalid", "evidence_available": False, "chat_seconds": 4.0},
    ]
    result = summarize(rows, 2)
    assert result["answer_pass_rate_all_processed"] == 0.5
    assert result["answer_pass_rate_valid_judgments"] == 1.0
    assert result["macro_recall_at_5"] is None
    assert result["chat_evidence"]["macro_evidence_recall"] == 1.0
    assert result["chat_evidence"]["unavailable_cases"] == 1
    assert result["chat_evidence"]["degraded_cases"] == 1
    assert result["chat_seconds_mean"] == 3.0


def test_judge_requires_every_point_and_no_material_or_forbidden_claim():
    verdict = {"points": [{"index": 1, "met": True, "reason": "supported"},
                           {"index": 2, "met": False, "reason": "missing"}],
               "material_error": False, "forbidden_claim": False, "reason": "partial"}
    assert not validate_judgment(verdict, 2)
    verdict["points"][1]["met"] = True
    assert validate_judgment(verdict, 2)
    verdict["forbidden_claim"] = True
    assert not validate_judgment(verdict, 2)


@pytest.mark.parametrize("standalone", [False, True])
def test_run_uses_chat_trace_and_search_is_explicit(monkeypatch, tmp_path, standalone):
    import asyncio
    from types import SimpleNamespace
    import anthropic
    import sys
    chromadb = SimpleNamespace(HttpClient=None, Settings=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "chromadb", chromadb)
    import httpx
    from evaluation.run_rag_eval import run

    chunk = {"id": "a", "title": "VPN", "chunk": 0, "content": "401 means authentication failure"}
    data = {"version": "test", "annotation_note": "offline", "corpus": [chunk], "cases": [
        {"id": "one", "question": "VPN 401?", "answerable": True,
         "relevant_chunk_ids": ["a"], "required_points": ["Authentication failure"],
         "forbidden_claims": ["No performed action"]},
    ]}
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps(data), encoding="utf-8")
    collection = SimpleNamespace(get=lambda **kwargs: {
        "ids": ["a"], "documents": [chunk["content"]],
        "metadatas": [{"title": "VPN", "chunk_index": 0, "status": "ready"}],
    })
    monkeypatch.setattr(chromadb, "HttpClient", lambda **kwargs: SimpleNamespace(get_collection=lambda name: collection))
    calls = []

    class FakeHttp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            calls.append(url)
            if url.endswith("/chat"):
                body = {"request_id": "request-one", "response": "Authentication failure", "knowledge_used": True}
            else:
                assert url.endswith("/search") and standalone
                body = {"results": [chunk]}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

        async def get(self, url):
            calls.append(url)
            assert url.endswith("/trace/tool/request-one")
            body = {"found": True, "trace": {"tool_calls": [
                {"tool_name": "search_knowledge_base", "retrieval": {"evidence": [chunk], "degraded": False}},
            ]}}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    class FakeJudge:
        def __init__(self, **kwargs):
            self.messages = self

        async def create(self, **kwargs):
            verdict = {"points": [{"index": 1, "met": True, "reason": "supported"}],
                       "material_error": False, "forbidden_claim": False, "reason": "ok"}
            return SimpleNamespace(content=[{"type": "tool_use", "id": "judge-one",
                                             "name": "submit_evaluation", "input": verdict}], stop_reason="tool_use")

        async def close(self):
            pass

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: FakeHttp())
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeJudge)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "offline-test")
    args = SimpleNamespace(dataset=dataset, self_test=False, validate_only=False,
                           report=tmp_path / "report.json", application_revision="test",
                           retry_invalid_from=None, standalone_retrieval=standalone, api_base="http://offline")
    asyncio.run(run(args))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert report["summary"]["answer_passed"] == 1
    assert report["summary"]["chat_evidence"]["macro_evidence_recall"] == 1.0
    assert report["results"][0]["evidence_ids"] == ["a"]
    assert any(url.endswith("/search") for url in calls) == standalone
    assert report["corpus_unchanged_after_run"] is True
    with pytest.raises(RuntimeError, match="Report exists"):
        asyncio.run(run(args))
