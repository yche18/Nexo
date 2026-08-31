import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from evaluation.run_rag_eval import (
    JUDGE_CONTRACT_SHA256, JUDGE_PROTOCOL, judge_answer, retry_invalid,
    sha, validate_judgment,
)


def verdict():
    return {"points": [{"index": 1, "met": True, "reason": "说明所需字段"}],
            "material_error": False, "forbidden_claim": False, "reason": "符合要点"}


def tool_input(value=None, name="submit_evaluation"):
    return {"type": "tool_use", "id": "judge-one", "name": name,
            "input": verdict() if value is None else value}


class FakeClient:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason
        self.messages = self
        self.calls = []
        self.closed = False

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content, stop_reason=self.stop_reason)

    async def close(self):
        self.closed = True


def payload():
    return {"question": "需要提供什么？", "answer": "需提供系统和用途。",
            "answerable_from_corpus": True, "required_points": ["说明所需字段"],
            "forbidden_claims": ["不虚称已执行授权"], "reference_corpus": []}


def test_forced_tool_submission_preserves_sdk_content_and_existing_settings(monkeypatch):
    from anthropic.types import ToolUseBlock
    monkeypatch.setenv("LLM_THINKING_ENABLED", "false")
    block = ToolUseBlock(**tool_input())
    client = FakeClient([block])
    result = asyncio.run(judge_answer(client, "same-model", payload()))
    assert result["judge_valid"] and result["answer_passed"]
    assert result["judge"] == verdict()
    assert result["judge_content"] == [block.model_dump(mode="json")]
    assert json.loads(result["judge_raw"]) == result["judge_content"]
    assert result["judge_protocol"] == JUDGE_PROTOCOL
    assert result["judge_contract_sha256"] == JUDGE_CONTRACT_SHA256
    request = client.calls[0]
    assert request["tool_choice"] == {"type": "tool", "name": "submit_evaluation"}
    assert request["model"] == "same-model"
    assert request["max_tokens"] == 1024 and request["temperature"] == 0
    assert "output_config" not in request and "strict" not in request["tools"][0]
    schema = request["tools"][0]["input_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"points", "material_error", "forbidden_claim", "reason"}
    point = schema["properties"]["points"]["items"]
    assert point["additionalProperties"] is False
    assert set(point["required"]) == {"index", "met", "reason"}


@pytest.mark.parametrize("content,stop_reason", [
    ([], "end_turn"),
    ([{"type": "text", "text": json.dumps(verdict())}], "end_turn"),
    ([tool_input(name="another_tool")], "tool_use"),
    ([tool_input(), tool_input()], "tool_use"),
    ([tool_input(), tool_input(name="another_tool")], "tool_use"),
    ([tool_input()], "max_tokens"),
    ([{"type": "text", "text": json.dumps(verdict())}, tool_input({})], "tool_use"),
])
def test_invalid_protocol_keeps_raw_content_without_text_fallback(content, stop_reason):
    client = FakeClient(content, stop_reason)
    result = asyncio.run(judge_answer(client, "model", payload()))
    assert result["judge_valid"] is False
    assert result["answer_passed"] is False
    assert result["judge_content"] == content
    assert json.loads(result["judge_raw"]) == content
    assert result["judge_stop_reason"] == stop_reason
    assert result["judge_error"] == "ValueError"
    assert "judge" not in result
    assert len(client.calls) == 1


@pytest.mark.parametrize("change", [
    "missing_forbidden", "missing_met", "bool_index", "float_index", "string_met",
    "string_flag", "extra_root", "extra_point", "wrong_order", "wrong_count",
    "non_object", "non_list", "non_string_reason",
])
def test_schema_rejects_missing_wrong_types_and_extra_fields(change):
    value = verdict()
    if change == "missing_forbidden":
        del value["forbidden_claim"]
    elif change == "missing_met":
        del value["points"][0]["met"]
    elif change == "bool_index":
        value["points"][0]["index"] = True
    elif change == "float_index":
        value["points"][0]["index"] = 1.0
    elif change == "string_met":
        value["points"][0]["met"] = "true"
    elif change == "string_flag":
        value["material_error"] = "false"
    elif change == "extra_root":
        value["passed"] = True
    elif change == "extra_point":
        value["points"][0]["score"] = 1.0
    elif change == "wrong_order":
        value["points"][0]["index"] = 2
    elif change == "wrong_count":
        value["points"] = []
    elif change == "non_object":
        value = json.dumps(value)
    elif change == "non_list":
        value["points"] = {"index": 1, "met": True, "reason": "ok"}
    elif change == "non_string_reason":
        value["points"][0]["reason"] = None
    with pytest.raises(ValueError):
        validate_judgment(value, 1)
    result = asyncio.run(judge_answer(FakeClient([tool_input(value)]), "model", payload()))
    assert result["judge_valid"] is False
    assert result["judge_content"][0]["input"] == value
    assert result["judge_error_reason"]


def test_valid_failed_judgment_is_not_retried_or_converted_to_invalid():
    value = verdict()
    value["points"][0]["met"] = False
    client = FakeClient([tool_input(value)])
    result = asyncio.run(judge_answer(client, "model", payload()))
    assert result["judge_valid"] is True
    assert result["answer_passed"] is False
    assert "judge_error" not in result
    assert len(client.calls) == 1


def test_network_error_is_invalid_without_fabricated_raw_output():
    class BrokenClient:
        messages = None

        async def create(self, **kwargs):
            raise TimeoutError("provider unavailable")

    client = BrokenClient()
    client.messages = client
    result = asyncio.run(judge_answer(client, "model", payload()))
    assert not result["judge_valid"] and not result["answer_passed"]
    assert result["judge_error"] == "TimeoutError"
    assert "judge_raw" not in result


def make_retry_report(tmp_path):
    case = {"id": "case-one", "question": "需要提供什么？", "answerable": True,
            "required_points": ["说明所需字段"], "forbidden_claims": ["不虚称已授权"],
            "relevant_chunk_ids": ["one"]}
    valid_fail = {**deepcopy(case), "id": "case-two", "judge_valid": True,
                  "answer_passed": False, "judge": {"kept": "original valid failure"},
                  "chat": {"response": "原答案"}}
    invalid = {**case, "judge_valid": False, "answer_passed": False,
               "judge_content": [tool_input({})], "judge_raw": "original invalid output",
               "judge_error": "ValueError", "chat": {"response": "需提供系统和用途。"}}
    data = {"cases": [case, {**case, "id": "case-two"}], "corpus": []}
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(data), encoding="utf-8")
    report = {"dataset_sha256": sha(dataset.read_bytes()), "model": "same-model",
              "thinking_env": "false", "finished_at": "done", "summary": {},
              "judge_protocol": JUDGE_PROTOCOL, "judge_contract_sha256": JUDGE_CONTRACT_SHA256,
              "results": [invalid, valid_fail]}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    args = SimpleNamespace(retry_invalid_from=source, dataset=dataset, report=tmp_path / "retry.json")
    return args, data, report


def test_retry_uses_same_tool_helper_and_preserves_valid_failures(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_THINKING_ENABLED", "false")
    args, data, original = make_retry_report(tmp_path)
    before = args.retry_invalid_from.read_bytes()
    client = FakeClient([tool_input()])
    asyncio.run(retry_invalid(args, data, client, "same-model"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    assert args.retry_invalid_from.read_bytes() == before
    assert len(client.calls) == 1
    assert report["results"][1] == original["results"][1]
    assert report["results"][0]["judge_valid"] is True
    assert report["results"][0]["first_judge_attempt"]["judge_content"] == [tool_input({})]
    assert report["retry_count"] == 1
    assert client.closed


@pytest.mark.parametrize("legacy", [True, False])
def test_retry_rejects_different_output_protocol_or_contract_before_call(tmp_path, legacy):
    args, data, report = make_retry_report(tmp_path)
    if legacy:
        report.pop("judge_protocol")
    else:
        report["judge_contract_sha256"] = "different-contract"
    args.retry_invalid_from.write_text(json.dumps(report), encoding="utf-8")
    client = FakeClient([tool_input()])
    with pytest.raises(ValueError, match="Judge protocol differs"):
        asyncio.run(retry_invalid(args, data, client, "same-model"))
    assert client.calls == []
    assert not args.report.exists()
