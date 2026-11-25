import asyncio
import copy
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import DeviceAgent, NetworkAgent, Request
from core.intent_recognizer import IntentCategory
from harness.state import PROGRESS_SCHEMA, apply_progress, handoff_fields, new_case, prepare_case


ENVIRONMENT = {
    "operating_system": "Windows 11",
    "client_version": "6.2.1",
    "network_type": "公司 Wi-Fi",
    "impact_scope": "同部门多人",
    "device_type": "打印机",
    "device_model": "HP M404",
}


def test_environment_facts_keep_provenance_and_reach_handoff():
    message = "，".join(ENVIRONMENT.values())
    update = {"facts": [{"field": key, "value": value, "source_quote": value}
                        for key, value in ENVIRONMENT.items()]}
    original = new_case("u", "c")
    case = apply_progress(original, update, message, "r1")
    assert original["facts"] == {}
    assert handoff_fields(case) == ENVIRONMENT
    schema_fields = PROGRESS_SCHEMA["properties"]["facts"]["items"]["properties"]["field"]["enum"]
    assert all(key in schema_fields for key in ENVIRONMENT)
    assert all(fact["source"] == "user_reported" for fact in case["facts"].values())
    assert prepare_case(case, "换个问题")["facts"] == {}


def test_network_fact_conflict_needs_explicit_correction():
    case = apply_progress(new_case("u", "c"), {"facts": [
        {"field": "network_type", "value": "公司 Wi-Fi", "source_quote": "公司 Wi-Fi"}]},
        "公司 Wi-Fi", "r1")
    conflict = apply_progress(case, {"facts": [
        {"field": "network_type", "value": "手机热点", "source_quote": "手机热点"}]},
        "手机热点", "r2")
    assert "network_type" not in handoff_fields(conflict)
    corrected = apply_progress(conflict, {"facts": [
        {"field": "network_type", "value": "手机热点", "source_quote": "手机热点"}]},
        "说错了，实际是手机热点", "r3")
    assert handoff_fields(corrected)["network_type"] == "手机热点"
    assert corrected["conflicts"] == []


def test_inferred_impact_scope_is_not_a_confirmed_fact():
    case = apply_progress(new_case("u", "c"), {"facts": [
        {"field": "impact_scope", "value": "全公司", "source_quote": "两个同事也断网了"}]},
        "两个同事也断网了", "r1")
    assert "impact_scope" not in handoff_fields(case)
    assert case["candidates"][0]["value"] == "全公司"


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = self
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(content=self.responses.pop(0), stop_reason="end_turn")


@pytest.mark.parametrize("agent_class,intent", [
    (NetworkAgent, IntentCategory.NETWORK_ISSUE),
    (DeviceAgent, IntentCategory.DEVICE_FAILURE),
])
def test_new_agents_record_progress_and_preserve_request_state(agent_class, intent):
    update = {"facts": [{"field": "device_model", "value": "HP M404", "source_quote": "HP M404"}]}
    client = Client([
        [{"type": "tool_use", "id": "t1", "name": "record_support_progress", "input": update}],
        [{"type": "text", "text": "已记录设备信息。"}],
    ])
    req = Request(message="设备型号是 HP M404", user_id="u", conv_id="c",
                  intent=intent, harness_mode="enforce", case_state=new_case("u", "c"))
    before = copy.deepcopy(req.case_state)
    result = asyncio.run(agent_class(client, "test").handle(req))
    assert result.success
    assert result.state_updates == [update]
    assert len(client.calls) == 2
    assert req.case_state == before
    assert req.entities == {}


@pytest.mark.parametrize("agent_class", [NetworkAgent, DeviceAgent])
def test_new_agents_report_missing_progress_without_forcing_extra_model_calls(agent_class):
    client = Client([[{"type": "text", "text": "请补充具体问题。"}]])
    req = Request(message="请继续排查", user_id="u", conv_id="c",
                  intent=IntentCategory.OTHER, harness_mode="enforce")
    result = asyncio.run(agent_class(client, "test").handle(req))
    assert result.success
    assert any(event["event"] == "progress_not_recorded" for event in result.harness_events)
    assert len(client.calls) == 1


def test_reported_paper_removal_preserves_done_action_and_failed_outcome():
    message = "我已经取出卡纸了，还是无法打印"
    result = apply_progress(new_case("u", "c"), {"steps": [{
        "step_id": "paper_removal", "description": "取出卡纸", "status": "user_reported_done",
        "source_quote": message, "result": "还是无法打印",
    }]}, message, "r1")
    assert result["steps"][0]["status"] == "user_reported_done"
    assert result["steps"][0]["result"] == "还是无法打印"
    assert result["steps"][0]["source_quote"] == message
    assert result["progress_warnings"] == []


@pytest.mark.parametrize("message", ["还没有取纸", "如果取纸了还是无法打印怎么办"])
def test_unperformed_device_action_is_not_recorded_as_done(message):
    result = apply_progress(new_case("u", "c"), {"steps": [{
        "step_id": "paper_removal", "description": "取纸", "status": "user_reported_done",
        "source_quote": message, "result": "",
    }]}, message, "r1")
    assert not any(step["status"] == "user_reported_done" for step in result["steps"])
    assert result["progress_warnings"][0]["kind"] == "invalid_step"
