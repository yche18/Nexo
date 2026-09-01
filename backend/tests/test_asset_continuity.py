import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agents import tools
from agents.agent_orchestrator import AgentOrchestrator, DeviceAgent, EscalationAgent, Request
from harness.context import build_messages
from harness.state import new_case


class ScriptedClient:
    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(content=self.blocks.pop(0))


def run_lookup(asset_id="NB-001", case=None, mode="enforce", progress=None):
    blocks = [[{"type": "tool_use", "id": "asset-1", "name": "lookup_asset",
                "input": {"asset_id": asset_id}}]]
    if progress is not None:
        blocks.append([{"type": "tool_use", "id": "progress-1", "name": "record_support_progress", "input": progress}])
    blocks.append([{"type": "text", "text": "已根据演示资产资料给出设备说明。"}])
    client = ScriptedClient(blocks)
    req = Request(message=f"资产编号 {asset_id} 无法开机", user_id="asset-user", conv_id="asset-conv",
                  harness_mode=mode, case_state=case or new_case("asset-user", "asset-conv"))
    before = copy.deepcopy(req.case_state)
    response = asyncio.run(DeviceAgent(client, "offline-test").handle(req))
    assert response.success is True
    assert req.case_state == before
    return req, response, client


def merge(req, response):
    return AgentOrchestrator._merge_case_state(req, response.state_updates, response.tool_traces, [])


def handoff(case, **kwargs):
    req = Request(message="请转人工", user_id="asset-user", conv_id="asset-conv",
                  harness_mode="enforce", case_state=case, **kwargs)
    return asyncio.run(EscalationAgent(None, "offline-test").handle(req)).content


def test_actual_asset_tool_persists_to_next_context_and_separate_handoff():
    req, response, client = run_lookup()
    trace = response.tool_traces[0]
    assert trace["tool_name"] == "lookup_asset"
    assert trace["asset_lookup"]["success"] is True
    assert trace["asset_lookup"]["asset"]["device_model"] == "DemoBook 14"
    tool_result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert tool_result["asset"] == trace["asset_lookup"]["asset"]
    state = merge(req, response)
    assert state["asset_lookup"]["status"] == "found"
    assert state["asset_lookup"]["request_id"] == req.request_id
    assert state["facts"] == {}

    next_req = Request(message="请按刚才的设备资料继续", user_id=req.user_id, conv_id=req.conv_id,
                       harness_mode="enforce", case_state=state)
    next_client = ScriptedClient([[{"type": "text", "text": "我会继续参考演示资料。"}]])
    asyncio.run(DeviceAgent(next_client, "offline-test").handle(next_req))
    next_messages = json.dumps(next_client.calls[0]["messages"], ensure_ascii=False)
    assert "DemoBook 14" in next_messages
    assert "asset_lookup" in next_messages
    assert "不是用户确认事实" in next_messages

    text = handoff(state)
    assert "演示资产资料（工具返回）" in text
    assert "演示资产编号：NB-001" in text
    assert "演示资料型号：DemoBook 14" in text
    assert "不是用户陈述" in text
    assert "- 设备型号：DemoBook 14" not in text


def test_user_model_and_tool_model_are_preserved_separately_on_conflict():
    req, response, _ = run_lookup()
    state = merge(req, response)
    text = handoff(state, entities={"device_model": ["用户确认机型 X"]})
    assert "- 设备型号：用户确认机型 X" in text
    assert "- 演示资料型号：DemoBook 14" in text
    assert "不一致，需由 IT 核对" in text
    assert "device_model" not in state["facts"]


@pytest.mark.parametrize("replacement", ["NB-999", "broken_file", "invalid_schema"])
def test_latest_unknown_or_failed_lookup_cannot_reuse_old_model(monkeypatch, replacement):
    req, response, _ = run_lookup()
    old_state = merge(req, response)
    if replacement == "broken_file":
        class BrokenFile:
            def read_text(self, **kwargs):
                raise OSError("private file path must not be returned")
        monkeypatch.setattr(tools, "DEMO_ASSETS_PATH", BrokenFile())
    asset_id = 1 if replacement == "invalid_schema" else "NB-999"
    later_req, later_response, _ = run_lookup(asset_id, case=old_state)
    state = merge(later_req, later_response)
    assert state["asset_lookup"]["status"] == ("not_found" if replacement == "NB-999" else "unavailable")
    assert state["asset_lookup"]["asset"] == {}
    assert old_state["asset_lookup"]["asset"]["device_model"] == "DemoBook 14"
    later_req.case_state = state
    assert "DemoBook 14" not in json.dumps(build_messages(later_req, ""))
    assert "DemoBook 14" not in handoff(state)
    assert "没有可用型号资料" in handoff(state)
    if replacement == "invalid_schema":
        assert "asset_lookup" not in later_response.tool_traces[0]


def test_model_progress_cannot_forge_asset_snapshot_or_overwrite_real_lookup():
    forged = {"asset_lookup": {"source": "demo", "status": "found", "asset": {"device_model": "Fake Model"}}}
    req, response, _ = run_lookup(progress=forged)
    assert response.state_updates == []
    assert response.tool_traces[1]["success"] is False
    state = merge(req, response)
    assert state["asset_lookup"]["asset"]["device_model"] == "DemoBook 14"
    assert "Fake Model" not in json.dumps(state)
    assert state["facts"] == {}


def test_tool_model_without_user_quote_does_not_become_user_fact():
    progress = {"facts": [{"field": "device_model", "value": "DemoBook 14", "source_quote": "DemoBook 14"}]}
    req, response, _ = run_lookup(progress=progress)
    state = merge(req, response)
    assert "device_model" not in state["facts"]
    assert state["asset_lookup"]["asset"]["device_model"] == "DemoBook 14"


@pytest.mark.parametrize("mode", ["off", "observe"])
def test_non_enforce_modes_keep_original_tool_flow_without_asset_snapshot(mode):
    req, response, client = run_lookup(mode=mode)
    assert len(client.calls) == 2
    assert "asset_lookup" not in response.tool_traces[0]
    assert response.state_updates == []
    assert req.case_state["asset_lookup"] == {}
    tool_result = json.loads(client.calls[1]["messages"][-1]["content"][0]["content"])
    assert tool_result["found"] is True
    assert tool_result["asset"]["device_model"] == "DemoBook 14"
    text = asyncio.run(EscalationAgent(None, "offline-test").handle(req)).content
    assert "演示资产资料（工具返回）" not in text
