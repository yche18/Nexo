import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import BaseAgent, DeviceAgent, Request
from agents import tools
from agents.tools import device_tools, lookup_asset


@pytest.mark.parametrize("asset_id,device_type", [
    ("NB-001", "笔记本"), ("PC-001", "台式机"), ("PR-001", "打印机"),
])
def test_lookup_public_demo_asset(asset_id, device_type):
    result = lookup_asset(None, {"asset_id": asset_id})
    assert result["success"] is True
    assert result["found"] is True
    assert result["source"] == "demo"
    assert result["asset_id"] == asset_id
    assert result["asset"]["asset_id"] == asset_id
    assert result["asset"]["device_type"] == device_type
    assert set(result["asset"]) == {"asset_id", "device_type", "device_model", "operating_system", "configuration", "warranty_until"}
    assert "虚构" in result["notice"]


@pytest.mark.parametrize("asset_id", ["nb-001", " NB-001 ", "\tNb-001\n"])
def test_lookup_normalizes_case_and_surrounding_whitespace(asset_id):
    result = lookup_asset(None, {"asset_id": asset_id})
    assert result["found"] is True
    assert result["asset_id"] == "NB-001"


@pytest.mark.parametrize("asset_id", ["NB-999", "NB-00", "NB-001-extra", "DemoBook 14"])
def test_unknown_or_partial_identifier_is_not_a_service_error(asset_id):
    result = lookup_asset(None, {"asset_id": asset_id})
    assert result["success"] is True
    assert result["found"] is False
    assert result["asset"] is None
    assert result["source"] == "demo"


@pytest.mark.parametrize("args", [{}, {"asset_id": ""}, {"asset_id": " \t"}, {"asset_id": None}, {"asset_id": 1}, {"asset_id": []}, None])
def test_invalid_identifier_returns_clear_failure(args):
    result = lookup_asset(None, args)
    assert result["success"] is False
    assert "非空字符串" in result["error"]


@pytest.mark.parametrize("payload", ["not json", "{}", "[null]", '[{"asset_id": "NB-001"}]'])
def test_bad_fixture_returns_failure_without_leaking_path(tmp_path, monkeypatch, payload):
    path = tmp_path / "private-demo-assets.json"
    path.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(tools, "DEMO_ASSETS_PATH", path)
    result = lookup_asset(None, {"asset_id": "NB-001"})
    assert result["success"] is False
    assert "暂不可用" in result["error"]
    assert str(path) not in json.dumps(result, ensure_ascii=False)
    assert path.name not in json.dumps(result, ensure_ascii=False)


def test_missing_fixture_returns_failure_without_leaking_path(tmp_path, monkeypatch):
    path = tmp_path / "missing-demo-assets.json"
    monkeypatch.setattr(tools, "DEMO_ASSETS_PATH", path)
    result = lookup_asset(None, {"asset_id": "NB-001"})
    assert result["success"] is False
    assert str(path) not in json.dumps(result)
    assert path.name not in json.dumps(result)


def test_tool_schema_requires_identifier_and_accepts_no_owner_or_path():
    spec = device_tools()["lookup_asset"]
    assert spec.handler is lookup_asset
    assert spec.input_schema["required"] == ["asset_id"]
    assert spec.input_schema["properties"]["asset_id"]["type"] == "string"
    BaseAgent._validate_tool_input(spec, {"asset_id": "NB-001"})
    for args in ({}, {"asset_id": 1}, {"asset_id": "NB-001", "owner": "someone"},
                 {"asset_id": "NB-001", "path": "another.json"}):
        with pytest.raises(ValueError):
            BaseAgent._validate_tool_input(spec, args)


@pytest.mark.parametrize("asset_id,found", [(" nb-001 ", True), ("NB-999", False)])
def test_asset_result_returns_through_device_harness_tool_loop(asset_id, found):
    class ToolClient:
        def __init__(self):
            self.calls = []
            self.messages = self

        async def create(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            if len(self.calls) == 1:
                return SimpleNamespace(content=[{
                    "type": "tool_use", "id": "asset-test-1", "name": "lookup_asset",
                    "input": {"asset_id": asset_id},
                }])
            result_block = kwargs["messages"][-1]["content"][0]
            assert result_block["type"] == "tool_result"
            assert result_block["tool_use_id"] == "asset-test-1"
            result = json.loads(result_block["content"])
            assert result["success"] is True
            assert result["found"] is found
            assert result["source"] == "demo"
            if found:
                assert result["asset"]["device_model"] == "DemoBook 14"
            else:
                assert result["asset"] is None
            return SimpleNamespace(content=[{"type": "text", "text": "已根据公开虚构演示资产查询结果说明下一步。"}])

    client = ToolClient()
    agent = DeviceAgent(client, "offline-test")
    req = Request(message=f"请查询资产 {asset_id}", user_id="demo", conv_id="asset-test", harness_mode="enforce")
    response = asyncio.run(agent.handle(req))
    assert response.success is True
    assert response.tools_used == ["lookup_asset"]
    assert response.tool_traces[0]["tool_name"] == "lookup_asset"
    assert response.tool_traces[0]["success"] is True
    assert response.tool_traces[0]["result_success"] is True
    assert len(client.calls) == 2
    assert "lookup_asset" in {tool["name"] for tool in client.calls[0]["tools"]}
    assert "record_support_progress" in {tool["name"] for tool in client.calls[0]["tools"]}
    assert any(event["event"] == "tool_finished" for event in response.harness_events)
