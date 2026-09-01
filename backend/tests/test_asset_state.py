import asyncio
import copy
import json

import pytest

from harness.context import case_context
from harness.state import (
    ASSET_FIELDS, FACT_FIELDS, PROGRESS_DESCRIPTION, PROGRESS_SCHEMA, CaseStore,
    apply_asset_lookup, apply_progress, new_case, prepare_case,
)

from test_harness_state import redis_stub


ASSET = {"asset_id": "NB-001", "device_type": "笔记本", "device_model": "DemoBook 14",
         "operating_system": "Windows 11", "configuration": "16GB 内存，512GB SSD",
         "warranty_until": "2028-06-30"}


def trace(asset_id="NB-001", found=True):
    return {"tool_name": "lookup_asset", "input": {"asset_id": asset_id},
            "success": True, "result_success": True,
            "asset_lookup": {"success": True, "source": "demo", "found": found,
                             "asset_id": asset_id.strip().upper(),
                             "asset": dict(ASSET) if found else None}}


def saved_case():
    return apply_asset_lookup(new_case("u", "c"), trace(), "lookup-1")


def test_asset_snapshot_is_separate_whitelisted_copy_with_backend_source():
    original = new_case("u", "c")
    original["facts"] = {"device_model": {"value": "用户自己的型号", "source": "user_reported"}}
    event = trace(" nb-001 ")
    event["asset_lookup"]["asset"]["owner"] = "must not persist"
    before, event_before = copy.deepcopy(original), copy.deepcopy(event)
    updated = apply_asset_lookup(original, event, "r1")
    assert original == before and event == event_before
    assert updated["facts"] == original["facts"]
    assert updated["asset_lookup"] == {"asset_id": "NB-001", "source": "demo", "status": "found",
                                       "asset": ASSET, "request_id": "r1"}
    assert set(updated["asset_lookup"]["asset"]) == set(ASSET_FIELDS)
    event["asset_lookup"]["asset"]["device_model"] = "mutated"
    assert updated["asset_lookup"]["asset"]["device_model"] == "DemoBook 14"


def test_unknown_id_replaces_previous_asset_without_changing_user_facts():
    case = saved_case()
    case["facts"] = {"device_type": {"value": "笔记本", "source": "user_reported"}}
    updated = apply_asset_lookup(case, trace("NB-999", found=False), "r2")
    assert updated["asset_lookup"] == {"asset_id": "NB-999", "source": "demo", "status": "not_found",
                                      "asset": {}, "request_id": "r2"}
    assert case["asset_lookup"]["asset"] == ASSET
    assert updated["facts"] == case["facts"]


@pytest.mark.parametrize("failure", [
    "execution", "result", "payload", "source", "returned_id", "nested_id",
    "missing_field", "invalid_field", "missing_payload", "schema_error", "found_not_bool",
])
def test_failed_or_inconsistent_lookup_clears_stale_asset(failure):
    event = trace()
    if failure == "execution":
        event["success"] = False
    elif failure == "result":
        event["result_success"] = False
    elif failure == "payload":
        event["asset_lookup"]["success"] = False
    elif failure == "source":
        event["asset_lookup"]["source"] = "model"
    elif failure == "returned_id":
        event["asset_lookup"]["asset_id"] = "PC-001"
    elif failure == "nested_id":
        event["asset_lookup"]["asset"]["asset_id"] = "PC-001"
    elif failure == "missing_field":
        event["asset_lookup"]["asset"].pop("configuration")
    elif failure == "invalid_field":
        event["asset_lookup"]["asset"]["configuration"] = {"command": "bad"}
    elif failure == "missing_payload":
        event.pop("asset_lookup")
    elif failure == "schema_error":
        event = {"tool_name": "lookup_asset", "input": {}, "success": False, "result_success": None}
    elif failure == "found_not_bool":
        event["asset_lookup"]["found"] = "true"
    updated = apply_asset_lookup(saved_case(), event, "r2")
    assert updated["asset_lookup"]["status"] == "unavailable"
    assert updated["asset_lookup"]["asset"] == {}
    assert updated["asset_lookup"]["request_id"] == "r2"


def test_other_tool_cannot_write_asset_snapshot():
    case = saved_case()
    event = trace("PC-001")
    event["tool_name"] = "record_support_progress"
    assert apply_asset_lookup(case, event, "r2") == case


@pytest.mark.parametrize("message", ["设备是 PC-001", "现在看 pr-001 的问题", "NB-001 正常，PR-001 卡纸"])
def test_explicit_different_demo_identifier_clears_only_asset_snapshot(message):
    case = saved_case()
    case["version"] = 3
    case["facts"] = {"issue_description": {"value": "无法开机", "source": "user_reported"}}
    updated = prepare_case(case, message)
    assert updated["asset_lookup"] == {}
    assert updated["case_id"] == case["case_id"]
    assert updated["version"] == 3
    assert updated["facts"] == case["facts"]
    assert case["asset_lookup"]["status"] == "found"


@pytest.mark.parametrize("message", ["还是 nb-001 的问题", "依然无法开机", "错误码 PR-001-extra", "XPC-001"])
def test_continuation_or_partial_identifier_keeps_previous_snapshot(message):
    case = saved_case()
    assert prepare_case(case, message)["asset_lookup"] == case["asset_lookup"]


def test_new_topic_clears_asset_and_existing_case_data():
    case = saved_case()
    case["facts"] = {"device_model": {"value": "旧型号", "source": "user_reported"}}
    updated = prepare_case(case, "换个问题，帮我看看打印机")
    assert updated["asset_lookup"] == {} and updated["facts"] == {}
    assert updated["case_id"] != case["case_id"]


def test_progress_cannot_forge_backend_asset_lookup():
    assert "asset_lookup" not in PROGRESS_SCHEMA["properties"]
    assert "asset_lookup" not in FACT_FIELDS
    with pytest.raises(ValueError, match="未知字段"):
        apply_progress(new_case("u", "c"), {"asset_lookup": {"status": "found", "asset": ASSET}},
                       "请查询 NB-001", "r1")
    assert "proposed 步骤只填 step_id、description、status" in PROGRESS_DESCRIPTION
    assert "不带 source_quote、result 或 result_quote" in PROGRESS_DESCRIPTION
    assert "资产工具结果由后端独立保存" in PROGRESS_DESCRIPTION


def test_case_context_keeps_tool_and_user_sources_distinct():
    case = saved_case()
    case["facts"] = {"device_model": {"value": "用户说的另一型号", "source": "user_reported"}}
    text = case_context(case)
    assert "工具演示资料" in text and "不是用户确认事实或实时检测" in text
    assert "不能用工具资料覆盖用户 facts" in text
    assert "DemoBook 14" in text and "用户说的另一型号" in text
    payload = json.loads(text[text.index("{"):])
    assert payload["asset_lookup"]["source"] == "demo"
    assert payload["facts"]["device_model"]["source"] == "user_reported"


@pytest.mark.parametrize("status", ["found", "not_found", "unavailable"])
def test_asset_snapshot_survives_case_store_round_trip_with_ttl_and_isolation(redis_stub, status):
    async def scenario():
        store = CaseStore(redis_stub)
        case = await store.load("asset-user", "asset-conv")
        event = trace("NB-999", found=False) if status == "not_found" else trace()
        if status == "unavailable":
            event["success"] = False
        updated = apply_asset_lookup(case, event, "lookup-persist")
        saved = await store.save(updated, case["version"])
        # Read through a fresh CaseStore instance, using only serialized Redis data.
        loaded = await CaseStore(redis_stub).load("asset-user", "asset-conv")
        assert loaded == saved
        assert loaded["asset_lookup"]["status"] == status
        assert loaded["asset_lookup"]["request_id"] == "lookup-persist"
        assert loaded["asset_lookup"]["asset"] == (ASSET if status == "found" else {})
        assert loaded["facts"] == {}
        assert loaded["version"] == 1 and updated["version"] == 0
        assert redis_stub.expiries[store.key("asset-user", "asset-conv")] == 86400
        other_user = await store.load("other-user", "asset-conv")
        other_conversation = await store.load("asset-user", "other-conv")
        assert other_user["asset_lookup"] == other_conversation["asset_lookup"] == {}
    asyncio.run(scenario())


@pytest.mark.parametrize("message", [
    "资产编号改为 ASSET-999，请转人工", "资产编号：ASSET-999，请转人工",
    "资产编号是 asset-999，请转人工", "资产编号更正为 INV_999，请转人工",
])
def test_explicit_unknown_identifier_handoff_does_not_keep_previous_model(message):
    from agents.agent_orchestrator import EscalationAgent, Request
    from core.intent_recognizer import IntentCategory
    previous = saved_case()
    previous["facts"] = {"issue_description": {"value": "无法开机", "source": "user_reported"}}
    prepared = prepare_case(previous, message)
    assert prepared["asset_lookup"] == {}
    assert prepared["facts"] == previous["facts"]
    assert prepared["case_id"] == previous["case_id"]
    request = Request(message=message, user_id="u", conv_id="c", intent=IntentCategory.HUMAN_HANDOFF,
                      harness_mode="enforce", case_state=prepared)
    response = asyncio.run(EscalationAgent(None, "offline").handle(request))
    assert response.success
    assert "DemoBook 14" not in response.content
    assert "NB-001" not in response.content
    assert "无法开机" in response.content
    assert previous["asset_lookup"]["asset"] == ASSET


@pytest.mark.parametrize("message", ["资产编号是 nb-001，仍然无法开机", "资产编号：NB-001", "还是刚才的问题"])
def test_same_explicit_identifier_or_no_identifier_keeps_asset(message):
    previous = saved_case()
    assert prepare_case(previous, message)["asset_lookup"] == previous["asset_lookup"]
