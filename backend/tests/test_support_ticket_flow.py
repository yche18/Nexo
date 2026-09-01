import asyncio
import copy
import json

import pytest

from agents.agent_orchestrator import AccessAgent, AgentOrchestrator, EscalationAgent, Request
from agents.tools import access_field_values, check_access_fields, create_ticket, parse_ticket_action
from core.intent_recognizer import IntentCategory
from core.ticket_store import TicketStore
from harness.state import new_case, prepare_case
from test_ticket_workflow import request, router, run


def fact(value):
    return {"value": value, "source": "user_reported", "source_quote": value}


def application_case():
    state = new_case("ticket-user", "ticket-conv")
    state["request_kind"] = "access_request"
    state["facts"] = {"target_system": fact("项目系统"), "purpose": fact("查看需求"),
                      "requested_access": fact("只读")}
    return state


def test_access_fields_use_confirmed_prior_turn_without_mutating_routing_entities():
    state = application_case()
    state["facts"].pop("purpose")
    req = request("用于查看需求", case=state, entities={"purpose": ["查看需求"]})
    before = copy.deepcopy(req)
    checked = check_access_fields(req, {})
    packet = json.loads(AccessAgent(None, "offline")._build_role_packet(req))
    assert checked["missing_fields"] == []
    assert packet["application_fields"]["missing_fields"] == []
    assert req.entities == before.entities and req.case_state == before.case_state


@pytest.mark.parametrize("problem", ["candidate", "conflict", "different_current"])
def test_access_fields_do_not_silently_accept_unconfirmed_or_conflicting_values(problem):
    state = application_case()
    entities = {}
    if problem == "candidate":
        state["facts"]["target_system"]["source"] = "model_generated"
    elif problem == "conflict":
        state["conflicts"] = [{"field": "target_system"}]
        entities = {"target_system": ["项目系统"]}
    else:
        entities = {"target_system": ["邮箱"]}
    result = check_access_fields(request("目标系统是邮箱", case=state, entities=entities), {})
    assert "target_system" in result["missing_fields"]


def test_explicit_current_correction_can_be_checked_without_overwriting_saved_state():
    req = request("更正为邮箱", case=application_case(), entities={"target_system": ["邮箱"]})
    assert access_field_values(req)["target_system"] == ["邮箱"]
    assert req.case_state["facts"]["target_system"]["value"] == "项目系统"


def test_observe_mode_keeps_original_current_turn_field_check():
    result = check_access_fields(request(case=application_case(), harness_mode="observe"), {})
    assert result["missing_fields"] == ["target_system", "purpose"]


def test_application_without_fault_description_creates_and_queries_complete_request(tmp_path):
    store = TicketStore(tmp_path / "tickets.sqlite3")
    orch = router(store)
    result = run(orch, request(case=application_case()))
    assert result.execution_success
    ticket_id = result.case_state["ticket_receipt"]["ticket_id"]
    saved = store.get_ticket(ticket_id, "ticket-user", "ticket-conv")["ticket"]
    assert all(value in saved["summary"] for value in ("权限申请", "项目系统", "查看需求", "只读"))
    assert saved["snapshot"]["user_facts"]["purpose"]["value"] == "查看需求"
    queried = run(orch, request("查询当前工单", case=result.case_state))
    assert ticket_id in queried.response and "待人工处理" in queried.response
    assert not queried.escalated
    handoff = asyncio.run(EscalationAgent(None, "offline").handle(
        request("转人工", case=result.case_state, intent=IntentCategory.HUMAN_HANDOFF)))
    assert ticket_id in handoff.content and "待人工处理" in handoff.content
    assert "创建工单" not in handoff.content and "尚未通知" not in handoff.content


@pytest.mark.parametrize("field", ["target_system", "purpose"])
def test_application_requires_business_fields_even_with_generic_issue(tmp_path, field):
    state = application_case(); state["facts"].pop(field)
    state["facts"]["issue_description"] = fact("我要申请权限")
    result = asyncio.run(create_ticket(request(case=state), TicketStore(tmp_path / "tickets.sqlite3")))
    assert result["success"] is False and "补充" in result["error"]


@pytest.mark.parametrize("field", ["target_system", "purpose", "requested_access"])
def test_application_conflicts_block_submission(tmp_path, field):
    state = application_case();state["conflicts"] = [{"field": field}]
    result = asyncio.run(create_ticket(request(case=state), TicketStore(tmp_path / "tickets.sqlite3")))
    assert result["success"] is False and "冲突" in result["error"]


def test_incidental_purpose_does_not_change_fault_ticket_kind(tmp_path):
    state = application_case();state["request_kind"] = ""
    state["facts"]["issue_description"] = fact("VPN连接超时")
    result = run(router(TicketStore(tmp_path / "tickets.sqlite3")), request(case=state))
    assert result.execution_success
    assert result.case_state["request_kind"] == ""


def test_existing_permission_intent_records_kind_and_new_topic_clears_it():
    original = new_case("ticket-user", "ticket-conv")
    req = request("申请权限", case=original, intent=IntentCategory.PERMISSION_REQUEST)
    state = AgentOrchestrator._merge_case_state(req, [], [], [])
    assert state["request_kind"] == "access_request" and original["request_kind"] == ""
    assert prepare_case(state, "换个问题，电脑无法开机")["request_kind"] == ""


@pytest.mark.parametrize("message", ["可以", "可以，帮我提交", "请转人工", "不用创建工单", "如果需要再建单"])
def test_no_generic_confirmation_or_handoff_authorizes_creation(message):
    assert parse_ticket_action(message) is None


@pytest.mark.parametrize("message", ["查询当前工单", "查询我的工单进度", "查一下刚才那张单"])
def test_current_ticket_query_without_receipt_is_not_fabricated(tmp_path, message):
    result = run(router(TicketStore(tmp_path / "tickets.sqlite3")), request(message))
    assert not result.execution_success and "完整工单编号" in result.response
    assert not result.escalated and not result.case_state["ticket_receipt"]


def test_current_ticket_query_checks_persisted_store_scope(tmp_path):
    store = TicketStore(tmp_path / "tickets.sqlite3");orch = router(store)
    created = run(orch, request(case=application_case()))
    other = copy.deepcopy(created.case_state);other['user_id'] = 'other-user'
    result = run(orch, request("查询当前工单", case=other, user_id="other-user"))
    assert "未找到" in result.response
    assert not result.escalated


def test_handoff_without_ticket_stops_at_waiting_human_without_suggesting_creation():
    response = asyncio.run(EscalationAgent(None, "offline").handle(
        request("请转人工", intent=IntentCategory.HUMAN_HANDOFF)))
    assert "待人工处理" in response.content
    assert "创建工单" not in response.content
    assert "尚未通知" not in response.content


@pytest.mark.parametrize("intent", [IntentCategory.ACCOUNT, IntentCategory.ACCOUNT_SECURITY])
def test_account_recovery_does_not_require_permission_application_purpose(intent):
    req = request("企业邮箱忘记密码", intent=intent, entities={"target_system": ["企业邮箱"]})
    result = check_access_fields(req, {})
    packet = json.loads(AccessAgent(None, "offline")._build_role_packet(req))
    assert result["missing_fields"] == []
    assert packet["application_fields"]["missing_fields"] == []
    assert "账号问题" in packet["application_fields"]["check_scope"]


def test_mixed_application_keeps_required_fields_when_current_turn_is_account():
    state = application_case();state["facts"].pop("purpose")
    req = request("账号还需要恢复", case=state, intent=IntentCategory.ACCOUNT)
    assert check_access_fields(req, {})["missing_fields"] == ["purpose"]
