import asyncio
import copy
import json
import sqlite3
import threading
from collections import deque

import pytest

from agents.agent_orchestrator import AgentOrchestrator, AgentType, EscalationAgent, Request
from agents.tools import create_ticket, device_tools, escalation_tools, get_ticket, parse_ticket_action
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.ticket_store import TicketStore
from harness.context import case_context
from harness.state import apply_progress, new_case


def case_state():
    case = new_case("ticket-user", "ticket-conv")
    case["facts"]["issue_description"] = {"value": "笔记本无法开机", "source": "user_reported", "source_quote": "笔记本无法开机"}
    case["candidates"] = [{"field": "device_model", "value": "unverified-model"}]
    case["pending_questions"] = ["是否有电源指示？"]
    case["steps"] = [{"step_id": "power", "description": "检查外部供电", "status": "proposed"}]
    return case


def request(message="请为当前问题创建工单", case=None, **kwargs):
    values = dict(message=message, user_id="ticket-user", conv_id="ticket-conv",
                  harness_mode="enforce", case_state=case if case is not None else case_state(),
                  intent=IntentCategory.GREETING, urgency=UrgencyLevel.LOW,
                  history=[{"role": "user", "content": "full-history-not-for-ticket"}])
    values.update(kwargs)
    return Request(**values)


def router(store):
    result = AgentOrchestrator.__new__(AgentOrchestrator)
    result._ticket_store = store
    result._recent_tool_traces = deque()
    return result


def run(orchestrator, req):
    # A ticket command must not require an intent recognizer or model client.
    return asyncio.run(orchestrator.run(req))


@pytest.mark.parametrize("message", [
    "请为当前问题创建工单", "帮我创建报修单", "帮我创建工单。", "请创建工单！", "创建工单",
])
def test_complete_creation_commands_are_recognized(message):
    assert parse_ticket_action(message) == {"action": "create_ticket"}


@pytest.mark.parametrize("message", [
    "转人工", "请不要创建工单", "别帮我创建报修单", "请创建工单？", "可以创建工单吗",
    "如果排查失败就创建工单", "创建工单，然后帮我申请权限", "请为当前问题创建工单，先不要提交",
    "怎么创建工单", "他说帮我创建报修单", "请创建工单但不要保存", "查询工单 NEXO-123 并创建工单",
])
def test_denial_questions_conditions_and_mixed_text_never_authorize_creation(tmp_path, message):
    store = TicketStore(tmp_path / "tickets.sqlite3")
    assert parse_ticket_action(message) is None
    result = asyncio.run(create_ticket(request(message), store))
    assert result["success"] is False
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_real_store_create_repeat_query_and_reload_keep_original_ticket(tmp_path):
    store = TicketStore(tmp_path / "tickets.sqlite3")
    orchestrator = router(store)
    req = request()
    original_case = copy.deepcopy(req.case_state)
    result = run(orchestrator, req)
    receipt = result.case_state["ticket_receipt"]
    ticket_id = receipt["ticket_id"]
    assert ticket_id.startswith("NEXO-")
    assert result.tools_used == ["create_ticket"]
    assert result.tool_traces[0]["created"] is True
    assert result.tool_traces[0]["result_success"] is True
    assert "尚未通知 IT" in result.response
    assert req.case_state == original_case
    assert result.case_state["facts"] == original_case["facts"]
    assert ticket_id in case_context(result.case_state)
    stored = store.get_ticket(ticket_id, req.user_id, req.conv_id)["ticket"]
    assert stored["submission_text"] == req.message
    assert "full-history-not-for-ticket" not in json.dumps(stored["snapshot"])
    assert "unverified-model" not in json.dumps(stored["snapshot"])
    assert set(stored["snapshot"]) == {"user_facts", "asset_lookup", "steps", "pending_questions", "conflicts"}

    # SQLite committed, but Redis save failed: the caller still has the old case.
    retry_req = request("帮我创建报修单", case=original_case)
    retry_req.case_state["facts"]["issue_description"]["value"] = "后来补充的故障现象"
    repeated = run(orchestrator, retry_req)
    assert repeated.case_state["ticket_receipt"]["ticket_id"] == ticket_id
    assert repeated.tool_traces[0]["created"] is False
    assert repeated.case_state["ticket_receipt"]["request_id"] == req.request_id
    assert store.get_ticket(ticket_id, req.user_id, req.conv_id)["ticket"] == stored

    reloaded_router = router(TicketStore(store.path))
    query_req = request(f"查询工单 {ticket_id}", case=result.case_state,
                        intent=IntentCategory.HUMAN_HANDOFF, urgency=UrgencyLevel.CRITICAL)
    queried = run(reloaded_router, query_req)
    assert queried.escalated is False
    assert queried.agent_type is AgentType.GENERAL
    assert queried.tools_used == ["get_ticket"]
    assert queried.case_state == query_req.case_state
    assert stored["created_at"] in queried.response
    assert "笔记本无法开机" in queried.response


@pytest.mark.parametrize("mode", ["off", "observe"])
@pytest.mark.parametrize("message", ["请为当前问题创建工单", "查询工单 NEXO-123"])
def test_explicit_commands_are_disabled_without_model_or_storage_calls(mode, message):
    class NoStoreCalls:
        def create_ticket(self, *args):
            raise AssertionError("must not write")
        def get_ticket(self, *args):
            raise AssertionError("must not query")
    result = run(router(NoStoreCalls()), request(message, harness_mode=mode))
    assert "当前未开启" in result.response
    assert result.tools_used == []
    assert result.tool_traces == []
    assert result.escalated is False


@pytest.mark.parametrize("invalid", ["missing_case", "missing_issue", "wrong_scope", "unverified_issue"])
def test_creation_requires_stable_scope_and_user_reported_problem(tmp_path, invalid):
    state = case_state()
    if invalid == "missing_case":
        state = {}
    elif invalid == "missing_issue":
        state["facts"] = {}
    elif invalid == "wrong_scope":
        state["user_id"] = "another-user"
    else:
        state["facts"]["issue_description"]["source"] = "model_generated"
    store = TicketStore(tmp_path / "tickets.sqlite3")
    result = run(router(store), request(case=state))
    assert result.execution_success is False
    assert not result.case_state.get("ticket_receipt")
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_query_other_case_does_not_replace_current_receipt_or_escalate(tmp_path):
    store = TicketStore(tmp_path / "tickets.sqlite3")
    orchestrator = router(store)
    first = run(orchestrator, request())
    second = run(orchestrator, request(case=case_state()))
    current = copy.deepcopy(second.case_state)
    query = run(orchestrator, request(f"查询工单 {first.case_state['ticket_receipt']['ticket_id']}", case=current))
    assert query.case_state == current
    assert query.escalated is False
    unknown = run(orchestrator, request("查询工单 NEXO-UNKNOWN", case=current))
    assert "未找到" in unknown.response
    assert unknown.case_state == current
    assert unknown.escalated is False


def test_failed_store_does_not_fabricate_receipt_and_retry_uses_same_case(tmp_path):
    real_store = TicketStore(tmp_path / "tickets.sqlite3")
    class FailAfterCommit:
        def __init__(self):
            self.failed = False
        def create_ticket(self, *args):
            result = real_store.create_ticket(*args)
            if not self.failed:
                self.failed = True
                raise OSError("private/storage/path")
            return result
    orchestrator = router(FailAfterCommit())
    req = request()
    failed = run(orchestrator, req)
    assert failed.execution_success is False
    assert not failed.case_state.get("ticket_receipt")
    assert "private/storage/path" not in failed.response
    assert "ticket_id" not in failed.tool_traces[0]
    retried = run(orchestrator, request(case=req.case_state))
    assert retried.tool_traces[0]["created"] is False
    assert retried.case_state["ticket_receipt"]["ticket_id"]
    with sqlite3.connect(real_store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 1


def test_model_progress_cannot_invent_ticket_receipt_and_tools_are_not_exposed():
    state = case_state()
    with pytest.raises(ValueError):
        apply_progress(state, {"ticket_receipt": {"ticket_id": "NEXO-FAKE"}}, "请创建工单", "r1")
    assert "ticket_receipt" not in device_tools()
    assert "create_ticket" not in device_tools()
    assert "create_ticket" not in escalation_tools()
    assert state["ticket_receipt"] == {}


def test_plain_handoff_keeps_original_text_and_does_not_create_ticket(tmp_path):
    store = TicketStore(tmp_path / "tickets.sqlite3")
    req = request("请转人工", intent=IntentCategory.HUMAN_HANDOFF)
    response = asyncio.run(EscalationAgent(None, "offline").handle(req))
    assert "尚未创建工单或通知 IT 人员" in response.content
    assert req.case_state["ticket_receipt"] == {}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_storage_work_runs_outside_event_loop_thread(tmp_path):
    main_thread = threading.get_ident()
    class CheckedStore(TicketStore):
        def create_ticket(self, *args):
            assert threading.get_ident() != main_thread
            return super().create_ticket(*args)
        def get_ticket(self, *args):
            assert threading.get_ident() != main_thread
            return super().get_ticket(*args)
    store = CheckedStore(tmp_path / "tickets.sqlite3")
    created = asyncio.run(create_ticket(request(), store))
    queried = asyncio.run(get_ticket(request(f"查询工单 {created['ticket']['ticket_id']}"), store))
    assert queried["found"] is True
