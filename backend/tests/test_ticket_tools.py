import asyncio
import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import AgentOrchestrator, AgentType, GeneralAgent, TechnicalAgent, RoutingDecision
from agents.tools import create_ticket, prepare_ticket, ticket_confirmation_matches, ticket_tools
from core.ticket_store import TicketStore
from core.intent_recognizer import IntentCategory
from harness.state import prepare_case, apply_progress
from test_harness_runtime import Client, tool, text
from test_ticket_workflow import request, case_state, router, run


def pending_case():
    state = case_state()
    state['ticket_confirmation'] = prepare_ticket(request('信息完整了', case=state, request_id='prepare'))['ticket_confirmation']
    state['ticket_confirmation']['delivered'] = True
    return state


@pytest.mark.parametrize('reply', ['需要', '好的', '可以', '确认', '好的，谢谢。'])
def test_short_reply_needs_matching_delivered_proposal(tmp_path, reply):
    store = TicketStore(tmp_path / 'tickets.sqlite3')
    assert not asyncio.run(create_ticket(request(reply), store))['success']
    state = pending_case()
    result = asyncio.run(create_ticket(request(reply, case=state), store))
    assert result['success'] and result['created']


@pytest.mark.parametrize('reply', ['不需要', '先别提交', '需要吗？', '如果可以再提交', '好的，但是先别提交', '转人工'])
def test_denial_or_ambiguity_never_creates_even_with_pending(tmp_path, reply):
    store = TicketStore(tmp_path / 'tickets.sqlite3')
    result = asyncio.run(create_ticket(request(reply, case=pending_case()), store))
    assert not result['success']
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('select count(*) from tickets').fetchone()[0] == 0


def test_confirmation_invalidated_by_change_cancel_new_topic_and_not_delivered(tmp_path):
    state = pending_case()
    changed = copy.deepcopy(state)
    changed['facts']['issue_description']['value'] = '打印机无法开机'
    assert not ticket_confirmation_matches(request('需要', case=changed))
    for message in ['不要登记', '改一下，设备是打印机', '换个问题，申请权限']:
        assert not prepare_case(state, message).get('ticket_confirmation')
    state['ticket_confirmation']['delivered'] = False
    assert not ticket_confirmation_matches(request('需要', case=state))
    state['ticket_confirmation']['delivered'] = True
    assert not ticket_confirmation_matches(request('需要', case=state, request_id='prepare'))


def test_model_cannot_supply_confirmation_or_snapshot():
    for key in ['ticket_confirmation', 'ticket_receipt']:
        with pytest.raises(ValueError):
            apply_progress(case_state(), {key: {'delivered': True}}, '需要', 'x')
    for spec in ticket_tools(None, None).values():
        assert spec.input_schema['properties'] == {}
        assert spec.input_schema['additionalProperties'] is False


def test_same_turn_prepare_cannot_authorize_write(tmp_path):
    client = Client([tool('prepare_ticket', {}) + tool('create_ticket', {}, 'c'), text('请确认')])
    agent = GeneralAgent(client, 'offline');agent._ticket_store = TicketStore(tmp_path / 'tickets.sqlite3')
    result = asyncio.run(agent.handle(request('需要')))
    assert result.tool_traces[0]['result_success'] is True
    assert result.tool_traces[1]['result_success'] is False
    assert not result.tool_traces[0]['ticket_state']['ticket_confirmation']['delivered']


def test_prepare_reads_same_turn_recorded_facts_and_delivers_question(tmp_path):
    state = case_state();state['facts'] = {};state['steps'] = [];state['pending_questions'] = []
    client = Client([tool('record_support_progress', {'facts': [
        {'field': 'issue_description', 'value': '笔记本无法开机', 'source_quote': '笔记本无法开机'}]})
        + tool('prepare_ticket', {}, 'p'), text('已整理问题。')])
    orch = router(TicketStore(tmp_path / 'tickets.sqlite3'))
    orch._pool[AgentType.GENERAL][0]._client = client
    req = request('笔记本无法开机', case=state)
    result = run(orch, req)
    assert result.case_state['ticket_confirmation']['delivered'] is True
    assert '待登记信息：笔记本无法开机' in result.response
    assert '需要我按以上信息登记工单吗' in result.response
    assert not result.case_state['ticket_receipt']
    assert req.case_state == state and not state['facts']


def test_failed_final_answer_never_activates_proposal(tmp_path):
    orch = router(TicketStore(tmp_path / 'tickets.sqlite3'))
    orch._pool[AgentType.GENERAL][0]._client = Client([tool('prepare_ticket', {}), RuntimeError('offline model failure')])
    result = run(orch, request('信息完整了'))
    assert not result.case_state['ticket_confirmation']


def test_create_is_selected_by_model_not_command_bypass(tmp_path):
    store = TicketStore(tmp_path / 'tickets.sqlite3');orch = router(store)
    client = Client([text('还需确认问题')]);orch._pool[AgentType.GENERAL][0]._client = client
    result = run(orch, request())
    assert len(client.calls) == 1 and not result.case_state['ticket_receipt']
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('select count(*) from tickets').fetchone()[0] == 0


def test_confirmed_tool_creates_once_under_concurrent_calls_and_restart(tmp_path):
    store = TicketStore(tmp_path / 'tickets.sqlite3');state = pending_case()
    async def scenario():
        return await asyncio.gather(*(create_ticket(request('需要', case=state), store) for _ in range(5)))
    results = asyncio.run(scenario())
    assert sum(item['created'] for item in results) == 1
    ids = {item['ticket']['ticket_id'] for item in results};assert len(ids) == 1
    reloaded = TicketStore(store.path)
    assert reloaded.get_ticket(ids.pop(), 'ticket-user', 'ticket-conv')['found']


def test_write_then_model_failure_preserves_receipt_and_blocks_fallback(tmp_path):
    store = TicketStore(tmp_path / 'tickets.sqlite3');orch = router(store)
    client = Client([tool('create_ticket', {}), RuntimeError('offline answer error')])
    technical = orch._pool[AgentType.TECHNICAL][0];technical._client = client
    response = asyncio.run(orch._execute(request(), AgentType.TECHNICAL))
    assert not response.success
    assert any(e.get('reason') == 'ticket_write_attempted' for e in response.harness_events)
    assert len(orch._pool[AgentType.GENERAL][0]._client.calls) == 0
    # Verify the delivery boundary also recovers the committed receipt.
    orch._pool[AgentType.GENERAL][0]._client = Client([tool('create_ticket', {}), RuntimeError('offline answer error')])
    result = run(orch, request())
    assert result.case_state['ticket_receipt']['ticket_id'] in result.response
    assert '待人工处理' in result.response
    assert any(e['event'] == 'ticket_receipt_recovered' for e in result.harness_events)


def test_supporting_agent_cannot_call_ticket_tools(tmp_path):
    client = Client([tool('create_ticket', {}), text('不建单')])
    agent = TechnicalAgent(client, 'offline');agent._ticket_store = TicketStore(tmp_path / 'tickets.sqlite3')
    result = asyncio.run(agent.handle(request(ticket_tools_allowed=False)))
    assert result.tool_traces[0]['success'] is False
    assert 'create_ticket' not in [item['name'] for item in client.calls[0]['tools']]


def test_parallel_dispatch_allows_ticket_tools_only_for_primary(tmp_path):
    orch = router(TicketStore(tmp_path / 'tickets.sqlite3'));seen=[]
    from agents.agent_orchestrator import AgentResponse
    async def execute(req, kind):
        seen.append((kind, req.ticket_tools_allowed))
        return AgentResponse(kind, '已记录', True)
    async def compose(req, responses):
        return '已记录'
    orch._execute=execute;orch._composer.compose=compose
    asyncio.run(orch.run_parallel(request(), RoutingDecision(AgentType.TECHNICAL, [AgentType.ACCESS])))
    assert seen == [(AgentType.TECHNICAL, True), (AgentType.ACCESS, False)]

def test_progress_after_committed_ticket_cannot_replace_receipt(tmp_path):
    client = Client([tool('create_ticket', {}) + tool('record_support_progress', {'new_case': True}, 'p'), text('已登记')])
    agent = GeneralAgent(client, 'offline');agent._ticket_store = TicketStore(tmp_path / 'tickets.sqlite3')
    result = asyncio.run(agent.handle(request()))
    assert result.tool_traces[0]['ticket_state']['ticket_receipt']['ticket_id']
    assert result.tool_traces[1]['result_success'] is False
    assert not result.state_updates


def test_runtime_and_role_contract_agree_on_available_ticket_tools(tmp_path):
    agent = GeneralAgent(Client([text()]), 'offline');agent._ticket_store = TicketStore(tmp_path / 'tickets.sqlite3')
    prompt = agent._build_system_prompt(request())
    line = next(line for line in prompt.splitlines() if line.startswith('允许的数据/工具范围'))
    assert 'prepare_ticket' in line and 'create_ticket' in line

def test_new_case_preparation_keeps_actual_runtime_case_identity(tmp_path):
    orch = router(TicketStore(tmp_path / 'tickets.sqlite3'))
    client = Client([tool('record_support_progress', {'new_case': True, 'facts': [
        {'field': 'issue_description', 'value': '打印机无法开机', 'source_quote': '打印机无法开机'}]})
        + tool('prepare_ticket', {}, 'p'), text('已整理')])
    orch._pool[AgentType.GENERAL][0]._client = client
    result = run(orch, request('换个问题，打印机无法开机'))
    pending = result.case_state['ticket_confirmation']
    assert pending['delivered']
    assert pending['case_id'] == result.case_state['case_id']
    assert ticket_confirmation_matches(request('需要', case=result.case_state))
