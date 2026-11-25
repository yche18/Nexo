"""Ticket round budget and leaked tool markup; no model/network calls."""
import asyncio
import sqlite3

import pytest

from agents.agent_orchestrator import AgentType, GeneralAgent
from agents.tools import make_tool
from core.intent_recognizer import IntentCategory
from core.ticket_store import TicketStore
from harness.state import new_case
from test_harness_runtime import Client, tool, text
from test_ticket_workflow import request, router, run

DSML = '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="prepare_ticket"><｜｜DSML｜｜ parameter name="target_system" string="true">项目系统</｜｜DSML｜｜ parameter></｜｜DSML｜｜ invoke></｜｜DSML｜｜ calls>'


def test_page_demo_four_tools_in_four_separate_rounds(tmp_path):
    message = '我想申请项目系统权限，用于查看需求'
    progress = {'facts': [
        {'field': 'target_system', 'value': '项目系统', 'source_quote': message},
        {'field': 'purpose', 'value': '查看需求', 'source_quote': message},
    ]}
    client = Client([
        tool('check_access_fields', {}),
        tool('search_knowledge_base', {'query': '项目系统权限申请流程'}, 'rag'),
        tool('record_support_progress', progress, 'progress'),
        tool('prepare_ticket', {}, 'prepare'), text('已整理申请信息。'),
    ])
    orch = router(TicketStore(tmp_path / 'tickets.sqlite3'))
    agent = orch._pool[AgentType.ACCESS][0];agent._client = client
    agent.set_shared_tools({'search_knowledge_base': make_tool('search_knowledge_base', '测试只读资料',
        {'query': {'type': 'string'}}, lambda req, args: {'success': True, 'results': []})})
    result = run(orch, request(message, intent=IntentCategory.PERMISSION_REQUEST,
                               case=new_case('ticket-user', 'ticket-conv')))
    assert result.tools_used == ['check_access_fields', 'search_knowledge_base', 'record_support_progress', 'prepare_ticket']
    assert len(client.calls) == 5
    assert result.case_state['ticket_confirmation']['delivered']
    assert '需要我按以上信息登记工单吗' in result.response
    assert not any(e['event'] == 'answer_finish_started' for e in result.harness_events)


def ticket_agent(tmp_path, responses):
    client = Client(responses)
    agent = GeneralAgent(client, 'offline');agent._ticket_store = TicketStore(tmp_path / 'tickets.sqlite3')
    return agent, client


def test_ticket_budget_five_then_only_one_clean_finish(tmp_path):
    agent, client = ticket_agent(tmp_path, [tool('inspect_request_context', {}, str(i)) for i in range(5)] + [text('摘要尚未准备完成。')])
    result = asyncio.run(agent.handle(request('请继续')))
    assert result.success and len(client.calls) == 6
    assert len(result.tools_used) == 5
    finish = client.calls[-1]
    assert finish['extra_body']['tool_choice'] == {'type': 'none'}
    assert '[工单工具契约]' not in finish['system']
    assert '[动态 Skills]' not in finish['system'] and '[任务进度]' not in finish['system']
    assert '紧接着调用 prepare_ticket' not in finish['system']
    assert '摘要尚未准备完成' in finish['system']
    assert '只有 create_ticket 的成功回执可以证明已建单' in finish['system']
    assert result.harness_events[-1]['rounds'] == 6


def test_supporting_agent_keeps_original_three_rounds(tmp_path):
    agent, client = ticket_agent(tmp_path, [tool('inspect_request_context', {}, str(i)) for i in range(3)] + [text()])
    result = asyncio.run(agent.handle(request(ticket_tools_allowed=False)))
    assert result.success and len(client.calls) == 4
    assert 'prepare_ticket' not in [t['name'] for t in client.calls[0]['tools']]


def test_ticket_agent_stops_immediately_on_normal_answer(tmp_path):
    agent, client = ticket_agent(tmp_path, [text('还需要补充目标系统。')])
    result = asyncio.run(agent.handle(request('申请权限')))
    assert result.success and len(client.calls) == 1


@pytest.mark.parametrize('content', [DSML, '已整理。\n' + DSML, DSML.replace('｜', '|')])
def test_normal_text_markup_is_rejected_without_execution(tmp_path, content):
    agent, client = ticket_agent(tmp_path, [text(content)])
    result = asyncio.run(agent.handle(request()))
    assert not result.success and len(client.calls) == 1 and not result.tools_used
    assert 'DSML' not in result.content
    assert any(e.get('reason') == 'tool_markup' for e in result.harness_events)
    with sqlite3.connect(agent._ticket_store.path) as conn:
        assert conn.execute('select count(*) from tickets').fetchone()[0] == 0


def test_markup_mixed_with_real_tool_block_is_not_executed(tmp_path):
    agent, client = ticket_agent(tmp_path, [text(DSML) + tool('create_ticket', {})])
    result = asyncio.run(agent.handle(request()))
    assert not result.success and not result.tools_used
    with sqlite3.connect(agent._ticket_store.path) as conn:
        assert conn.execute('select count(*) from tickets').fetchone()[0] == 0


def test_finish_markup_rejected_after_budget_without_retry(tmp_path):
    agent, client = ticket_agent(tmp_path, [tool('inspect_request_context', {}, str(i)) for i in range(5)] + [text(DSML)])
    result = asyncio.run(agent.handle(request()))
    assert not result.success and len(client.calls) == 6
    assert len(result.tools_used) == 5
    assert any(e['event'] == 'answer_finish_rejected' and e['reason'] == 'tool_markup' for e in result.harness_events)
    assert 'DSML' not in result.content


@pytest.mark.parametrize('finish', [False, True])
def test_receipt_survives_markup_after_write_without_second_write(tmp_path, finish):
    store = TicketStore(tmp_path / 'tickets.sqlite3');orch = router(store)
    responses = [tool('create_ticket', {})]
    if finish:
        responses += [tool('inspect_request_context', {}, str(i)) for i in range(4)]
    client = Client(responses + [text(DSML)])
    orch._pool[AgentType.GENERAL][0]._client = client
    result = run(orch, request())
    receipt = result.case_state['ticket_receipt']
    assert receipt['ticket_id'] in result.response and 'DSML' not in result.response
    assert any(e['event'] == 'ticket_receipt_recovered' for e in result.harness_events)
    assert sum(t['tool_name'] == 'create_ticket' for t in result.tool_traces) == 1
    assert len(client.calls) == (6 if finish else 2)
    with sqlite3.connect(store.path) as conn:
        assert conn.execute('select count(*) from tickets').fetchone()[0] == 1


def test_existing_total_tool_limit_still_applies(tmp_path, monkeypatch):
    monkeypatch.setenv('NEXO_HARNESS_MAX_TOOL_CALLS', '2')
    agent, client = ticket_agent(tmp_path, [tool('inspect_request_context', {}, str(i)) for i in range(3)] + [text()])
    result = asyncio.run(agent.handle(request()))
    assert len(result.tools_used) == 2
    assert result.tool_traces[2]['success'] is False
    assert any(e.get('reason') == 'tool_budget_exhausted' for e in result.harness_events)
