"""Bounded final text after three tool rounds; all model responses are local fakes."""
import asyncio
import copy
import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from agents.agent_orchestrator import GeneralAgent, Request
from agents.tools import make_tool
from core.intent_recognizer import IntentCategory


def tool(number):
    return {'type': 'tool_use', 'id': 't' + str(number), 'name': 'read_detail', 'input': {'number': number}}


def response(content, stop_reason='end_turn'):
    return SimpleNamespace(content=content, stop_reason=stop_reason,
                           usage={'input_tokens': 10, 'output_tokens': 5})


def text(value='根据已有结果，请核对设备连接。', stop_reason='end_turn'):
    return response([{'type': 'text', 'text': value}], stop_reason)


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.options = []
        self.messages = self

    def with_options(self, **kwargs):
        self.options.append(kwargs)
        return self

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_agent(client, executed):
    def read_detail(req, args):
        executed.append(args['number'])
        return {'success': True, 'detail': 'detail-' + str(args['number'])}
    agent = GeneralAgent(client, 'test')
    agent.set_shared_tools({'read_detail': make_tool('read_detail', 'read existing details',
        {'number': {'type': 'integer'}}, read_detail, ['number'])})
    return agent


def request(mode='enforce'):
    return Request(message='请继续排查', user_id='u', conv_id='c', harness_mode=mode,
                   intent=IntentCategory.TECHNICAL)


def test_enforce_finishes_once_with_latest_tool_results_and_no_tools(monkeypatch):
    monkeypatch.setenv('LLM_THINKING_ENABLED', 'false')
    client = Client([response([tool(i)], 'tool_use') for i in range(1, 4)] + [text()])
    executed = []
    agent = make_agent(client, executed)
    result = asyncio.run(agent.handle(request()))
    assert result.success and len(client.calls) == 4
    assert executed == [1, 2, 3]
    assert len(result.tool_traces) == 3
    finish = client.calls[-1]
    assert finish['extra_body'] == {'thinking': {'type': 'disabled'}, 'tool_choice': {'type': 'none'}}
    assert finish['tools'] == client.calls[-2]['tools']
    assert finish['messages'][-1]['content'][0]['tool_use_id'] == 't3'
    assert 'detail-3' in finish['messages'][-1]['content'][0]['content']
    assert finish['max_tokens'] == agent.profile.max_tokens
    assert finish['temperature'] == agent.profile.temperature
    assert client.options == [{'max_retries': 0}]
    assert sum(e['event'] == 'answer_finish_started' for e in result.harness_events) == 1
    event = next(e for e in result.harness_events if e['event'] == 'model_finished' and e.get('round') == 4)
    assert event['usage'] == {'input_tokens': 10, 'output_tokens': 5}
    assert result.harness_events[-1]['reason'] == 'answer_ready'


@pytest.mark.parametrize('mode', ['off', 'observe'])
def test_other_modes_still_stop_at_three_calls(mode):
    client = Client([response([tool(i)], 'tool_use') for i in range(1, 4)] + [text()])
    executed = []
    result = asyncio.run(make_agent(client, executed).handle(request(mode)))
    assert not result.success
    assert len(client.calls) == 3 and client.options == []
    assert executed == [1, 2, 3]
    assert not any(e['event'] == 'answer_finish_started' for e in result.harness_events)


@pytest.mark.parametrize('rounds', [1, 2, 3])
def test_normal_answers_do_not_get_an_extra_call(rounds):
    client = Client([response([tool(i)], 'tool_use') for i in range(1, rounds)] + [text()])
    result = asyncio.run(make_agent(client, []).handle(request()))
    assert result.success
    assert len(client.calls) == rounds and client.options == []


@pytest.mark.parametrize('last,reason', [
    (response([tool(4)], 'tool_use'), 'tool_use'),
    (text('   '), 'empty'),
    (text('未写完', 'max_tokens'), 'truncated'),
])
def test_invalid_finish_never_executes_more_tools_or_retries(last, reason):
    client = Client([response([tool(i)], 'tool_use') for i in range(1, 4)] + [last])
    executed = []
    result = asyncio.run(make_agent(client, executed).handle(request()))
    assert not result.success
    assert len(client.calls) == 4 and executed == [1, 2, 3]
    assert len(result.tool_traces) == 3
    assert any(e['event'] == 'answer_finish_rejected' and e['reason'] == reason for e in result.harness_events)
    assert result.harness_events[-1]['reason'] == 'model_round_budget_exhausted'


def test_model_failure_does_not_start_finish():
    client = Client([response([tool(1)], 'tool_use'), RuntimeError('provider unavailable')])
    result = asyncio.run(make_agent(client, []).handle(request()))
    assert not result.success and len(client.calls) == 2
    assert client.options == []
    assert result.harness_events[-1]['reason'] == 'model_or_runtime_error'


def test_failed_tool_result_survives_finish_without_reexecution(monkeypatch):
    monkeypatch.setenv('NEXO_HARNESS_MAX_TOOL_CALLS', '2')
    client = Client([response([tool(i)], 'tool_use') for i in range(1, 4)] + [text('第三项未执行，请人工核实。')])
    executed = []
    result = asyncio.run(make_agent(client, executed).handle(request()))
    assert result.success and executed == [1, 2]
    final_result = json.loads(client.calls[-1]['messages'][-1]['content'][0]['content'])
    assert final_result['success'] is False
    assert '预算' in final_result['error']
    assert len(result.tool_traces) == 3


@pytest.mark.parametrize('finish_status', [200, 503])
def test_real_sdk_serializes_none_and_does_not_retry_finish(monkeypatch, finish_status):
    monkeypatch.setenv('LLM_THINKING_ENABLED', 'false')
    requests = []
    def transport(req):
        body = json.loads(req.content)
        requests.append(body)
        number = len(requests)
        if number == 4 and finish_status != 200:
            return httpx.Response(finish_status, json={'type': 'error', 'error': {'type': 'api_error', 'message': 'unavailable'}})
        content = [tool(number)] if number < 4 else [{'type': 'text', 'text': '已根据工具结果整理答复。'}]
        return httpx.Response(200, json={'id': 'msg-' + str(number), 'type': 'message', 'role': 'assistant',
            'model': 'test', 'content': content, 'stop_reason': 'tool_use' if number < 4 else 'end_turn',
            'stop_sequence': None, 'usage': {'input_tokens': 10, 'output_tokens': 5}})
    async def run():
        http = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        client = anthropic.AsyncAnthropic(api_key='fake', base_url='http://mock', http_client=http, max_retries=2)
        try:
            executed = []
            output = await make_agent(client, executed).handle(request())
            assert executed == [1, 2, 3]
            return output
        finally:
            await client.close()
    result = asyncio.run(run())
    assert len(requests) == 4
    assert requests[-1]['tool_choice'] == {'type': 'none'}
    assert requests[-1]['thinking'] == {'type': 'disabled'}
    assert requests[-1]['tools'] == requests[-2]['tools']
    assert result.success == (finish_status == 200)
    if finish_status != 200:
        assert result.harness_events[-1]['reason'] == 'model_or_runtime_error'

def test_third_round_progress_is_retained_in_finish_and_result():
    update = {'facts': [{'field': 'issue_description', 'value': '无法开机', 'source_quote': '电脑无法开机'}]}
    progress = {'type': 'tool_use', 'id': 'progress-3', 'name': 'record_support_progress', 'input': update}
    client = Client([response([tool(1)], 'tool_use'), response([tool(2)], 'tool_use'),
                     response([progress], 'tool_use'), text()])
    req = request()
    req.message = '电脑无法开机'
    result = asyncio.run(make_agent(client, []).handle(req))
    assert result.success and result.state_updates == [update]
    assert result.tools_used[-1] == 'record_support_progress'
    last_result = client.calls[-1]['messages'][-1]['content'][0]
    assert last_result['tool_use_id'] == 'progress-3'
    assert json.loads(last_result['content'])['success'] is True
    assert req.case_state == {}


def test_balance_error_does_not_start_finish_or_retry():
    requests = []
    def transport(req):
        requests.append(req)
        return httpx.Response(402, json={'type': 'error', 'error': {'type': 'payment_error', 'message': 'insufficient balance'}})
    async def run():
        client = anthropic.AsyncAnthropic(api_key='fake', base_url='http://mock', max_retries=2,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
        try:
            return await make_agent(client, []).handle(request())
        finally:
            await client.close()
    result = asyncio.run(run())
    assert not result.success and len(requests) == 1
    assert not any(e['event'] == 'answer_finish_started' for e in result.harness_events)
