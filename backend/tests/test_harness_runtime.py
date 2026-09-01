import asyncio
import copy
import json
from types import SimpleNamespace

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, EscalationAgent,
    OrchestratorResult, Request, TechnicalAgent, GeneralAgent,
)
from agents.tools import make_tool
from core.intent_recognizer import IntentCategory, UrgencyLevel
from harness.state import new_case, apply_progress


class Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(content=response, stop_reason='end_turn')


def text(value='完成'):
    return [{'type': 'text', 'text': value}]


def tool(name, args, call_id='t1'):
    return [{'type': 'tool_use', 'name': name, 'input': args, 'id': call_id}]


def request(mode='observe', **kwargs):
    data = dict(message='VPN 连接超时', user_id='u', conv_id='c', harness_mode=mode,
                intent=IntentCategory.TECHNICAL_LOGIN, urgency=UrgencyLevel.HIGH)
    data.update(kwargs)
    return Request(**data)


def test_off_and_observe_send_identical_requests_without_extra_calls():
    clients = []
    outputs = []
    for mode in ['off', 'observe']:
        client = Client([tool('lookup_error_code', {'error_code': '401'}), text()])
        outputs.append(asyncio.run(TechnicalAgent(client, 'test').handle(request(mode))))
        clients.append(client)
    assert clients[0].calls == clients[1].calls
    assert len(clients[0].calls) == 2
    assert outputs[0].content == outputs[1].content
    assert outputs[0].harness_events == []
    assert outputs[1].harness_events[-1]['reason'] == 'answer_ready'


def test_progress_updates_are_staged_without_mutating_request_entities_or_case():
    req = request('enforce', message='系统是 SAP', case_state=new_case('u', 'c'), entities={})
    before = copy.deepcopy(req.case_state)
    update = {'facts': [{'field': 'target_system', 'value': 'SAP', 'source_quote': '系统是 SAP'}]}
    client = Client([tool('record_support_progress', update), text()])
    result = asyncio.run(TechnicalAgent(client, 'test').handle(req))
    assert result.success
    assert result.state_updates == [update]
    assert req.case_state == before
    assert req.entities == {}


def test_invalid_fact_warns_without_promoting_it_to_state():
    update = {'facts': [{'field': 'target_system', 'value': 'SAP', 'source_quote': '系统是 SAP'}]}
    client = Client([tool('record_support_progress', update), text()])
    req = request('enforce', message='你好', case_state=new_case('u', 'c'))
    result = asyncio.run(TechnicalAgent(client, 'test').handle(req))
    assert result.state_updates == [update]
    assert result.tool_traces[0]['success'] is True
    reply = json.loads(client.calls[1]['messages'][-1]['content'][0]['content'])
    assert reply['success'] is True
    assert reply['warnings'][0]['kind'] == 'invalid_fact'
    assert '来源必须' in reply['warnings'][0]['reason']
    merged = AgentOrchestrator._merge_case_state(req, result.state_updates, result.tool_traces, [])
    assert merged['facts'] == {}
    assert req.case_state['facts'] == {}
    assert 'record_support_progress' in [item['name'] for item in client.calls[1]['tools']]


def test_tool_budget_blocks_excess_calls_without_executing_them(monkeypatch):
    monkeypatch.setenv('NEXO_HARNESS_MAX_TOOL_CALLS', '1')
    client = Client([tool('lookup_error_code', {'error_code': '401'}) +
                     tool('lookup_error_code', {'error_code': '500'}, 't2'), text()])
    result = asyncio.run(TechnicalAgent(client, 'test').handle(request('enforce')))
    assert result.success
    assert result.tools_used == ['lookup_error_code']
    assert result.tool_traces[1]['success'] is False
    assert any(e.get('reason') == 'tool_budget_exhausted' for e in result.harness_events)


def test_observe_three_model_round_limit_keeps_failure_trace():
    client = Client([tool('lookup_error_code', {'error_code': '401'})] * 3)
    result = asyncio.run(TechnicalAgent(client, 'test').handle(request('observe')))
    assert not result.success
    assert len(client.calls) == 3
    assert len(result.tool_traces) == 3
    assert result.harness_events[-1]['reason'] == 'model_round_budget_exhausted'


def test_sop_guard_requires_real_matching_evidence_and_only_blocks_enforce():
    evidence = {'title': 'VPN 连接排查（演示资料）',
                'content': '连接超时应按网络、DNS、代理、防火墙和服务状态进行排查。'}
    def search(req, args):
        return {'success': True, 'results': [evidence], 'retrieval': {'evidence': [evidence]}}
    for mode in ['observe', 'enforce']:
        client = Client([tool('search_knowledge_base', {'query': 'VPN'}),
                         tool('build_diagnostic_plan', {'environment': 'VPN', 'reproduced': True}), text()])
        agent = TechnicalAgent(client, 'test')
        agent.set_shared_tools({'search_knowledge_base': make_tool(
            'search_knowledge_base', 'search', {'query': {'type': 'string'}}, search)})
        result = asyncio.run(agent.handle(request(mode)))
        assert result.success
        called = 'build_diagnostic_plan' in result.tools_used
        assert called == (mode == 'observe')


def test_finalizer_runs_on_composed_text_and_preserves_case_on_handoff(monkeypatch):
    from harness.validation import ValidationResult
    captured = []
    async def check(message, answer, traces, client, model, mode, case_state=None):
        captured.append((answer, case_state))
        return ValidationResult(content='unused', status='needs_handoff', escalated=True)
    monkeypatch.setattr('harness.validation.finalize_answer', check)
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._composer = SimpleNamespace(_client=Client([]), _model='test')
    orch._pool = {AgentType.ESCALATION: [EscalationAgent(Client([]), 'test')]}
    orch._recent_tool_traces = []
    case = apply_progress(new_case('u', 'c'), {'facts': [
        {'field': 'target_system', 'value': 'SAP', 'source_quote': '系统是 SAP'}]}, '系统是 SAP', 'r0')
    req = request('enforce', case_state=case)
    result = OrchestratorResult(req.request_id, '最终汇总文本', AgentType.TECHNICAL, req.intent)
    output = asyncio.run(orch._finalize_result(req, result))
    assert captured[0][0] == '最终汇总文本'
    assert output.escalated
    assert 'SAP' in output.response
    assert '待人工处理' in output.response
    assert output.case_state['status'] == 'handoff_ready'
    assert output.agent_type == AgentType.ESCALATION

def test_fallback_keeps_failed_attempt_evidence_and_progress():
    update = {'facts': [{'field': 'target_system', 'value': 'SAP', 'source_quote': '系统是 SAP'}]}
    evidence = {'id': 'doc1', 'title': 'VPN SOP', 'content': '已有排查资料'}
    trace = {'tool_name': 'search_knowledge_base', 'success': True, 'result_success': True,
             'retrieval': {'evidence': [evidence]}}
    class Failing:
        async def handle(self, req):
            return AgentResponse(AgentType.TECHNICAL, 'failed', False,
                                 tools_used=['search_knowledge_base'], tool_traces=[trace],
                                 state_updates=[update])
    seen = []
    class Fallback:
        async def handle(self, req):
            seen.append(req)
            return AgentResponse(AgentType.GENERAL, '继续处理', True)
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    mapping = {AgentType.TECHNICAL: Failing(), AgentType.GENERAL: Fallback()}
    orch._best_agent = lambda agent_type: mapping[agent_type]
    req = request('enforce', message='系统是 SAP', case_state=new_case('u', 'c'))
    response = asyncio.run(orch._execute(req, AgentType.TECHNICAL))
    assert response.tool_traces == [trace]
    assert response.state_updates == [update]
    assert seen[0].case_state['facts']['target_system']['value'] == 'SAP'
    assert '已有排查资料' in seen[0].context
    assert not req.case_state['facts']


def test_parallel_composer_receives_merged_state_once_for_new_topic():
    from agents.agent_orchestrator import RoutingDecision
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    one = {'new_case': True, 'facts': [
        {'field': 'target_system', 'value': 'SAP', 'source_quote': '另一个问题：SAP 用于查看需求'}]}
    two = {'new_case': True, 'facts': [
        {'field': 'purpose', 'value': '查看需求', 'source_quote': '另一个问题：SAP 用于查看需求'}]}
    async def execute(req, agent_type):
        return AgentResponse(agent_type, '候选回答', True,
                             state_updates=[one if agent_type == AgentType.TECHNICAL else two])
    captured = []
    class Composer:
        async def compose(self, req, responses):
            captured.append(copy.deepcopy(req.case_state))
            return '汇总完成'
    async def finish(req, result):
        captured.append(copy.deepcopy(result.case_state))
        return result
    orch._execute = execute
    orch._composer = Composer()
    orch._finalize_result = finish
    req = request('enforce', message='另一个问题：SAP 用于查看需求', case_state=new_case('u', 'c'))
    decision = RoutingDecision(AgentType.TECHNICAL, [AgentType.ACCESS], 'test', 1.0)
    result = asyncio.run(orch.run_parallel(req, decision))
    assert captured[0] == captured[1]
    assert set(result.case_state['facts']) == {'target_system', 'purpose'}
    assert req.case_state['facts'] == {}


def test_handoff_does_not_resurrect_conflicting_history_field():
    case = new_case('u', 'c')
    case['conflicts'] = [{'field': 'target_system', 'previous': 'SAP', 'value': 'OA'}]
    req = request('enforce', case_state=case, history=[{'role': 'assistant', 'content': '目标系统：SAP'}])
    output = asyncio.run(EscalationAgent(Client([]), 'test').handle(req))
    assert 'SAP' not in output.content


def test_new_topic_handoff_does_not_copy_previous_case_history():
    case = new_case('u', 'c')
    case['new_topic'] = True
    req = request('enforce', message='另一个问题，帮我转人工', case_state=case,
                  history=[{'role': 'assistant', 'content': '目标系统：SAP'}])
    output = asyncio.run(EscalationAgent(Client([]), 'test').handle(req))
    assert 'SAP' not in output.content


def test_missing_progress_is_reported_without_extra_calls_or_answer_changes():
    client = Client([text('请补充错误提示')])
    response = asyncio.run(TechnicalAgent(client, 'test').handle(request('enforce')))
    assert response.content == '请补充错误提示'
    assert len(client.calls) == 1
    assert any(e['event'] == 'progress_not_recorded' for e in response.harness_events)


def test_handoff_lists_only_user_reported_completed_steps():
    case = apply_progress(new_case('u', 'c'), {'steps': [
        {'step_id': 'restart', 'description': '重启客户端', 'status': 'user_reported_done',
         'result': '仍超时', 'source_quote': '重启客户端后仍超时'},
        {'step_id': 'dns', 'description': '检查 DNS', 'status': 'proposed'},
    ]}, '重启客户端后仍超时', 'r0')
    agent = EscalationAgent(Client([]), 'test')
    enforce = asyncio.run(agent.handle(request('enforce', case_state=case)))
    assert '重启客户端：仍超时' in enforce.content
    assert '检查 DNS' not in enforce.content
    assert '用户反馈' in enforce.content
    observe = asyncio.run(agent.handle(request('observe', case_state=case)))
    assert '已尝试及结果' not in observe.content


def test_invalid_top_level_progress_retires_tool_before_next_model_round():
    update = {'status': 'unsupported_status'}
    client = Client([tool('record_support_progress', update), text('请补充信息')])
    response = asyncio.run(TechnicalAgent(client, 'test').handle(request('enforce', message='VPN超时')))
    assert len(client.calls) == 2
    assert response.content == '请补充信息'
    assert 'record_support_progress' not in [item['name'] for item in client.calls[1]['tools']]
    assert any(event['event'] == 'progress_tool_retired' for event in response.harness_events)
    assert '不要重试' in client.calls[1]['messages'][-1]['content'][0]['content']


def test_conditional_escalation_flag_does_not_mark_case_handoff(monkeypatch):
    from harness.validation import ValidationResult
    async def check(*args, **kwargs):
        return ValidationResult(content='如果仍失败，可转人工', status='passed')
    monkeypatch.setattr('harness.validation.finalize_answer', check)
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._composer = SimpleNamespace(_client=Client([]), _model='test')
    orch._recent_tool_traces = []
    req = request('enforce', case_state=new_case('u', 'c'))
    req.case_state['status'] = 'waiting_user'
    result = OrchestratorResult(req.request_id, '如果仍失败，可转人工', AgentType.TECHNICAL,
                                req.intent, escalated=True)
    output = asyncio.run(orch._finalize_result(req, result))
    assert output.escalated is True  # original presentation flag retained
    assert output.case_state['status'] == 'waiting_user'


def test_runtime_failure_is_not_sent_to_semantic_validator(monkeypatch):
    async def forbidden(*args, **kwargs):
        raise AssertionError('runtime placeholder cannot be semantically repaired')
    monkeypatch.setattr('harness.validation.finalize_answer', forbidden)
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    client = Client([])
    orch._composer = SimpleNamespace(_client=client, _model='test')
    orch._pool = {AgentType.ESCALATION: [EscalationAgent(client, 'test')]}
    orch._recent_tool_traces = []
    case = apply_progress(new_case('u', 'c'), {'facts': [
        {'field': 'target_system', 'value': 'SAP', 'source_quote': 'SAP'}]}, 'SAP', 'r0')
    req = request('enforce', case_state=case)
    result = OrchestratorResult(req.request_id, '抱歉，处理出现问题', AgentType.TECHNICAL,
                                req.intent, execution_success=False)
    output = asyncio.run(orch._finalize_result(req, result))
    assert output.validation['status'] == 'runtime_failed'
    assert output.validation['extra_calls'] == 0
    assert output.case_state['status'] == 'handoff_ready'
    assert 'SAP' in output.response
    assert '待人工处理' in output.response
    assert client.calls == []


def test_model_finished_records_sdk_usage_and_latency_without_request_changes():
    class Usage:
        def model_dump(self, mode):
            assert mode == 'json'
            return {'input_tokens': 21, 'output_tokens': 8, 'cache_read_input_tokens': 5}
    class UsageClient(Client):
        async def create(self, **kwargs):
            response = await super().create(**kwargs)
            response.usage = Usage()
            return response
    clients = []
    for mode in ['off', 'observe']:
        client = UsageClient([text('请补充信息')])
        result = asyncio.run(TechnicalAgent(client, 'test').handle(request(mode)))
        clients.append(client)
        if mode == 'off':
            assert result.harness_events == []
        else:
            event = next(e for e in result.harness_events if e['event'] == 'model_finished')
            assert event['usage'] == {'input_tokens': 21, 'output_tokens': 8, 'cache_read_input_tokens': 5}
            assert event['latency_ms'] >= 0
    assert clients[0].calls == clients[1].calls
    assert len(clients[0].calls) == 1


def test_handoff_candidate_cannot_return_from_assistant_or_summary():
    case = new_case('u', 'c')
    case['candidates'] = [{'field': 'target_system', 'value': 'SAP', 'source': 'model_interpretation'}]
    agent = EscalationAgent(Client([]), 'test')
    req = request('enforce', case_state=case, context='[会话摘要]\n目标系统：SAP',
                  history=[{'role': 'user', 'content': '错误码：401'},
                           {'role': 'assistant', 'content': '目标系统：SAP'}])
    result = asyncio.run(agent.handle(req))
    assert 'SAP' not in result.content
    assert '401' in result.content  # unrelated original fields stay available
    req.entities = {'target_system': ['OA']}
    assert '目标系统：OA' in asyncio.run(agent.handle(req)).content
    req.case_state = apply_progress(case, {'facts': [
        {'field': 'target_system', 'value': 'VPN', 'source_quote': 'VPN'}]}, 'VPN', 'r0')
    result = asyncio.run(agent.handle(req))
    assert '目标系统：VPN' in result.content
    assert '目标系统：OA' not in result.content
    req.case_state['conflicts'] = [{'field': 'target_system'}]
    assert '目标系统：' not in asyncio.run(agent.handle(req)).content
    req.harness_mode = 'observe'
    req.entities = {}
    assert '目标系统：SAP' in asyncio.run(agent.handle(req)).content


def test_run_routes_then_executes_general_fallback_and_real_finalizer():
    from agents.agent_orchestrator import ResponseComposer
    evidence = {'chunk_id': 'vpn-chunk-1', 'title': 'VPN 连接排查（演示资料）',
                'content': '连接超时应按网络、DNS、代理、防火墙和服务状态进行排查。'}
    update = {'facts': [
        {'field': 'target_system', 'value': 'VPN', 'source_quote': '目标系统：VPN'},
        {'field': 'issue_description', 'value': '连接超时', 'source_quote': '问题现象：连接超时'},
    ], 'steps': [{'step_id': 'restart', 'description': '重启客户端',
                 'status': 'user_reported_done', 'result': '仍超时',
                 'source_quote': '重启客户端后仍超时'},
                {'step_id': 'invented', 'description': '重新安装',
                 'status': 'user_reported_done', 'result': '恢复正常',
                 'source_quote': '重新安装后恢复正常'}], 'evidence_refs': ['vpn-chunk-1']}
    for fallback_fails in [False, True]:
        technical = Client([tool('search_knowledge_base', {'query': 'VPN 超时'}) +
                            tool('record_support_progress', update, 'progress'), RuntimeError('technical failed')])
        general = Client([RuntimeError('general failed') if fallback_fails else text('按已有 VPN SOP 继续排查')])
        judge = Client([] if fallback_fails else [tool('submit_support_validation',
                         {'passed': True, 'issues': [], 'reason': '承接用户反馈并覆盖资料'})])
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        orch._pool = {AgentType.TECHNICAL: [TechnicalAgent(technical, 'test')],
                      AgentType.GENERAL: [GeneralAgent(general, 'test')],
                      AgentType.ESCALATION: [EscalationAgent(judge, 'test')]}
        orch._composer = ResponseComposer(judge, 'test')
        orch._recent_tool_traces = []
        def search(req, args):
            return {'success': True, 'results': [evidence], 'retrieval': {'evidence': [evidence]}}
        orch.set_shared_tools({'search_knowledge_base': make_tool(
            'search_knowledge_base', 'search', {'query': {'type': 'string'}}, search)})
        req = request('enforce', message='目标系统：VPN，问题现象：连接超时；重启客户端后仍超时',
                      case_state=new_case('u', 'c'))
        result = asyncio.run(orch.run(req))
        assert result.primary_agent == AgentType.TECHNICAL
        assert len(technical.calls) == 2 and len(general.calls) == 1
        assert len(judge.calls) == (0 if fallback_fails else 1)
        assert result.validation['status'] == ('runtime_failed' if fallback_fails else 'passed')
        assert result.agent_type == (AgentType.ESCALATION if fallback_fails else AgentType.GENERAL)
        assert any(e['event'] == 'agent_fallback' for e in result.harness_events)
        assert result.case_state['facts']['target_system']['value'] == 'VPN'
        assert result.case_state['evidence_refs'] == ['vpn-chunk-1']
        assert [step['step_id'] for step in result.case_state['steps']] == ['restart']
        assert any(warning['kind'] == 'invalid_step'
                   for warning in result.case_state['progress_warnings'])
        assert len(result.state_updates) == 1
        assert len(result.tool_traces) == 2
        assert result.tool_traces[0]['retrieval']['evidence'] == [evidence]
        assert 'VPN 连接排查' in str(general.calls[0]['messages'])
        assert '重启客户端' in str(general.calls[0]['messages'])
        assert not req.case_state['facts']
        assert orch.get_tool_trace(req.request_id)['validation'] == result.validation
        if not fallback_fails:
            handoff = request('enforce', message='帮我转人工', intent=IntentCategory.HUMAN_HANDOFF,
                              case_state=result.case_state)
            result = asyncio.run(orch.run(handoff))
            assert len(judge.calls) == 1  # explicit handoff adds no model request
        assert '目标系统：VPN' in result.response
        assert '重启客户端：仍超时' in result.response
        assert '待人工处理' in result.response
        assert result.case_state['status'] == 'handoff_ready'


def test_run_direct_escalation_keeps_latest_user_report_without_model_or_fake_purpose():
    from agents.agent_orchestrator import ResponseComposer
    initial = 'VPN 连接超时，重启客户端后仍超时'
    case = apply_progress(new_case('u', 'c'), {'facts': [
        {'field': 'target_system', 'value': 'VPN', 'source_quote': 'VPN'},
        {'field': 'issue_description', 'value': '连接超时', 'source_quote': '连接超时'},
    ], 'steps': [{'step_id': 'restart', 'description': '重启客户端',
                 'status': 'user_reported_done', 'result': '仍超时',
                 'source_quote': '重启客户端后仍超时'}]}, initial, 'r0')
    before = copy.deepcopy(case)
    client = Client([])
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._pool = {AgentType.ESCALATION: [EscalationAgent(client, 'test')]}
    orch._composer = ResponseComposer(client, 'test')
    orch._recent_tool_traces = []
    message = '我已经换过网络，也重新登录了，还是超时，下一步怎么办？'
    req = request('enforce', message=message, intent=IntentCategory.ESCALATION,
                  case_state=case, entities={}, context='[会话摘要]\n申请用途：缩小范围',
                  history=[{'role': 'user', 'content': '错误码：401'},
                           {'role': 'assistant', 'content': '这些信息用于缩小范围'}])
    result = asyncio.run(orch.run(req))
    assert result.primary_agent == AgentType.ESCALATION
    assert result.agent_type == AgentType.ESCALATION
    assert client.calls == []
    assert '目标系统：VPN' in result.response
    assert '错误码：401' in result.response
    assert '申请用途' not in result.response
    assert '缩小范围' not in result.response
    assert message in result.response
    assert '最新反馈（用户原话）' in result.response
    assert '待 IT 人员核验' in result.response
    assert result.case_state['latest_user_report'] == {
        'content': message, 'source': 'user_reported', 'request_id': req.request_id}
    assert result.case_state['steps'] == before['steps']
    assert result.case_state['facts'] == before['facts']
    assert req.entities == {}
    assert case == before
    # A later handoff-only request retains the report without inventing a step.
    next_req = request('enforce', message='那你帮我转交给人工吧',
                       intent=IntentCategory.HUMAN_HANDOFF, case_state=result.case_state)
    second = asyncio.run(orch.run(next_req))
    assert second.case_state['latest_user_report'] == result.case_state['latest_user_report']
    assert message in second.response
    assert '那你帮我转交给人工吧' not in second.response
    assert client.calls == []
    # Existing modes keep their presentation and do not persist this new field.
    observe_req = request('observe', message=message, intent=IntentCategory.ESCALATION,
                          case_state=before, context='[会话摘要]\n申请用途：缩小范围')
    observed = asyncio.run(orch.run(observe_req))
    assert '最新反馈（用户原话）' not in observed.response
    assert '申请用途：缩小范围' in observed.response
    assert 'latest_user_report' not in observed.case_state
    assert before == case
