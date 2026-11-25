import json
from pathlib import Path
from evaluation.run_harness_eval import summarize, validate_dataset


def test_harness_dataset_declares_live_and_fault_scenarios_separately():
    data = json.loads((Path(__file__).parents[1]/'evaluation/datasets/harness_v1.json').read_text(encoding='utf-8'))
    validate_dataset(data)
    assert sum(c['execution'] == 'live' for c in data['cases']) == 8
    assert sum(c['execution'] == 'deterministic' for c in data['cases']) == 4


def test_harness_summary_counts_invalid_judges_and_unrun_faults_honestly():
    rows = [{'execution':'live','judge_valid':True,'answer_passed':True,'turns':[{'seconds':1}]},
            {'execution':'live','judge_valid':False,'judge_error':'ValueError','turns':[]},
            {'execution':'deterministic','status':'not_run'}]
    report = summarize(rows)
    assert report['live_trials'] == 2 and report['pass_rate_all_trials'] == 0.5
    assert report['judge_errors'] == 1 and report['not_run_deterministic'] == 1


def test_state_acceptance_rejects_flash_failures_even_when_answer_passes():
    from evaluation.run_harness_eval import state_checks
    question = '普通网页可以打开，但 VPN 一直连接超时，排查顺序是什么？'
    turn = {'question': question, 'chat': {'case_state': {'status': 'collecting', 'facts': {}}}}
    checks = state_checks('h01', [turn])
    assert not checks['vpn_user_fact'] and not checks['issue_quote_preserved']
    state = turn['chat']['case_state']
    state['facts'] = {'target_system': {'value': 'VPN', 'source': 'user_reported'},
                      'issue_description': {'value': question, 'source': 'user_reported', 'source_quote': question}}
    assert all(state_checks('h01', [turn]).values())
    state['status'] = 'handoff_ready'
    assert not state_checks('h01', [turn])['not_prematurely_closed']


def test_followup_requires_two_done_actions_and_original_timeout_result():
    from evaluation.run_harness_eval import state_checks
    question = '我已经换过网络，也重新登录了，还是超时，下一步怎么办？'
    state = {'status': 'collecting', 'facts': {'target_system': {'value': 'VPN', 'source': 'user_reported'}},
             'steps': [{'source': 'user_reported', 'status': 'user_reported_done', 'source_quote': question, 'result': '还是超时'},
                       {'source': 'user_reported', 'status': 'user_reported_observation', 'source_quote': '重新登录了', 'result': '重新登录了'}]}
    # One broad quote cannot count as two different completed actions.
    state['steps'][0]['description'] = '换过网络'
    state['steps'][1]['description'] = '重新登录'
    turn = {'question': question, 'chat': {'case_state': state}}
    checks = state_checks('h02', [turn])
    assert checks['network_changed'] and not checks['logged_in_again']
    state['steps'][1].update(status='user_reported_done', source_quote=question, result='还是超时')
    assert all(state_checks('h02', [turn]).values())
    state['steps'][0]['result'] = '已换网络'
    assert not state_checks('h02', [turn])['network_changed']


def test_runtime_failure_detects_http_200_provider_and_budget_errors():
    from evaluation.run_harness_eval import runtime_failure
    assert runtime_failure({'chat': {'harness_events': [{'reason': 'model_or_runtime_error', 'error': 'APIStatusError'}]}})
    assert runtime_failure({'trace': {'trace': {'harness_events': [{'reason': 'model_round_budget_exhausted'}]}}})
    assert runtime_failure({'chat': {'harness': {'status': 'runtime_failed'}}})
    assert not runtime_failure({'chat': {'response': '文档解释 APIStatusError 的含义。'}})


def test_runner_stops_before_paid_judge_and_next_case_on_runtime_error(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    import httpx
    import anthropic
    from evaluation import run_harness_eval as runner
    calls = []
    class Response:
        is_success = True
        def __init__(self, body): self.body = body
        def raise_for_status(self): pass
        def json(self): return self.body
    class Http:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json):
            calls.append('chat')
            return Response({'conv_id': 'c', 'request_id': 'r', 'response': '保守交接文本'})
        async def get(self, url):
            return Response({'trace': {'harness_events': [{'reason': 'model_or_runtime_error', 'error': 'APIStatusError'}]}})
    class Judge:
        async def close(self): calls.append('close')
    async def forbidden_judge(*args):
        raise AssertionError('Provider/runtime failure must not incur judge call')
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: Http())
    monkeypatch.setattr(anthropic, 'AsyncAnthropic', lambda **kwargs: Judge())
    monkeypatch.setattr(runner, 'judge_answer', forbidden_judge)
    report = tmp_path/'stopped.json'
    args = SimpleNamespace(dataset=Path(runner.BACKEND/'evaluation/datasets/harness_v1.json'),
                           report=report, validate_only=False, application_revision='test', repeats=3,
                           api_base='http://offline', only='h01,h02')
    asyncio.run(runner.run(args))
    saved = json.loads(report.read_text(encoding='utf-8'))
    assert calls == ['chat', 'close']
    assert saved['status'] == 'stopped' and saved['summary']['runtime_failures'] == 1
    live = [r for r in saved['results'] if r.get('execution') == 'live']
    assert len(live) == 1 and live[0]['judge_skipped'] == 'runtime_failure'
    assert live[0]['turns'][0]['trace']['trace']['harness_events']


def test_judge_usage_wrapper_retains_usage_without_changing_response():
    import asyncio
    from types import SimpleNamespace
    from evaluation.run_harness_eval import JudgeUsageClient
    answer = SimpleNamespace(usage=SimpleNamespace(model_dump=lambda **kwargs: {'input_tokens': 10, 'output_tokens': 2}))
    seen = []
    async def create(**kwargs):
        seen.append(kwargs)
        return answer
    client = JudgeUsageClient(SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert asyncio.run(client.messages.create(model='offline', tools=['existing-contract'])) is answer
    assert seen == [{'model': 'offline', 'tools': ['existing-contract']}]
    assert client.last_call['usage'] == {'input_tokens': 10, 'output_tokens': 2}
    assert client.last_call['seconds'] >= 0


def test_correction_acceptance_requires_updated_purpose_and_resolved_conflicts():
    from evaluation.run_harness_eval import state_checks
    state = {'facts': {'target_system': {'value': 'GitLab', 'source': 'user_reported'},
                       'purpose': {'value': '查看需求', 'source': 'user_reported'}}, 'conflicts': []}
    turns = [{'question': '请整理刚才的权限申请信息。', 'chat': {'case_state': state}}]
    checks = state_checks('h03', turns)
    assert checks['corrected_system'] and not checks['corrected_purpose']
    state['facts']['purpose']['value'] = '查看项目代码'
    assert all(state_checks('h03', turns).values())
    for field in ['target_system', 'purpose']:
        state['conflicts'] = [{'field': field, 'old_value': '待核实'}]
        assert not state_checks('h03', turns)['correction_resolved']
    state['conflicts'] = []
    state['facts']['purpose']['value'] = '查看项目代码/查看需求'
    assert not state_checks('h03', turns)['corrected_purpose']
