import asyncio
from types import SimpleNamespace

from harness.validation import finalize_answer

QUESTION = '普通网页可以打开，但 VPN 一直连接超时，排查顺序是什么？'
TRACE = [{'tool_name': 'search_knowledge_base', 'success': True, 'result_success': True,
          'retrieval': {'evidence': [{'title': 'VPN 连接排查（演示资料）',
              'content': '连接超时按网络、DNS、代理、防火墙和服务状态顺序排查。'}]}}]

class Client:
    def __init__(self, responses):
        self.messages = self
        self.responses = iter(responses)
        self.calls = []
    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def verdict(passed):
    return SimpleNamespace(stop_reason='tool_use', content=[{
        'type': 'tool_use', 'name': 'submit_support_validation',
        'input': {'passed': passed, 'issues': [] if passed else ['缺少服务端状态核验'], 'reason': '核对适用资料'},
    }])


def test_observe_never_calls_model_or_changes_answer():
    client = Client([])
    result = asyncio.run(finalize_answer(QUESTION, '原回答', TRACE, client, 'model', 'observe'))
    assert result.content == '原回答'
    assert result.status == 'observed'
    assert result.extra_calls == 0 and client.calls == []


def test_no_evidence_does_not_invent_sop_or_force_validation():
    client = Client([])
    result = asyncio.run(finalize_answer(QUESTION, '原回答', [], client, 'model', 'enforce'))
    assert result.status == 'skipped' and result.content == '原回答'


def test_successful_answer_checks_once_with_named_tool(monkeypatch):
    monkeypatch.setenv('LLM_THINKING_ENABLED', 'false')
    client = Client([verdict(True)])
    result = asyncio.run(finalize_answer(QUESTION, '正确回答', TRACE, client, 'model', 'enforce'))
    assert result.status == 'passed' and result.extra_calls == 1
    assert client.calls[0]['tool_choice']['name'] == 'submit_support_validation'
    assert client.calls[0]['extra_body']['thinking']['type'] == 'disabled'


def test_failed_answer_repair_rechecks_and_keeps_three_call_budget():
    repair = SimpleNamespace(stop_reason='end_turn', content=[{'type': 'text', 'text': '修正回答'}])
    client = Client([verdict(False), repair, verdict(True)])
    result = asyncio.run(finalize_answer(QUESTION, '漏项回答', TRACE, client, 'model', 'enforce'))
    assert result.content == '修正回答' and result.status == 'repaired'
    assert result.repair_count == 1 and len(client.calls) == 3
    assert next(e for e in result.events if e['event'] == 'answer_candidate')['content'] == '漏项回答'
    assert next(e for e in result.events if e['event'] == 'answer_repair')['content'] == '修正回答'


def test_failed_recheck_returns_unverified_handoff_not_success():
    repair = SimpleNamespace(stop_reason='end_turn', content=[{'type': 'text', 'text': '仍然缺项'}])
    client = Client([verdict(False), repair, verdict(False)])
    result = asyncio.run(finalize_answer(QUESTION, '错误回答', TRACE, client, 'model', 'enforce'))
    assert result.status == 'needs_handoff' and result.escalated
    assert '尚未创建工单' in result.content and result.extra_calls == 3


def test_invalid_judge_never_passes_or_retries_forever():
    client = Client([SimpleNamespace(stop_reason='end_turn', content=[{'type': 'text', 'text': 'ok'}])])
    result = asyncio.run(finalize_answer(QUESTION, '待核验回答', TRACE, client, 'model', 'enforce'))
    assert result.status == 'needs_handoff' and result.extra_calls == 1


def test_failed_retrieval_not_used_as_validation_evidence():
    trace = [{**TRACE[0], 'result_success': False}]
    client = Client([])
    result = asyncio.run(finalize_answer(QUESTION, '原回答', trace, client, 'model', 'enforce'))
    assert result.status == 'skipped'


def test_invalid_timeout_configuration_uses_default(monkeypatch):
    monkeypatch.setenv('NEXO_HARNESS_VALIDATION_TIMEOUT', 'invalid')
    client = Client([verdict(True)])
    result = asyncio.run(finalize_answer(QUESTION, '正确回答', TRACE, client, 'model', 'enforce'))
    assert result.status == 'passed'
    calls = [e for e in result.events if e['event'] == 'validation_model_call']
    assert len(calls) == 1 and calls[0]['latency_ms'] >= 0


def test_contradictory_verdict_is_invalid_not_success():
    response = verdict(True)
    response.content[0]['input']['issues'] = ['仍有问题']
    result = asyncio.run(finalize_answer(QUESTION, '错误回答', TRACE, Client([response]), 'model', 'enforce'))
    assert result.status == 'needs_handoff'


def vpn_case():
    return {'facts': {'target_system': {'value': 'VPN', 'source': 'user_reported'},
                      'issue_description': {'value': '连接超时，网页正常', 'source': 'user_reported'}}}


def test_user_followup_without_vpn_uses_same_case_and_actual_evidence():
    client = Client([verdict(True)])
    result = asyncio.run(finalize_answer('我已经换过网络，也重新登录了，还是超时，下一步怎么办？',
        '承接排查结果', TRACE, client, 'model', 'enforce', case_state=vpn_case()))
    assert result.status == 'passed' and result.extra_calls == 1


def test_followup_without_timeout_uses_known_case_issue():
    client = Client([verdict(True)])
    result = asyncio.run(finalize_answer('还是不行，下一步怎么办？', '承接结果', TRACE,
        client, 'model', 'enforce', case_state=vpn_case()))
    assert result.status == 'passed'


def test_followup_needs_user_reported_case_and_actual_evidence():
    case = vpn_case()
    case['facts']['target_system']['source'] = 'inferred'
    result = asyncio.run(finalize_answer('还是超时，下一步怎么办？', '原回复', TRACE,
        Client([]), 'model', 'enforce', case_state=case))
    assert result.status == 'skipped'
    result = asyncio.run(finalize_answer('还是超时，下一步怎么办？', '原回复', [],
        Client([]), 'model', 'enforce', case_state=vpn_case()))
    assert result.status == 'skipped'


def test_new_topic_and_access_handoff_do_not_inherit_vpn_validation():
    for message in ('换个问题，打印机还是不行', '下一步申请项目权限', '还是不行，转人工吧'):
        result = asyncio.run(finalize_answer(message, '原回复', TRACE,
            Client([]), 'model', 'enforce', case_state=vpn_case()))
        assert result.status == 'skipped'
    case = {**vpn_case(), 'new_topic': True}
    result = asyncio.run(finalize_answer('还是超时，下一步怎么办？', '原回复', TRACE,
        Client([]), 'model', 'enforce', case_state=case))
    assert result.status == 'skipped'
