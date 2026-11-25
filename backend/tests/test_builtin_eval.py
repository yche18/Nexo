import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.evaluator import DEFAULT_INTENT_CASES, EndToEndEvaluator, LLMJudge, QualityScores


def run(value):
    return asyncio.run(value)


class Client:
    def __init__(self, content, stop_reason='tool_use'):
        self.messages = self
        self.content = content
        self.stop_reason = stop_reason
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.content, stop_reason=self.stop_reason)


def verdict(**updates):
    data = dict(relevance=0.9, accuracy=0.9, completeness=0.8, helpfulness=0.9, reason='本轮已补齐信息并给出下一步')
    data.update(updates)
    return {'type': 'tool_use', 'name': 'submit_evaluation', 'input': data}


def test_judge_uses_forced_tool_and_keeps_reason():
    client = Client([verdict()])
    result = run(LLMJudge(client, 'deepseek-flash').judge('下一步？', '可以确认建单', '已确认系统和用途'))
    assert not result.judge_failed and result.completeness == 0.8
    assert result.reason and result.evidence['content']
    assert client.calls[0]['tool_choice'] == {'type': 'tool', 'name': 'submit_evaluation'}
    assert '已确认系统和用途' in client.calls[0]['messages'][0]['content']


@pytest.mark.parametrize('content,stop', [
    ([{'type':'text','text':'{"relevance": 1}'}], 'end_turn'),
    ([verdict()], 'max_tokens'),
    ([verdict(), verdict()], 'tool_use'),
    ([verdict(accuracy=True)], 'tool_use'),
    ([verdict(completeness=1.5)], 'tool_use'),
    ([verdict(completeness=float('nan'))], 'tool_use'),
    ([verdict(reason='')], 'tool_use'),
    ([{'type':'tool_use','name':'submit_evaluation','input':{'relevance':1}}], 'tool_use'),
])
def test_invalid_judge_output_is_not_a_quality_score(content, stop):
    result = run(LLMJudge(Client(content, stop), 'deepseek-flash').judge('q', 'a'))
    assert result.judge_failed and result.error


class Judge:
    def __init__(self, invalid=False):
        self.contexts = []
        self.invalid = invalid

    async def judge(self, question, answer, context=None):
        self.contexts.append(context)
        return QualityScores(.9, .9, .9, .9, judge_failed=self.invalid, error='invalid' if self.invalid else None, reason='complete')


def evaluator(tmp_path, runner):
    result = EndToEndEvaluator(None, None, 'offline-test-key', model='deepseek-flash',
                              baseline_path=str(tmp_path/'baseline.json'), chat_runner=runner)
    result._judge = Judge()
    return result


def test_multiturn_uses_chat_state_and_isolates_user_and_run(tmp_path):
    calls = []
    states = {}
    async def chat(message, user_id, conv_id):
        calls.append((message, user_id, conv_id))
        state = states.setdefault(conv_id, [])
        state.append(message)
        return {'response':'回答', 'agent_type':'access','intent':'permission_request',
                'case_state':{'facts':list(state)},'tools_used':[]}
    ev = evaluator(tmp_path, chat)
    cases = [{'turns':['目标系统是项目平台','下一步？'], 'user_id':'real-user','conv_id':'real-conversation'}]
    first = run(ev.run(dialog_cases=cases))
    second = run(ev.run(dialog_cases=cases))
    assert calls[0][1:] == calls[1][1:]
    assert calls[0][1:] != calls[2][1:]
    assert all(user.startswith('eval_') and conv.startswith('eval_') for _,user,conv in calls)
    assert first.results[1].metadata['chat']['case_state']['facts'] == ['目标系统是项目平台','下一步？']
    assert '目标系统是项目平台' in ev._judge.contexts[1]
    assert not first.metadata['baseline_comparable'] and second.metadata['baseline_comparable']
    assert first.total == 2 and first.pass_rate == 1


def test_legacy_report_preserved_and_only_comparable_versions_regress(tmp_path):
    original = b'{"timestamp":"old","avg_scores":{"completeness":1.0}}'
    (tmp_path/'baseline.json').write_bytes(original)
    async def chat(*args):
        return {'response':'a'}
    ev = evaluator(tmp_path, chat)
    first = run(ev.run(dialog_cases=[{'question':'q'}]))
    assert first.regressions == [] and not first.metadata['baseline_comparable']
    archived = list((tmp_path/'history').glob('*.json'))
    assert any(p.read_bytes() == original for p in archived)
    snapshot = (tmp_path/'history'/f"{first.metadata['run_id']}.json").read_bytes()
    changed = run(ev.run(dialog_cases=[{'question':'different'}]))
    assert not changed.metadata['baseline_comparable']
    assert (tmp_path/'history'/f"{first.metadata['run_id']}.json").read_bytes() == snapshot
    assert ev._detect_regressions({'completeness':.4})


def test_failed_judge_kept_in_denominator_excluded_from_average(tmp_path):
    async def chat(*args):
        return {'response':'a'}
    ev = evaluator(tmp_path, chat)
    ev._judge = Judge(invalid=True)
    report = run(ev.run(dialog_cases=[{'question':'q'}]))
    assert report.total == 1 and report.passed == 0 and report.avg_scores == {}
    assert report.results[0].scores == {} and report.metadata['judge_invalid'] == 1
    assert not any('达到建议阈值' in r for r in report.recommendations)


def test_current_labels_and_legacy_labels_are_separate():
    labels = {c.message:c.expected_intent for c in DEFAULT_INTENT_CASES}
    legacy = json.loads((Path(__file__).parents[1]/'evaluation/datasets/builtin_v1.json').read_text(encoding='utf-8'))
    old = {c['message']:c['expected_intent'] for c in legacy['intent_cases']}
    assert labels['办公软件怎么安装？'] == 'software_install' and old['办公软件怎么安装？'] == 'query'
    assert labels['帮我配置打印机'] == 'device' and old['帮我配置打印机'] == 'request'
    assert len(labels) == len(old) == 11


def test_api_evaluation_adapter_calls_normal_chat(monkeypatch):
    from api import main
    seen = []
    async def chat(req):
        seen.append(req)
        return SimpleNamespace(model_dump=lambda **kwargs: {'response':'正常状态链路'})
    monkeypatch.setattr(main, 'chat', chat)
    result = run(main._evaluation_chat('下一步', 'eval-user', 'eval-conversation'))
    assert result['response'] == '正常状态链路'
    assert seen[0].message == '下一步' and seen[0].conv_id == 'eval-conversation'
