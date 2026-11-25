import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from evaluation import run_domain_eval


def test_domain_runner_stops_after_provider_failure(monkeypatch, tmp_path):
    calls = []
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'conv_id': 'c', 'response': '暂时无法处理', 'agent_type': 'network',
                    'validation': {'status': 'runtime_failed'}}
    class HTTP:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, json):
            calls.append(json['message'])
            return Response()
    monkeypatch.setattr(run_domain_eval.httpx, 'AsyncClient', lambda **kwargs: HTTP())
    args = SimpleNamespace(dataset=Path(__file__).parents[1]/'evaluation/datasets/domain_agents_v1.json',
                           only='', validate_only=False, report=tmp_path/'results.json',
                           model='fake', application_revision='test', api_base='http://fake')
    asyncio.run(run_domain_eval.run(args))
    report = json.loads(args.report.read_text(encoding='utf-8'))
    assert len(calls) == 1
    assert report['status'] == 'stopped'
    assert report['stopped_reason'] == 'runtime_failed'
    assert not report['results'][0]['automatic_checks_passed']


def test_domain_migrations_preserve_original_labels():
    directory = Path(__file__).parents[1]/'evaluation/datasets'
    old = json.loads((directory/'support_v1.json').read_text(encoding='utf-8'))
    new = json.loads((directory/'domain_agents_v1.json').read_text(encoding='utf-8'))
    labels = {case['id']: case['expected_intent'] for case in old['intent_cases']}
    assert len(labels) == 60
    for item in new['legacy_label_migrations']:
        assert labels[item['id']] == item['old']
        assert item['new'] != item['old'] and item['reason']
    assert len({case['id'] for case in new['cases']}) == len(new['cases'])
