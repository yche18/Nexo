"""Bounded HTTP acceptance for new IT domains; no paid evaluation Judge or retries.

Keep all raw answers/traces and review each turn against the frozen criteria.
Automatic checks cover routing and literal handoff fields, not semantic accuracy.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx

from evaluation.run_harness_eval import runtime_failure
from evaluation.run_support_eval import check_turn

BACKEND = Path(__file__).resolve().parents[1]


async def run(args):
    raw = args.dataset.read_bytes()
    data = json.loads(raw)
    cases = data['cases']
    if args.only:
        selected = args.only.split(',')
        unknown = set(selected) - {case['id'] for case in cases}
        if unknown:
            raise ValueError('Unknown case IDs: ' + ','.join(sorted(unknown)))
        cases = [case for case in cases if case['id'] in selected]
    if args.validate_only:
        print(json.dumps({'scenarios': len(cases), 'turns': sum(len(c['turns']) for c in cases)}))
        return
    if args.report.exists():
        raise ValueError('Report exists; preserve it and choose a new filename')
    report = {'dataset': data['version'], 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'model': args.model,
              'application_revision': args.application_revision, 'results': [],
              'limits': ['One attempt per scene, sequentially; no evaluation Judge calls.',
                         'Checks are routing/literal assertions, not answer accuracy.',
                         'Trace usage covers only instrumented calls, not the full account bill.',
                         'Every turn needs manual review against review_points.']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save()
    async with httpx.AsyncClient(timeout=180.0) as http:
        for case in cases:
            row = {'id': case['id'], 'review_points': case['review_points'], 'turns': []}
            report['results'].append(row)
            user_id = 'domain-eval-' + uuid.uuid4().hex
            conv_id = None
            try:
                for spec in case['turns']:
                    started = time.monotonic()
                    response = await http.post(args.api_base.rstrip('/')+'/chat', json={
                        'message': spec['message'], 'user_id': user_id, 'conv_id': conv_id})
                    response.raise_for_status()
                    body = response.json()
                    conv_id = body['conv_id']
                    turn = {'question': spec['message'], 'chat': body,
                            'seconds': round(time.monotonic()-started, 3),
                            'checks': check_turn(spec, body)}
                    row['turns'].append(turn)
                    if body.get('request_id'):
                        trace = await http.get(args.api_base.rstrip('/')+'/trace/tool/'+body['request_id'])
                        trace.raise_for_status()
                        turn['trace'] = trace.json()
                    failure = runtime_failure(turn)
                    if failure:
                        report['stopped_reason'] = failure
                    save()
                    print(json.dumps({'id': case['id'], 'turn': len(row['turns']),
                                      'agent': body.get('agent_type'),
                                      'checks_passed': all(c['passed'] for c in turn['checks']),
                                      'runtime_failure': failure}, ensure_ascii=False), flush=True)
                    if failure:
                        break
            except Exception as exc:
                row['error'] = type(exc).__name__
                report['stopped_reason'] = type(exc).__name__
            row['automatic_checks_passed'] = (len(row['turns']) == len(case['turns'])
                and not row.get('error') and not report.get('stopped_reason')
                and all(c['passed'] for t in row['turns'] for c in t['checks']))
            save()
            if report.get('stopped_reason'):
                break
    report['status'] = 'stopped' if report.get('stopped_reason') else 'completed'
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BACKEND/'evaluation/datasets/domain_agents_v1.json')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--api-base', default='http://127.0.0.1:18009')
    parser.add_argument('--only', default='')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--application-revision', default='uncommitted')
    parser.add_argument('--validate-only', action='store_true')
    asyncio.run(run(parser.parse_args()))
