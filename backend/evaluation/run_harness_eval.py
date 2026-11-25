"""Harness end-to-end pilot. Preserve every turn and repeat; no best-run selection.

Runs 8 HTTP scenarios; 4 fault/state scenarios require deterministic tests and are
reported as not_run here. Old RAG labels and judge protocol remain unchanged.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from evaluation.run_rag_eval import judge_answer, JUDGE_PROTOCOL, JUDGE_CONTRACT_SHA256


def validate_dataset(data):
    assert data['version'] == 'harness-v1'
    assert len(data['cases']) == 12
    assert len({x['id'] for x in data['cases']}) == 12
    for case in data['cases']:
        assert case['execution'] in ('live', 'deterministic')
        assert case['required_points']
        if case['execution'] == 'live':
            assert case['turns'] and case['forbidden_claims']


def summarize(rows):
    live = [r for r in rows if r.get('execution') == 'live']
    valid = [r for r in live if r.get('judge_valid')]
    return {'live_trials': len(live), 'judge_valid': len(valid),
            'passed': sum(bool(r.get('answer_passed')) for r in live),
            'chat_errors': sum(bool(r.get('chat_error')) for r in live),
            'judge_errors': sum(bool(r.get('judge_error')) for r in live),
            'pass_rate_all_trials': sum(bool(r.get('answer_passed')) for r in live)/len(live) if live else None,
            'not_run_deterministic': len([r for r in rows if r.get('status') == 'not_run']),
            'state_checks': len([r for r in live if 'state_check' in r]),
            'state_checks_passed': sum(r.get('state_check') is True for r in live),
            'acceptance_passed': sum(r.get('acceptance_passed') is True for r in live),
            'runtime_failures': sum(bool(r.get('runtime_error')) for r in live),
            'chat_seconds': round(sum(t.get('seconds', 0) for r in live for t in r.get('turns', [])), 3)}


class JudgeUsageClient:
    """Record only evaluation Judge usage; this is not total chat/model cost."""
    def __init__(self, client):
        self.client = client
        self.messages = self
        self.last_call = {}

    async def create(self, **kwargs):
        started = time.monotonic()
        self.last_call = {}
        try:
            response = await self.client.messages.create(**kwargs)
            usage = getattr(response, 'usage', None)
            if hasattr(usage, 'model_dump'):
                usage = usage.model_dump(mode='json')
            self.last_call['usage'] = usage if isinstance(usage, dict) else None
            return response
        except Exception as exc:
            self.last_call['error'] = type(exc).__name__
            raise
        finally:
            self.last_call['seconds'] = round(time.monotonic() - started, 3)

    async def close(self):
        await self.client.close()


def state_checks(case_id, turns):
    """Judge quality and task-state correctness are independent acceptance gates."""
    state = turns[-1]['chat'].get('case_state') or {}
    if not state:
        return {}  # Baseline does not expose task state.
    facts = state.get('facts', {})
    target = facts.get('target_system', {})
    checks = {}
    if case_id in ('h01', 'h02'):
        checks['vpn_user_fact'] = target.get('source') == 'user_reported' and 'VPN' in str(target.get('value', '')).upper()
        checks['not_prematurely_closed'] = state.get('status') in ('collecting', 'waiting_user')
    if case_id == 'h01':
        question = turns[0]['question']
        issue = facts.get('issue_description', {})
        quote = issue.get('source_quote', '')
        checks['issue_quote_preserved'] = bool(quote) and issue.get('source') == 'user_reported' and quote in question and '超时' in quote
    elif case_id == 'h02':
        question = turns[-1]['question']
        done = [step for step in state.get('steps', []) if step.get('status') == 'user_reported_done'
                and step.get('source') == 'user_reported']
        action_matches = []
        for label, words in [('network_changed', ('换', '网络')), ('logged_in_again', ('重新', '登录'))]:
            matches = [step for step in done if all(word in step.get('description', '') for word in words)
                       and all(word in step.get('source_quote', '') for word in words)]
            action_matches.append(matches)
            checks[label] = any(step.get('source_quote') in question and '超时' in str(step.get('result', ''))
                                and str(step.get('result', '')) in question for step in matches)
        checks['two_distinct_actions'] = any(first is not second for first in action_matches[0] for second in action_matches[1])
    elif case_id == 'h03':
        checks['corrected_system'] = target.get('source') == 'user_reported' and 'GitLab' in str(target.get('value', ''))
        purpose = facts.get('purpose', {})
        checks['corrected_purpose'] = purpose.get('source') == 'user_reported' and '查看项目代码' in str(purpose.get('value', '')) and '查看需求' not in str(purpose.get('value', ''))
        checks['correction_resolved'] = not any(item.get('field') in ('target_system', 'purpose')
                                                for item in state.get('conflicts', []))
    elif case_id == 'h04':
        checks['handoff_ready'] = state.get('status') == 'handoff_ready'
    elif case_id == 'h05':
        initial = turns[0]['chat'].get('case_state') or {}
        checks['new_case'] = bool(initial.get('case_id')) and initial['case_id'] != state.get('case_id')
    return checks


def runtime_failure(turn):
    """Stop paid evaluation on provider/runtime errors even when /chat returns 200."""
    provider_errors = {'APIStatusError', 'RateLimitError', 'AuthenticationError', 'PermissionDeniedError',
                       'APITimeoutError', 'APIConnectionError', 'InternalServerError'}
    def inspect(value):
        if isinstance(value, dict):
            for key in ('error', 'reason', 'status'):
                item = value.get(key)
                if isinstance(item, str) and (item in provider_errors or item in (
                        'runtime_failed', 'model_or_runtime_error', 'model_round_budget_exhausted')):
                    return item
            for item in value.values():
                found = inspect(item)
                if found:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = inspect(item)
                if found:
                    return found
        return None
    found = inspect({'chat': turn.get('chat', {}), 'trace': turn.get('trace', {})})
    if found:
        return found
    answer = turn.get('chat', {}).get('response', '')
    if '抱歉，处理您的请求时出现问题' in answer:
        return 'runtime_error_placeholder'
    return None


async def run(args):
    import httpx
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(BACKEND.parent / '.env')
    raw = args.dataset.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    validate_dataset(data)
    if args.validate_only:
        print('12 scenarios valid: 8 live and 4 deterministic-only.')
        return
    if args.report.exists():
        raise ValueError('Report already exists; choose a new path to retain prior results')
    cfg = {'api_key': os.getenv('ANTHROPIC_API_KEY') or os.getenv('LLM_API_KEY') or ''}
    base = os.getenv('ANTHROPIC_BASE_URL') or os.getenv('LLM_BASE_URL')
    if base:
        cfg['base_url'] = base
    model = os.getenv('ANTHROPIC_MODEL') or os.getenv('LLM_MODEL') or 'claude-3-5-sonnet-20241022'
    judge = JudgeUsageClient(anthropic.AsyncAnthropic(**cfg, max_retries=0, timeout=60))
    corpus = json.loads((BACKEND/'evaluation/datasets/rag_support_v2.json').read_text(encoding='utf-8-sig'))['corpus']
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'dataset': data['version'],
              'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'application_revision': args.application_revision,
              'model': model, 'judge_protocol': JUDGE_PROTOCOL, 'judge_contract_sha256': JUDGE_CONTRACT_SHA256,
              'repeats': args.repeats, 'thinking_env': os.getenv('LLM_THINKING_ENABLED'), 'sdk_version': anthropic.__version__, 'results': [], 'limits': [
                  'Synthetic pilot; not production benchmark or independently human annotated.',
                  'HTTP runs do not inject tool/Composer faults or force compaction; those scenarios require deterministic tests.',
                  'Isolated user and conversation per trial; shared server cache/monitor state is retained and must be recorded in comparison setup.',
                  'Semantic judge remains fallible; retain manual review of VPN and risky advice.',
                  'judge_call usage covers evaluation Judge only, not total chat, retrieval, verifier or account cost.',
              ]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report['summary'] = summarize(report['results'])
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    stop_batch = False
    async with httpx.AsyncClient(timeout=660) as http:
        for case in data['cases']:
            if case['execution'] != 'live':
                report['results'].append({'id': case['id'], 'execution': 'deterministic', 'status': 'not_run', 'reason': case['note']})
        for repeat in range(1, args.repeats+1):
            if stop_batch:
                break
            for case in data['cases']:
                if case['execution'] != 'live' or (args.only and case['id'] not in args.only.split(',')):
                    continue
                row = {'id': case['id'], 'execution': 'live', 'repeat': repeat, 'turns': [], 'judge_valid': False, 'answer_passed': False}
                report['results'].append(row)
                ident = 'harness-' + uuid.uuid4().hex
                conv_id = None
                try:
                    for question in case['turns']:
                        started = time.monotonic()
                        response = await http.post(args.api_base.rstrip('/')+'/chat', json={'message':question,'user_id':ident,'conv_id':conv_id})
                        response.raise_for_status()
                        body = response.json()
                        conv_id = body['conv_id']
                        turn = {'question':question, 'chat':body, 'seconds':round(time.monotonic()-started,3)}
                        row['turns'].append(turn)
                        if body.get('request_id'):
                            trace = await http.get(args.api_base.rstrip('/')+'/trace/tool/'+body['request_id'])
                            turn['trace'] = trace.json() if trace.is_success else {'error':trace.status_code}
                        failure = runtime_failure(turn)
                        if failure:
                            row['runtime_error'] = failure
                            row['judge_skipped'] = 'runtime_failure'
                            report['stopped_reason'] = failure
                            stop_batch = True
                        save()
                        if stop_batch:
                            break
                    if stop_batch:
                        break
                    payload = {'question':'\n'.join(case['turns']), 'answer':row['turns'][-1]['chat']['response'],
                               'required_points':case['required_points'], 'forbidden_claims':case['forbidden_claims'],
                               'reference_corpus':corpus, 'conversation':[{'question':t['question'],'answer':t['chat']['response']} for t in row['turns']],
                               'answerable_from_corpus':case['id'] != 'h06'}
                    row.update(await judge_answer(judge, model, payload))
                    row['judge_call'] = dict(judge.last_call)
                    state = row['turns'][-1]['chat'].get('case_state') or {}
                    row['state_observed'] = bool(state)
                    row['final_case_status'] = state.get('status')
                    checks = state_checks(case['id'], row['turns'])
                    if checks:
                        row['state_checks'] = checks
                        row['state_check'] = all(checks.values())
                    row['acceptance_passed'] = bool(row['answer_passed']) and row.get('state_check', True)
                    if row.get('judge_error') and row['judge_error'] != 'ValueError':
                        report['stopped_reason'] = 'judge_provider_error:' + row['judge_error']
                        stop_batch = True
                except Exception as exc:
                    row['chat_error'] = type(exc).__name__
                    row['judge_skipped'] = 'chat_error'
                    report['stopped_reason'] = 'chat_error:' + type(exc).__name__
                    stop_batch = True
                save()
                print(json.dumps({'id':case['id'],'repeat':repeat,'passed':row['answer_passed'],'judge_valid':row['judge_valid'],'chat_error':row.get('chat_error')},ensure_ascii=False),flush=True)
                if stop_batch:
                    break
    await judge.close()
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    report['status'] = 'stopped' if stop_batch else 'completed'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BACKEND/'evaluation/datasets/harness_v1.json')
    parser.add_argument('--report', type=Path, default=BACKEND.parent/'data/eval/harness'/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'))
    parser.add_argument('--api-base', default='http://localhost:18000')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--application-revision', default=None)
    parser.add_argument('--only', default='')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    asyncio.run(run(args))
