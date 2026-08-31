# Agent Harness v1

Nexo's application runtime keeps model/tool execution, task state, delivery checks
and evaluation trace together. It reuses the existing agents and plain Python
functions; no new agent framework, model service or database is required.

## Scope and modes

| Setting | Behavior |
| --- | --- |
| `NEXO_HARNESS_MODE=off` | Original model/tool behavior; no Harness events or case storage |
| `observe` (default) | Same model requests and answers; adds trace and applicability records without additional model calls |
| `enforce` | Adds task progress, tool budget, relevant SOP validation and controlled failure handoff |
| `NEXO_HARNESS_MAX_TOOL_CALLS=12` | Tool invocation cap per Agent |
| `NEXO_HARNESS_VERIFY_VPN=true` | Enables the narrow VPN-timeout SOP verifier in enforce mode |
| `NEXO_HARNESS_VALIDATION_TIMEOUT=60` | Timeout for each added verification/repair request |

Edit the local `.env`, then recreate the backend container to apply configuration.
The existing three tool-interaction rounds per Agent and normal 600 second SDK
timeout remain. If all three rounds end in tool use, enforce allows at most one
additional answer-only call using the returned results, with tool_choice=none
and SDK retries disabled. off/observe keep the original three-call limit. A qualifying answer adds at most one check, one repair and one recheck;
the added calls disable automatic SDK retries. The same configured model serves
business requests and the verifier. Evaluation Judge calls are separate.

The minimal runtime fixes leave intent recognition, entity extraction/scoring,
domain routing, existing tool implementations, hybrid RAG, memory compression
and business Skills unchanged. Agent request statistics now count completed
successes/failures, excluding pending and cancelled requests; Monitor formulas
and fallback thresholds are unchanged. See [runtime fix results](DOMAIN_RUNTIME_FIX_RESULTS.md).

## Request flow

1. Recognize the intent with the original inputs and route normally.
2. In enforce mode, load a Redis case keyed by user/conversation. Explicit topic
   changes start a new case and exclude old untagged summaries from execution context.
3. Run the existing Agent loop. A request-local `record_support_progress` tool stages
   user-quoted facts, actions, outcomes, pending questions and next actions.
4. Merge parallel updates before composing an answer. A General fallback receives
   already returned evidence and staged progress. All delivered answers reach the
   same finalization function.
5. Check applicability using actual retrieved VPN SOP evidence. Enforce performs
   semantic validation and at most one repair. All-Agent failure skips semantic
   repair and prepares deterministic handoff information.
6. Save the case using Redis WATCH/version checking, then append chat memory.
   Competing updates return 409 instead of silently overwriting the snapshot.

## State contract

Facts require literal user provenance. `issue_description` can retain the user's
original report while keeping the model's paraphrase as a label; at most four prior
reports are retained. Other unsupported paraphrases remain unverified candidates.
Explicit corrections use the surrounding source sentence. Invalid fact/step items
are skipped with warnings without discarding valid items. Invalid top-level state
commands still reject the update.

Steps distinguish proposed actions, user-reported observations and user-reported
completed actions. The stored source context retains qualifiers. Outcomes require
literal user text; an unknown outcome stays empty. Tool execution is not a
user-performed action. Only actual returned chunk IDs become evidence references.
A user must explicitly confirm recovery before the case can become `resolved`.

In enforce mode, handoff uses user messages and confirmed case facts instead of
assistant prose or untagged summaries. It excludes conflicting values and prevents
unverified candidate fields from returning through legacy history parsing.
It can include user-reported completed steps and outcomes. A direct handoff also
preserves the current user report verbatim as `latest_user_report`, without
pretending that a progress tool parsed it or that the system performed an action.
This closes the path where routing bypasses the professional Agent loop. It does not create a
ticket, notify anyone, grant access or modify a device.

## Trace and evaluation

The existing workbench trace panel displays case status, validation outcome,
repair count and Harness events. Successful Agent model calls record SDK usage
and latency; verifier calls and the evaluation Judge also record available usage.
These records do not cover every intent, rewrite, profile, summary or Composer
call, so their sum is not the full account bill. Cache usage fields are retained
as supplied by the provider.

The versioned dataset has eight live scenarios and four deterministic scenarios.
The live runner keeps every turn, tool trace, raw Judge result and state check.
`answer_passed` is the existing answer-quality verdict; `acceptance_passed` also
requires configured state checks. A provider/runtime error stops the remaining
batch and avoids an unnecessary Judge request. Existing reports cannot be overwritten.

Example from the backend directory (use a new report filename for every run):

```powershell
python -m evaluation.run_harness_eval --validate-only
python -m evaluation.run_harness_eval --api-base http://localhost:18000 --only h01,h02 --repeats 1 --report ../data/eval/harness/manual-run.json
```

Use an isolated enforce backend for acceptance tests. Begin with targeted cases,
then expand only where needed; repeated full-suite runs incur model charges.
Offline verification uses `python -m pytest tests -q` with backend dependencies
and pytest available. Current results and manual findings are in
[HARNESS_RESULTS.md](HARNESS_RESULTS.md).

## Boundaries

- Progress recording is model-selected and may omit details. Source matching proves
  provenance, not complete semantic truth. Finite Chinese rules can conservatively
  reject unfamiliar wording; they are not a general language parser.
- The verifier covers a registered VPN-timeout SOP when applicable evidence was
  actually retrieved. It is not a general safety filter, and Judge verdicts can miss
  omissions or risky advice. Manual review remains necessary.
- User/conversation keys provide logical isolation, not authentication. The API
  retains its existing user-ID trust boundary.
- Case CAS and chat-memory append are not one transaction. Concurrent callers may
  already have spent model calls before a version conflict. This is continuation
  between saved turns, not arbitrary process checkpointing or exactly-once execution.
- After a topic switch, untagged summaries are excluded. Recorded facts survive
  compression, but details never recorded in the case may be lost.
- No overall accuracy, cost reduction or latency improvement is claimed from the
  small synthetic pilot, especially across Pro and Flash model changes.
