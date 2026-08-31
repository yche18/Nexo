# IT domain Agent expansion

The first-run findings below are preserved. Subsequent minimal fixes and the
fixed-scope rerun are documented in [runtime fix results](DOMAIN_RUNTIME_FIX_RESULTS.md).

## Scope

Nexo now has General, Technical, Access, Network, Device and Escalation roles.
Network handles connection, Wi-Fi, DNS and VPN faults. Device handles device-level
triage, including uncertain printer or audio-device symptoms; it does not assume
that a hardware root cause is already known. Technical retains application
installation/update, authentication, crashes and general technical support.
Access and deterministic handoff retain their existing responsibilities.

New coarse categories are `network` and `device`; new specific intents are
`network_issue`, `device_failure` and `software_install` (grouped under technical).
The weighted vote, confidence threshold and coarse-group-first selection formula
are unchanged. Separating domains and adding templates can still change the
winning classification. Explicit handoff and critical-priority routing remain.

Six environment entities extend the original six: `operating_system`,
`client_version`, `network_type`, `impact_scope`, `device_type`, `device_model`.
OS/client version reuse existing Harness fact fields. New entities do not earn
routing bonuses. The original three bonuses remain: error code to Technical
(+0.20), requested access to Access (+0.15), target system to General (+0.10).
New-domain keyword matching excludes explicit negative/hypothetical fragments;
this remains a limited rule list rather than a general language parser.

Network reuses existing technical tools and shared knowledge search. Device uses
shared knowledge search only, plus the existing enforce-mode progress tool.
New Skills follow the current complete structure and SOP-first rule. Existing
Skills change only their domain boundaries. The four added default SOPs cover
Wi-Fi, DNS/intranet, printer faults and a computer that will not power on.
There are no asset APIs, ticket integrations or actual device-control operations.

Harness fact/schema lists and applicability lists include the new domains. The
completed-action vocabulary includes basic device actions; provenance, negation,
conflict, budget and result-validation algorithms stay unchanged. Handoff carries
new environment facts and short symptoms from user messages; it does not paste
entire conversations or promote assistant suggestions to new environment facts.

The frontend already renders role/trace data dynamically and needs no source
change. Hybrid RAG, business tool implementations, memory, Monitor formulas,
model round limits, model token limits and 600-second SDK timeout are unchanged.
The work is on `feat/it-domain-agents`, based on `095d167`.

## Offline verification

- 312 tests passed using the existing production Python dependencies in an
  isolated container with network disabled.
- Scope audit confirms unchanged business tools, hybrid retrieval/reranking,
  memory, Monitor, frontend and old datasets. Weighted intent voting,
  BaseAgent execution, Monitor fallback and execution fallback match the baseline.
- All 60 original cases retain identical primary/supporting routes when supplied
  with their original frozen intent labels. This is route regression, not fresh
  model classification accuracy.
- The separate versioned migration list moves only i06 (installation) to Technical,
  i49 (intranet) to Network, i51/i52 (printer/microphone triage) to Device. The
  original dataset is untouched. i04 policy remains General; i39 authentication
  and i47 crash-after-update remain Technical in this fixed-label check.
- Original Pattern top-one labels remain unchanged for these 60 cases. Local
  n-gram Embedding top-one labels change on 10 cases, including some unrelated
  low-confidence candidates. The complete 60-case model suite was not rerun.

## First bounded Flash run

The frozen dataset is `datasets/domain_agents_v1.json`: 8 scenarios / 10 user
turns, one attempt per scenario. An isolated enforce service used its own local
Chroma database, Elasticsearch index and Redis database; ordinary services were
not replaced. All model roles used `deepseek-flash`, thinking disabled. There were
no paid evaluation-Judge calls. Existing runtime VPN verification still ran.

The runner stopped on `model_round_budget_exhausted` after 5 scenarios / 7 user
turns. It did not repeat failures or continue the remaining three scenarios.
Automatic routing/handoff checks passed in 5 of 7 turns; these are not semantic
answer-accuracy measurements. The isolated service was stopped afterward.

| Scenario | Observed result |
| --- | --- |
| n01 VPN first report and follow-up | First turn reached Network and recorded VPN/Windows facts. Follow-up was correctly classified network_issue, but Monitor sent it to General. Both attempted actions and their unsuccessful results were retained. Expected continuing Network route failed. |
| n02 Wi-Fi | Reached Network, searched knowledge, preserved OS/network/scope, gave ordered checks. |
| n03 VPN permission | Reached Access, checked existing fields and explained approval boundaries without claiming authorization. |
| n04 Printer and handoff | Reached Device and searched its SOP. Explicit handoff retained printer type, HP M227 model, jam symptom and scope. |
| n05 No power | Correct device_failure classification, but Monitor sent it to General. Three model turns all requested tools; no final answer remained within the existing budget, so deterministic handoff ran. Failed. |
| n06 installation / n07 crash / n08 mixed request | Not run after the runtime stop; offline route coverage exists, live answer acceptance remains pending. |

### Failure analysis and limits

The Monitor problem predates this change: BaseAgent increments total at request
start and success only when the request finishes. A periodic sample while the
first request is still running observes 0/1 success and applies a 0.5 penalty.
That meets the unchanged fallback threshold. A successful completion does not
clear the penalty until the next Monitor sample. The logs show this for Network
at 11:40:06/16 UTC and Device at 11:41:06 UTC. Adding new roles to the existing
Monitor makes them subject to this same behavior. No threshold/statistics fix
was bundled into domain expansion.

For n05, General used search/context inspection in model turn 1, required-fields
in turn 2, and progress recording in turn 3. The existing three-turn budget then
ended with tool_use rather than final text. This is not a provider/balance error,
failed RAG retrieval, or evidence that Device itself could not answer: Device was
not executed in this turn. No budget increase or best-run retry was made.

Manual review also found that n01 named the full VPN SOP order but did not turn
all checks into explicit next actions; its verifier accepted an overstrong
follow-up claim that self-checks were exhausted. This known narrow verifier limit
remains. The printer answer mentioned typical access locations without a retrieved
model-specific manual; do not treat that as verified HP M227-specific guidance.
The old broad Access keyword `开通` can still trigger collaboration in statements
such as "VPN 权限已经开通". These findings are retained rather than reported as
complete business acceptance.

The test-only SDK recorder observed 43 completed Flash message calls:
`input_tokens=44161`, `cache_read_input_tokens=60032`, `output_tokens=6843`,
`cache_creation_input_tokens=0`; provider errors=0. These are the returned SDK
fields, not a currency bill, and exclude a separately billed reranking service.
Raw answers, tool traces, every usage row, frozen source hashes and scope audit
are retained locally under `data/eval/domain_agents_v1/` (Git-ignored).

## Run and knowledge migration

The usual service and local `.env` remain unchanged (observe mode). A new empty
knowledge database receives all 10 default documents. An existing database does
not auto-load the four added SOPs: import the new documents using the existing
knowledge administration interface or `POST /knowledge/add`, which publishes to
both stores. Do not clear the existing knowledge base. Tests preserve all original
six default document IDs and content.

The bounded runner preserves existing report files and stops on runtime/provider
errors; it does not add model Judge calls:

```powershell
# From backend, against a deliberately prepared isolated service.
python -m evaluation.run_domain_eval --validate-only --report ../data/eval/domain_agents_v1/check.json
python -m evaluation.run_domain_eval --api-base http://127.0.0.1:18009 --report ../data/eval/domain_agents_v1/new-run.json
```

The second command makes paid chat calls; run only the intended small scope.
Use `--only n06,n07,n08` to select remaining cases after reviewing the failure
conditions. Automatic checks do not replace per-turn manual answer review.
