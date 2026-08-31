# Minimal Monitor and answer-finish fixes

## Scope

Based on `468edcf` on `feat/it-domain-agents`, this change modifies only two
production paths:

1. `BaseAgent.handle` counts a request when it succeeds or fails, together with
   its duration and outcome. Pending/cancelled requests no longer lower success
   rate. Monitor formulas, thresholds, sampling and routing remain unchanged.
2. `run_agent` retains the original three tool-interaction rounds. Only enforce
   mode may add one answer-only request after the third tool result. It preserves
   the complete history, evidence and staged progress, requests tool_choice=none,
   disables SDK retries for that added request, and rejects further tool calls,
   empty text or truncated output without executing any additional tool.

Normal answers in rounds 1-3 and all off/observe behavior remain unchanged.
The original profile max_tokens, timeout, thinking configuration, tool budget,
intent/route/entity logic, RAG, Skills, knowledge documents and state validation
were not changed. Existing failure/handoff behavior still handles unsuccessful
execution. A comment and relevant documentation were synchronized afterward.

The installed Anthropic SDK 0.40.0 does not declare none in its older type union;
the existing extra_body mechanism sends it without upgrading dependencies and
preserves the thinking setting. [DeepSeek's Anthropic compatibility documentation](https://api-docs.deepseek.com/guides/anthropic_api/)
documents tool_choice none. SDK MockTransport tests check the actual serialized
request and verify that an added-call HTTP 503 is not automatically retried.

## Offline verification

333 tests passed in the existing production dependency image with networking
disabled. New cases cover:

- Monitor sampling while a request is pending, then normal professional routing;
- genuine failures retaining the original penalty and fallback;
- concurrent pending requests and cancellation;
- exactly one finish request retaining the third tool result and progress;
- no new tool execution even if the finish response requests a tool;
- empty/truncated finish responses, provider failures and HTTP 402;
- unchanged off/observe limits and no extra call for normal early answers;
- real SDK serialization with tools/history retained, thinking disabled and
  max_retries=0 for the finish request.

An independent code review found no blocking issue. Source/AST audit confirms
23 protected areas, including the original three-round loop, are unchanged.

## Fixed-scope Flash run

The unchanged `domain_agents_v1.json` dataset selected n01,n05,n06,n07,n08:
5 scenarios, 6 user messages. Each scenario ran once, sequentially, in an isolated
enforce service with its own data stores. No evaluation Judge or best-run retry
was added. Existing runtime VPN verification remained enabled.

All 5 scenario routing/runtime checks passed, with no monitor_fallback,
model-round exhaustion, provider error or deterministic failure handoff.
This is not a statement that all answers passed semantic quality checks.

| Scenario | Route/runtime result | Manual answer review against frozen criteria |
| --- | --- | --- |
| n01 VPN first report + follow-up | Both turns stayed Network; prior attempts were acknowledged without repeating them | Incomplete: DNS/service-status names appear without sufficiently explicit verification actions. The unchanged verifier accepted the response. |
| n05 Laptop cannot power on | Device answered normally, instead of Monitor sending it to General | Failed the frozen battery constraint: the answer conditionally suggests removing an externally removable battery, then says battery operations are outside self-service. No Skill or content-policy fix was bundled here. |
| n06 Installation | Technical | Approved channels, permission and backup boundaries covered; no claim to install on the user's behalf. |
| n07 Crash after update | Technical | Addresses the crash instead of treating the update background as a fresh installation request. |
| n08 VPN + GitLab permission | Network + Access; no extra Technical | Both needs addressed without claiming actual authorization or notification. State conservatively retained an unsupported target-system paraphrase as a candidate. |

These manual judgments are about these specific outputs, not a benchmark
accuracy percentage. The VPN completeness and device-advice findings remain
outside the two approved runtime fixes; their original criteria were not relaxed.

### Important coverage boundary

All business Agent executions naturally finished in 2-3 rounds. There were zero
answer_finish_started events and zero tool_choice none HTTP requests in this
batch. Thus the live run verifies the Monitor regression and normal operation,
but does not exercise the newly added fourth-call provider path. That path has
offline/SDK coverage and documented protocol support; no extra paid request was
made merely to force it to trigger. Named submit_support_validation requests are
existing VPN verification calls and must not be counted as answer-finish calls.

### Usage and retained evidence

The test-only recorder observed 45 completed calls, all deepseek-flash, with no
provider errors. Returned SDK fields:

- input_tokens: 39,647
- cache_read_input_tokens: 70,144
- output_tokens: 8,965
- cache_creation_input_tokens: 0

These are SDK usage fields, not a currency bill or reranking-service billing.
The added answer-finish branch contributed zero calls in this particular run.

Full records are retained locally, excluded from Git:

- Previous run: `data/eval/domain_agents_v1/live-first/`
- This run: `data/eval/domain_agents_v1/runtime-fix-first/`
- `results.json`: every raw answer, check and tool trace
- `provider-usage.jsonl`: every observed message call and tool-choice setting
- `verified-summary.json`: aggregate counts separating named validation tools
  from the absent answer-finish requests; the initial raw summary is retained
- `setup.json` / `scope-audit.json`: tested source hashes and scope checks
- `initial-health.json` / `final-health.json`: role/model and Monitor snapshots

The isolated evaluation service was stopped. The ordinary service, local .env,
EcoMind and SpringMind were not changed. No repeat batch was run.
