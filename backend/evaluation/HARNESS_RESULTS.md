# Harness v1 verification record

Date: 2026-11-10. Implementation baseline: `74fffc1`.
Final local verification used DeepSeek Flash with thinking disabled. Earlier Pro
results are retained for diagnosis, not treated as a controlled model comparison.

## Offline and build checks

- 228 backend tests passed in the production Python dependency environment.
- Tests include original behavior, mode/request compatibility, tool budgets,
  provenance/negation/correction handling, full General-fallback finalization,
  API user/conversation isolation, Redis version conflicts and real MemoryManager
  compression followed by API continuation (external IO/model responses are fakes).
- A separate real Redis check verified persistence, TTL and competing writes.
- Backend and frontend production images build successfully.
- Source/AST comparison confirms intent/entity recognition, routing/scoring,
  business tools, hybrid RAG and memory algorithms remain unchanged.

## Retained live runs

Each listed Flash batch used one attempt per selected scenario, sequentially.
No failed report was replaced with a better repeat.

| Revision / local report folder | Cases | Answer Judge | State / acceptance findings |
| --- | --- | --- | --- |
| Initial Pro Harness / `harness_v1` | 24 trials | 1 pass, 1 valid failure, 22 invalid | Runtime/state regressions plus provider 429/402; not a quality benchmark |
| Flash v2 / `harness_v1/flash-v2-*` | h01, h02 | 2 passed | Progress-reference rejection and step/result classification defects remained |
| Flash v3 / `harness_v3/vpn-first-followup.json` | h01, h02 | 2 passed | Both configured state gates passed |
| Flash v3 / `harness_v3/core-continuity.json` | h03-h08 | 6 passed | h03 correction state failed; remaining configured gates passed |
| Flash v4 / `harness_v4/correction-handoff.json` | h03, h04 | 2 passed | Corrected system/purpose and handoff gates passed |
| Flash v4 / `harness_v4/vpn-regression.json` | h01, h02 | 1 passed, 1 failed | h02 was classified as escalation and bypassed the Agent progress tool; current feedback was lost |
| Final direct-handoff replay / `harness_v5/direct-handoff-replay.json` | Original failed h02 second-turn input | No model/Judge calls | Current user report preserved, fake assistant-derived purpose absent, input snapshot unchanged |
| Final API smoke / `harness_v5/direct-handoff-live.json` | One explicit human-handoff request | No evaluation Judge | API persisted latest user report and returned complete feedback; no actual external handoff |

Raw JSON is retained locally under `data/eval/` and excluded from Git. The final
replay fixes the deterministic delivery path; it does not retroactively turn the
v4 h02 run into a pass. The original h02 classification/continuation expectations
remain unchanged. The final code has not rerun every live case or the full 60-case
intent suite. No universal accuracy or stability percentage is claimed.

## Manual review matters

- v3 h02 overstated that network/login causes were ruled out after unsuccessful
  attempts. Some service-status/firewall requirements appeared only as possible
  causes instead of explicit checks, yet the verifier accepted the answer.
- v3 h05's first turn suggested letting a colleague try the user's account, which
  risks account sharing. The final-answer Judge evaluated the later printer reply
  and did not catch that earlier advice. This business-response issue remains;
  the narrow VPN verifier is not a comprehensive safety filter.
- Some follow-up questions were not saved to `pending_questions`. Progress remains
  model-selected; absent data is not claimed as a completed workflow.
- v4 h02's Judge incorrectly described internal handoff marking and the absence of
  ticket creation as contradictory. Its separate finding that attempted actions
  were missing was valid.

## Cost and operating defaults

Only targeted Flash runs were added after the provider-balance interruption.
Agent-loop, verifier and evaluation-Judge usage is now recorded where available.
Intent, query rewriting, profile/summary and Composer calls are not all included;
recorded usage is partial, and rounded account balance is not an exact run bill.

Local/default mode stays `observe` to avoid additional Harness model calls during
ordinary use. `enforce` activates state/validation and can add up to three verifier
calls on applicable replies. The isolated evaluation service is stopped after
verification. Enable enforce deliberately for the demonstrated behavior; retain
manual checks and the documented scope limits.
