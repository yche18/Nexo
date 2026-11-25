# Architecture

Nexo uses FastAPI for conversation, knowledge, tool-trace and evaluation endpoints, with a Vue 3 workbench. Agents use an Anthropic Messages / Tool Use-compatible interface. Orchestration, state management and tool execution are implemented in project-local Python modules.

## Request flow

1. Receive the message and load conversation context. Fuse LLM few-shot classification, template similarity and pattern rules into intent, entities, urgency and confidence.
2. Select a primary agent using intent, domain scores, keywords, entity bonuses and Monitor state; add supporting agents when needed. Explicit human handoff and CRITICAL requests enter the deterministic Escalation node.
3. In enforce mode, load independent task state. Inject Skills by role and keywords. Within the tool-use loop, the agent chooses knowledge, diagnostic, asset or ticket tools.
4. Use a single agent's response directly, or compose results after parallel primary/supporting execution. Low confidence without usable domain signals leads to clarification; usable specialist signals retain specialist handling. Classification and routing run again each turn.
5. Merge state updates, validate responses where applicable, persist task state, conversation memory and traces, then return the answer and any ticket receipt.

Intent routing does not force knowledge retrieval. Agents choose whether to search based on the question, role contract and Skills. See [Hybrid RAG](rag.md) for the retrieval pipeline.

## Roles and domains

| Role | Main responsibility |
| --- | --- |
| GeneralAgent | General IT questions, software guidance and clarification |
| TechnicalAgent | Authentication, software installation/updates, crashes and general troubleshooting |
| AccessAgent | Account/access-request preparation and required-field checks |
| NetworkAgent | Wi-Fi, DNS, intranet and VPN connectivity |
| DeviceAgent | Initial device triage and demo asset lookup |
| EscalationAgent | Deterministic handoff summary and awaiting-human-handling marker; no LLM call |

The recognizer has 19 intent enum values and four urgency levels. Domains can overlap: VPN connection timeouts and VPN access requests belong to Network and Access respectively. Primary/supporting roles handle compound requests; not every request invokes every agent.

Intent-template similarity uses local character n-grams, not a remote semantic embedding model. This is separate from ChromaDB vectorization in RAG. Patterns match business expressions; entities retain structured system names, error codes, access needs and environment details. Not all entities contribute routing bonuses. The three current bonuses are error code to Technical (+0.20), requested access to Access (+0.15), and target system to General (+0.10).

## Agent harness

The harness manages model/tool execution, call budgets, independent task state, answer validation and traces. It distinguishes user facts, proposed steps, completed actions and tool-sourced information.

| Mode | Behavior |
| --- | --- |
| `off` | Disables harness state/events; local tickets are unavailable |
| `observe` (default) | Adds execution and applicability records without extra harness validation calls or answer rewriting |
| `enforce` | Enables task progress, budgets, applicable answer validation and local tickets |

Recreate the backend container after changing `.env`.

| Setting | Default |
| --- | --- |
| `NEXO_HARNESS_MODE` | `observe` |
| `NEXO_HARNESS_MAX_TOOL_CALLS` | `12`, per-agent tool invocation cap |
| `NEXO_HARNESS_VERIFY_VPN` | `true`, applicable VPN SOP verification in enforce mode |
| `NEXO_HARNESS_VALIDATION_TIMEOUT` | `60` seconds per validation or repair request |

Ordinary agents have at most three tool-interaction rounds. An enforce-mode primary agent with ticket tools has at most five. An earlier final answer ends execution immediately. If the last available round still returns tool use, enforce allows at most one additional answer-only call with tools disabled and SDK retries disabled. Ticket finalization uses a separate summary instruction to avoid further tool commands. Leaked DSML markup is rejected, never interpreted as executable tool calls.

Semantic answer verification currently covers applicable VPN-timeout SOPs, not every business domain. It adds at most one check, one repair and one recheck, with automatic SDK retries disabled for those requests. The normal SDK timeout remains 600 seconds. Evaluation Judge calls are separate from runtime verification.

### Task state and provenance

- `record_support_progress` stages user facts, actions, outcomes, pending questions and next steps. Parallel updates are merged before response composition.
- User facts require traceable literal user evidence. Unsupported inferences remain unverified candidates rather than confirmed facts. Asset models returned by tools retain tool provenance.
- Proposed actions, user-completed actions and outcomes are stored separately. A successful tool call does not mean the user performed a troubleshooting step; unconfirmed outcomes remain unknown.
- A case becomes resolved only after explicit user confirmation of recovery. An explicit topic change starts a new case.
- Redis WATCH and version checks protect state updates. Conflicts return 409 rather than overwriting another turn's changes.
- Specialist failures can fall back under the routing rules. Attempts at writes or other side effects restrict automatic fallback; successful ticket receipts survive final-answer failure.

Task state is separate from conversation summaries and does not provide arbitrary execution-point recovery. Some recording constraints depend on the model submitting appropriate fields. Trace warnings expose rejected or unverified updates.

## Layered memory and Skills

Redis stores working memory with a 24-hour TTL. At the 15-message threshold, older content is compressed while the latest five messages are retained. ChromaDB stores historical summaries and user profiles for later recall. Independent task state holds operational progress and cannot be replaced by a general conversation summary.

Skills separate general reception, technical, access, network and device SOPs from tool code. They are injected into the system prompt by agent type and keywords, support runtime inspection/reload, and have a 5000-character injection cap. Applicable knowledge-base SOPs take precedence; generic diagnostic plans are a fallback when no suitable SOP is available.

## Local ticket workflow

Assets and tickets use local demonstration data. Ticket creation and lookup are available only in enforce mode:

1. `lookup_asset` reads fictional asset records by user-provided ID. Results are not live diagnostics or warranty confirmation.
2. After collecting the issue or request, the agent calls `prepare_ticket` to validate material and prepare a summary. Access requests require a target system and purpose; fault tickets require a user-reported issue description.
3. After the summary and confirmation question have been delivered, a supported confirmation authorizes only that unchanged proposal. Changed material, negation or ambiguity requires clarification or another proposal. Explicit creation commands can also authorize submission.
4. The agent chooses `create_ticket`; the backend validates confirmation, conversation scope and idempotency. SQLite stores the ticket with a unique case key. Repeated submissions return the original record.
5. Supported lookup commands follow a deterministic path that reads SQLite without creating another ticket or escalating again.

The current Chinese command forms are retained here as executable inputs, not translated aliases: confirmations include `需要`, `可以` and `确认`; creation can use `请为当前问题创建工单`; lookup can use `查询当前工单` or `查询工单 <full ID>`.

Only the primary agent can prepare and create tickets. SQLite is the source of truth for created tickets; Redis holds conversation state and receipts. If SQLite succeeds but subsequent state persistence fails, a retry should still return the original ticket. Later conversation does not automatically rewrite an existing ticket snapshot; a new issue needs a new case.

The default `pending` status means a local record awaits handling, not that IT staff have received or accepted it. There is no real approval, dispatch or status-transition backend. An explicit request for human support produces a concise summary of known facts and receipts, marked as awaiting human handling, without proactively requiring a ticket.

`NEXO_TICKET_DB_PATH` controls the SQLite location. Compose uses `/app/data/tickets/tickets.sqlite3` on the separate `ticket-data` volume. Retaining that volume preserves tickets across container recreation; deleting it removes them. Demo user and conversation IDs isolate records but do not provide authenticated identity.

## Monitoring and evaluation

Monitor uses completed agent requests to track success rate, latency and consecutive failures, influencing routing scores. Pending and cancelled requests are not counted as completed failures. Tool traces expose arguments, results, retrieval evidence and harness state.

Available provider token usage is recorded in corresponding traces, but not every intent, summary, rewrite or composition call appears in the same aggregate. A single trace is not the account's total bill. See [Evaluation](evaluation.md) for datasets and results.
