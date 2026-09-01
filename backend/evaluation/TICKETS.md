# Local demo tickets

## Scope

Ticket creation and lookup are available only with `NEXO_HARNESS_MODE=enforce`.
Explicit ticket commands in `off` or `observe` return an unavailable response;
other requests keep their existing behavior. The database is initialized only
in enforce mode. Initialization failure stops startup visibly rather than
pretending tickets are available.

They store local demo records in SQLite; they do not connect to an enterprise
service desk, notify IT staff, dispatch a technician or perform device repairs.
A created ticket starts as `pending`. This means the local record exists and
awaits handling, not that an external team has received it.

Ticket snapshots keep user-reported facts, proposed or completed steps and
public demo asset results separate. A model returned by `lookup_asset` is tool
data, not a user statement, live diagnosis or a real warranty confirmation.

The existing `user_id` and `conv_id` delimit this demo's records. They are supplied
by the caller and are not authenticated identities. This scope check is not a
production authorization system. Do not expose the demo as a real multi-user
enterprise ticket service without authenticated identity and access control.

## Explicit actions

An ordinary request to transfer to a human still produces the existing handoff;
it does not create a ticket. Creation requires one of these complete commands:

- `请为当前问题创建工单`
- `帮我创建报修单`
- `帮我创建工单`
- `请创建工单`
- `创建工单`
- `请为当前问题创建报修单`

A final full stop or exclamation mark is accepted. Questions, conditional or
negative statements and a command mixed with other text do not authorize a write.
The current conversation must already have a stable case and a user-reported
problem description. Otherwise, the response asks for the missing problem
information instead of creating an empty record.

Use `查询工单 NEXO-<id>` or `请查询工单 NEXO-<id>` with the returned ticket ID to
read its status. Lookup does not create a ticket, modify its receipt or trigger
human escalation. Results are restricted to the current demo user/conversation.

A minimal flow in the same user/conversation is:

1. Describe the device problem and, if available, provide its asset ID such as `NB-001`.
2. Report only checks actually performed and their results. Requesting human handoff collects this information without creating a ticket.
3. Send `请为当前问题创建工单` as a separate message. Repeat it to retrieve the original ticket instead of creating another.
4. Send `查询工单 <the complete returned ticket ID>` to read the pending record without another escalation.

These actions are handled deterministically and do not use the model to authorize
or execute a ticket write. The `/chat` API retains its existing intent-recognition
call before orchestration; that upstream call can still consume model tokens or
fail. A recognized intent alone never authorizes ticket creation.

## Persistence and retries

SQLite is the source of truth for a created ticket. Its unique case key prevents
repeated creation commands for the same user, conversation and case from creating
multiple tickets. A retry after SQLite succeeds but the Redis case save fails
must return the original ticket. Redis holds the conversation state and receipt;
it is not the authoritative ticket database.

A new issue must start a new case through the existing explicit new-topic flow.
Repeated creation for an existing case returns the original record, rather than
silently replacing its submitted description. A service/storage failure must be
reported as a failure, not as a successful submission or notification.

## Configuration

`NEXO_TICKET_DB_PATH` sets the SQLite file. The sample configuration uses
`./data/tickets/tickets.sqlite3` relative to the local process working directory.
When unset, the backend uses the repository's `data/tickets/tickets.sqlite3`.

Compose explicitly uses `/app/data/tickets/tickets.sqlite3` and mounts the separate
`ticket-data` named volume at `/app/data/tickets`. The production image creates
this directory for the existing non-root `nexo` user. Recreating the backend with
the same volume retains tickets; removing that volume removes its records.
The volume is separate from the existing ChromaDB, Redis and evaluation storage.

Changing the configuration files does not update an already-running service.
Set enforce mode in the local configuration and recreate the backend to enable
tickets. Keep the existing `ticket-data` volume to retain its records.

## Validation

The offline and bounded live results are recorded below. They do not establish production authorization, external notification or a measured success rate.

## Initial isolated-service verification

452 offline tests passed. A single fixed run prepared a fixed device-support conversation using Flash, created one ticket, repeated the creation, queried it, restarted the isolated service and queried the same ticket again. The database contained one row; its complete contents and the two query responses were unchanged across restart. Query responses did not escalate.

All records, including initial offline output, source hashes, database snapshots, model usage and restart timestamps, are retained locally under `data/eval/tickets_v1/first/`. Ticket execution is deterministic; the API still performs model-based intent recognition upstream. During that initial isolated verification, the ordinary service was not rebuilt or replaced.

## Daily-deployment and frontend verification

The ordinary Nexo backend was subsequently rebuilt with Flash and enforce mode,
using the independent `nexo_ticket-data` volume. The existing knowledge base
retained all eight chunks. This verification is separate from the initial
isolated-service run above.

One frontend conversation exercised the complete flow: an `NB-001` device
problem, user-reported power-connection and socket checks with no indicator light,
human handoff, explicit creation, repeated creation, status lookup and another
lookup after restarting the backend. Handoff left the database empty; explicit
creation produced one pending ticket; repeating the command returned the same
ID. Both status lookups returned pending without human escalation.

The complete SQLite row was identical after creation, before restart and after
the post-restart lookup. The two lookup responses were identical. The final case
version was 7 and working memory retained 14 messages. Records are retained under
`data/eval/ticket_deployment_v1/first/`, including database and conversation
snapshots, the build log and restart timestamps. This verifies the bounded
frontend, state and persistence flow; it does not establish general diagnostic
answer quality or external IT delivery.
