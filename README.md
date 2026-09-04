# Nexo

English | [简体中文](README.zh-CN.md)

A multi-agent IT helpdesk assistant for office software, account access, network and device issues. Nexo connects triage, knowledge retrieval, troubleshooting, confirmed ticket creation and status queries.

Built with Python, FastAPI and Vue 3, it combines multi-agent orchestration, hybrid retrieval, an agent harness and layered memory. The workbench exposes intent, routing, tool calls and runtime status.

## Features

- **Intent recognition:** combines model classification, template-vector similarity and pattern rules to produce 19 intent categories, structured entities and four urgency levels.
- **Multi-agent collaboration:** selects a primary role using intent, keywords and entities; compound requests can involve supporting agents and a composed response.
- **Knowledge retrieval:** agents invoke ChromaDB vector retrieval and Elasticsearch BM25 through tool calling, with query rewriting, two-stage RRF, independent reranking, neighboring-chunk expansion and caching.
- **Conversation memory:** Redis manages working memory; ChromaDB stores episodic summaries and user profiles.
- **Demo assets and local tickets:** looks up fictional assets by user-provided IDs. In enforce mode, agents select ticket tools while the backend validates confirmation, prevents duplicate submissions and persists tickets for status queries.
- **Role-scoped Skills:** loads IT support procedures by role and keywords, with runtime reload support.
- **Monitoring and evaluation:** provides agent metrics, tool traces, built-in dialogue evaluation and a separate RAG evaluation suite. Retrieval recall, actual chat evidence and answer quality are measured separately.
- **Agent harness:** manages model/tool execution, independent task state and traces, with optional tool budgets, applicable VPN SOP validation and bounded repair.

## Documentation

- [Architecture](docs/architecture.md): roles, orchestration, the harness, layered memory and local tickets.
- [Hybrid RAG](docs/rag.md): retrieval, configuration, document ingestion and legacy migration.
- [Evaluation](docs/evaluation.md): datasets, metric definitions, commands and recorded results.

## Agent roles

| Role | Responsibility |
| --- | --- |
| GeneralAgent | General IT questions, software guidance and clarification |
| TechnicalAgent | Authentication, software installation/updates and application troubleshooting |
| NetworkAgent | Wi-Fi, DNS, intranet and VPN connectivity |
| DeviceAgent | Initial device triage and fictional asset lookup for computers, phones and printers |
| AccessAgent | Account/access-request guidance and required-field checks |
| EscalationAgent | A concise handoff summary marked as awaiting human handling |

See [Architecture](docs/architecture.md) for role boundaries and execution details.

Access checks combine current inputs with stored user facts, require a target system and purpose, and ask for clarification when values conflict. Nexo does not connect to enterprise identity, approval, asset or ticketing systems, grant access, modify accounts or repair devices. Asset records are fictional, not live diagnostics or warranty confirmation. An authorized ticket submission creates a local demo record with status `pending`; it does not notify IT staff or indicate that a human has accepted the request. The knowledge base contains demonstration IT SOPs.

## Technology stack

| Layer | Technology |
| --- | --- |
| Frontend | Vue 3, Vite, Nginx |
| Backend | Python 3.12, FastAPI |
| Model interface | Anthropic Messages / Tool Use-compatible API |
| Storage and retrieval | Redis, ChromaDB, Elasticsearch (BM25), SQLite (local tickets) |
| Deployment | Docker Compose |

## Quick start

Install Docker and Docker Compose. Create a local configuration on first use:

```powershell
Copy-Item .env.example .env
```

Configure the model in `.env`:

```dotenv
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_BASE_URL=your-compatible-api-base-url
ANTHROPIC_MODEL=your-model-name
```

For the official Anthropic API, `ANTHROPIC_BASE_URL` can be empty. The provider must support Messages and tool use. If `.env` already exists, edit it instead of replacing it.

Configure `RERANK_URL`, `RERANK_API_KEY` and `RERANK_MODEL` separately. An empty endpoint disables independent reranking and retains the RRF order. Elasticsearch starts with Compose. See [Hybrid RAG](docs/rag.md) for configuration and legacy knowledge migration.

Start the services:

```bash
docker compose up -d --build
```

### Access points

| Entry | Address |
| --- | --- |
| Workbench | http://localhost:18080 |
| API documentation | http://localhost:18000/docs |
| Health check | http://localhost:18000/health |

### Ports

| Service | Port |
| --- | ---: |
| Frontend | 18080 |
| Backend API | 18000 |
| Nginx | 18081 |
| Prometheus | 19090 |
| ChromaDB | 18001 |
| Redis | 16379 |
| Elasticsearch (localhost only) | 19200 |

Host ports can be changed with the variables in `.env.example`; they are separate from container ports such as `API_PORT` and `CHROMA_PORT`.

The backend reads `.env` through `env_file`. Container addresses and paths explicitly set in Compose override those variables. After configuration changes, run `docker compose up -d` to recreate affected containers.

Redis, ChromaDB, Elasticsearch and Prometheus use dedicated Nexo volumes. SQLite tickets use the separate `ticket-data` volume; retaining it preserves tickets across backend recreation. Local fallback data, evaluation baselines and application logs are stored under `data/chroma`, `data/eval` and `logs`. Frontend requests pass through a separate Nginx service with a 600-second read timeout. Compose uses subnet `172.19.0.0/16`, which matches the Nginx metrics access restriction.

```bash
# Inspect service status.
docker compose ps

# Read backend logs.
docker compose logs -f backend

# Stop services while retaining data.
docker compose stop
```

## Agent harness

The default `NEXO_HARNESS_MODE=observe` adds execution records without extra harness model calls or answer rewriting. `enforce` enables persistent task progress, tool budgets, applicable VPN SOP validation and local tickets. Validation adds at most one check, one repair and one recheck. `off` disables these features.

Explicit ticket creation or lookup requests in `off` and `observe` return an unavailable response without writing or querying tickets. Recreate the backend container after changing `.env`.

Task state is stored separately from conversation summaries and distinguishes user reports, proposed steps and unverified candidates. Demo asset results retain tool provenance. State carries progress across turns; SQLite separately persists submitted tickets. This is not an external dispatch workflow or arbitrary execution-point recovery.

The workbench trace panel shows state and validation results. See [Architecture](docs/architecture.md) for scope and limitations.

## Usage examples

The current demo, confirmation rules and evaluation fixtures use Chinese. These executable sample messages are retained in their original language:

- `办公软件应该怎么安装？` — Ask how to install office software.
- `今天 VPN 无法登录，错误码 401。` — Report a VPN login failure with error 401.
- `我要申请项目系统只读权限，用于查看需求。` — Request read-only project access to view requirements.
- `VPN 登录失败，同时需要申请项目权限。` — Combine troubleshooting and an access request.
- `转 IT 人工处理。` — Request human handling.

In enforce mode, try the following device-to-ticket flow with the same user and conversation:

1. Describe the issue, for example `我的笔记本资产编号是 NB-001，无法开机，插上电源也没有指示灯` (asset NB-001 will not power on and has no indicator light when connected to power).
2. Report checks actually performed and their results. The assistant proposes a ticket when further handling is needed; routine questions and resolved issues do not require one.
3. After a proposed summary is displayed, reply `需要` or `确认` to confirm it. The agent calls the ticket tool and returns an ID. An explicit `请为当前问题创建工单` also requests creation. Repeated submissions return the original ticket.
4. Send `查询当前工单` or `查询工单 <full returned ID>` to read its `pending` status. Lookup does not trigger another handoff.

An access request can be submitted once the target system and purpose are available; it does not require an invented fault description. An explicit human-handoff request produces a summary marked as awaiting human handling without proactively suggesting a ticket. Existing ticket IDs are included.

Tickets are isolated by demo user and conversation, not authenticated identity. See the [local ticket workflow](docs/architecture.md#local-ticket-workflow) for confirmation, persistence and scope.

The workbench also provides knowledge management, Skills inspection, monitoring and evaluation. A handoff does not automatically create a ticket, and a local ticket does not notify IT staff.

## Project structure

```text
backend/
  core/         Intent recognition, model response parsing and Skills loading
  agents/       Agent roles, orchestration and tools
  mcp/          Retrieval tool management and knowledge storage
  memory/       Conversation memory and user profiles
  harness/      Execution loop, task state and answer validation
  monitor/      Runtime metrics and alerts
  evaluation/   Intent, dialogue and RAG evaluation
  knowledge/    Demo manuals and import data
  api/          FastAPI endpoints
  skills/       IT support procedures
  tests/        Backend tests
frontend/       Vue workbench
docs/           Architecture, retrieval and evaluation guides
config/         Elasticsearch, Prometheus and Nginx configuration
compose.yaml    Deployment configuration
```

## Evaluation snapshot

A single full run recorded on 2026-11-13 used `deepseek-flash`, disabled thinking and an isolated enforce service. The custom dataset contains 80 questions over 40 source manuals and 104 knowledge chunks: 66 answerable questions and 14 unanswerable ones.

| Metric | Result |
| --- | ---: |
| Standalone macro Recall@5 (66 questions) | 93.18% |
| Actual chat evidence macro recall (66 questions) | 55.30% |
| Chats using knowledge retrieval | 43/80 |
| Automated Judge passes / all questions | 79/80 |
| Invalid Judge results | 1/80 |

Automated passes are not human-verified answer accuracy. Review identified an answer that passed despite missing required steps. These results describe a historical run on a fixed custom dataset, not production performance or future versions. See [Evaluation](docs/evaluation.md) for definitions and known issues.

## Development and testing

The backend uses Python 3.12. Create a virtual environment, then install dependencies and run tests from `backend`:

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest tests -q
```

The frontend uses Node.js 22. From `frontend`:

```bash
npm ci
npm run dev
```

The development server proxies API requests to `http://localhost:18000`. Build the frontend with:

```bash
npm run build
```

Some Windows dependencies may require C++ build tools; Docker Compose can run the complete application. Offline tests do not need model credentials. Live conversations and model evaluation require an available model service.

## Deployment scripts

Run these Bash commands from the project root:

- `bash build-image.sh help`: backend image build, tagging and push commands.
- `bash run-image.sh help`: standalone backend container and status commands.
- `bash docker-deploy.sh help`: Compose deployment, status, logs, backup and recovery commands.

Use `docker compose up -d --build` for the complete frontend/backend stack. Generated `backups/` can contain local configuration and are excluded from Git.
