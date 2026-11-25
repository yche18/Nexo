# Nexo

[English](README.md) | 简体中文

企业 IT 支持多智能体助手，覆盖办公软件、账号权限、网络与设备故障，串联问题分诊、知识检索、排查、确认建单与状态查询。

基于 Python、FastAPI 和 Vue 3，结合多 Agent 编排、混合检索、Agent Harness 与分层会话记忆，支持在工作台中查看意图、路由、工具调用和运行状态。

## 功能

- **意图识别**：融合模型语义判断、向量相似度与关键词规则，输出 19 类意图、结构化实体和 4 级紧急度。
- **多 Agent 协作**：根据意图、关键词和实体选择主处理角色；复合问题可由多个角色协作，并合并回答。
- **知识库检索**：Agent 通过 Tool Calling 按需调用 ChromaDB 向量检索和 Elasticsearch BM25，结合查询改写、两级 RRF、独立重排、邻接扩窗与缓存。
- **会话记忆**：使用 Redis 管理工作记忆，使用 ChromaDB 存储情景记忆和用户画像。
- **演示资产与本地工单**：按用户提供的编号查询公共演示资产；在 enforce 模式下，由 Agent 选择建单工具，后端校验用户确认并防止重复提交，支持持久化工单状态查询。
- **业务 Skills**：按角色和关键词加载 IT 支持规范，支持运行时重新加载。
- **监控与评测**：提供 Agent 运行指标、工具 Trace、前端内置评测与独立 RAG 评测，区分检索召回、实际聊天证据和回答质量。
- **Agent Harness**：统一模型与工具执行、独立任务状态和执行记录；可选启用工具预算、VPN SOP 回答核验及有限修正。

## 文档导航

- [架构说明](docs/architecture.md)：角色与编排、Harness、分层记忆及本地工单。
- [RAG 说明](docs/rag.md)：混合检索、配置、文档导入和旧库迁移。
- [评测说明](docs/evaluation.md)：评测数据、指标口径、运行方法与已保存结果。

## Agent 角色

| 角色 | 职责 |
| --- | --- |
| GeneralAgent | IT 咨询、软件使用指引与需求澄清 |
| TechnicalAgent | 登录、办公软件安装更新与应用故障排查 |
| NetworkAgent | Wi-Fi、DNS、内网与 VPN 连接故障排查 |
| DeviceAgent | 电脑、手机、打印机等设备故障初诊与公共演示资产资料查询 |
| AccessAgent | 账号与权限申请指引、申请信息完整性检查 |
| EscalationAgent | 整理已知信息，提供人工交接说明（待人工处理） |

角色边界、编排流程、Harness 和工单机制见 [架构说明](docs/architecture.md)。

权限信息检查结合本轮信息与会话中已保存的用户事实，核对目标系统和申请用途是否齐全，冲突信息需澄清。项目不连接真实企业身份、审批、资产或工单系统，不执行授权、账号修改或设备维修。资产查询使用公共虚构资料，不代表实时检测或真实保修状态。明确建单指令会保存 Nexo 本地演示工单，初始状态为 `pending`，不代表已通知 IT 或人工接单。内置知识库为演示用 IT SOP。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Vue 3、Vite、Nginx |
| 后端 | Python 3.12、FastAPI |
| 模型接口 | Anthropic Messages / Tool Use 兼容接口 |
| 数据存储与检索 | Redis、ChromaDB、Elasticsearch（BM25）、SQLite（本地工单） |
| 部署 | Docker Compose |

## 快速启动

准备 Docker 和 Docker Compose。首次运行时创建本地配置：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中填写模型配置：

```dotenv
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_BASE_URL=your-compatible-api-base-url
ANTHROPIC_MODEL=your-model-name
```

使用 Anthropic 官方接口时，`ANTHROPIC_BASE_URL` 可留空。模型服务需要支持 Messages 与工具调用。已有 `.env` 时直接编辑，无需重新复制。

重排服务需另行配置 `RERANK_URL`、`RERANK_API_KEY` 和 `RERANK_MODEL`；端点留空时保留 RRF 排序，不执行独立重排。Elasticsearch 随 Compose 启动。配置及旧知识库迁移见 [RAG 说明](docs/rag.md)。

启动服务：

```bash
docker compose up -d --build
```

### 访问入口

| 入口 | 地址 |
| --- | --- |
| 工作台 | http://localhost:18080 |
| API 文档 | http://localhost:18000/docs |
| 健康检查 | http://localhost:18000/health |

### 主要端口

| 服务 | 端口 |
| --- | ---: |
| 前端工作台 | 18080 |
| 后端 API | 18000 |
| Nginx | 18081 |
| Prometheus | 19090 |
| ChromaDB | 18001 |
| Redis | 16379 |
| Elasticsearch（仅本机） | 19200 |

宿主机端口可通过 `.env.example` 列出的端口变量调整；与 `API_PORT`、`CHROMA_PORT` 等容器内部端口区分。

后端通过 `env_file` 接收 `.env` 中的配置；Compose 中声明的容器地址和路径会覆盖同名配置。修改配置后运行 `docker compose up -d` 重建受影响的容器。

Redis、ChromaDB、Elasticsearch 和 Prometheus 使用 Nexo 独立数据卷；SQLite 工单单独保存在 `ticket-data` 数据卷，保留该卷即可在后端容器重建后继续查询。本地降级数据、评测基线和应用日志分别位于 `data/chroma`、`data/eval`、`logs`。前端请求经过独立 Nginx，读取超时为 600 秒。Compose 使用独立网段 `172.19.0.0/16`，Nginx 的指标访问限制与该网段对应。

```bash
# 查看服务状态
docker compose ps

# 查看后端日志
docker compose logs -f backend

# 停止服务，保留数据
docker compose stop
```

## Agent Harness

默认 `NEXO_HARNESS_MODE=observe`，普通对话只增加执行记录，不额外调用模型或改写回答。
切换为 `enforce` 后启用任务进度保存、工具预算、适用 VPN SOP 的回答核验及本地工单；
核验最多增加一次检查、一次修正和一次复查。`off` 关闭这些处理。
明确的工单创建或查询指令在 `off`、`observe` 模式下返回功能未开启，不写入或查询工单。
修改 `.env` 后重新创建后端容器即可生效。

任务状态与聊天摘要分开保存，区分用户反馈、建议步骤和待核验信息；工具返回的演示资产资料单独记录来源。
会话状态支持轮次之间的处理进度延续，SQLite 单独保存明确提交的本地工单；不提供真实 IT 派单、人工处理流程或任意执行位置的断点恢复。
工作台的执行记录区域可查看状态与核验结果；范围、评测方法及限制见
[架构说明](docs/architecture.md)。

## 使用示例

- 办公软件应该怎么安装？
- 今天 VPN 无法登录，错误码 401。
- 我要申请项目系统只读权限，用于查看需求。
- VPN 登录失败，同时需要申请项目权限。
- 转 IT 人工处理。

在 `enforce` 模式下，可在同一用户、同一会话中尝试设备到工单的流程：

1. 描述问题，例如“我的笔记本资产编号是 NB-001，无法开机，插上电源也没有指示灯”。
2. 根据实际情况反馈已做检查及结果。需要后续处理时，助手会给出建单指引；普通咨询或已解决问题不要求建单。
3. 在助手展示待登记摘要后回复“需要”或“确认”，由 Agent 调用建单工具并返回编号；明确发送 `请为当前问题创建工单` 也可请求建单。重复提交返回原单。
4. 发送 `查询当前工单`，或 `查询工单 <返回的完整编号>`，查看 `pending`（待人工处理）状态。查询不会再次转人工。

权限申请补齐目标系统和用途后也可显式建单，无需描述虚构故障。用户明确说“转人工”时，助手仅整理摘要并停在“待人工处理”，不主动建议建单；已有工单则带上编号。

工单记录按演示用户和会话隔离，这不是身份认证。完整命令、持久化及验证边界见 [架构说明中的本地工单流程](docs/architecture.md#local-ticket-workflow)。

工作台同时提供知识库维护、Skills 查看、运行监控与评测功能。普通人工交接整理已收集的信息，不自动创建工单；本地工单也不会通知 IT 人员。

## 项目结构

```text
backend/
  core/         意图识别、模型响应解析、Skills 加载
  agents/       Agent 角色、路由编排与工具
  mcp/          检索工具管理与知识库
  memory/       会话记忆与用户画像
  harness/      执行循环、任务状态与结果核验
  monitor/      运行指标与告警
  evaluation/   意图、对话与 RAG 评测
  knowledge/    演示知识手册与导入数据
  api/          FastAPI 入口
  skills/       IT 业务规范
  tests/        后端测试
frontend/       Vue 工作台
docs/           架构、检索与评测说明
config/         Elasticsearch、Prometheus 和 Nginx 配置
compose.yaml    部署配置
```

## 评测摘要

2026-11-13 保存的单次全量运行使用 `deepseek-flash`、关闭思考及隔离 enforce 服务。自建数据共 80 题，覆盖 40 份来源手册、104 个知识片段；其中 66 题可回答，14 题无答案。

| 指标 | 结果 |
| --- | ---: |
| 独立检索宏平均 Recall@5（66 题） | 93.18% |
| 实际聊天证据宏平均召回（66 题） | 55.30% |
| 聊天调用知识库 | 43/80 |
| 自动 Judge 通过 / 总题数 | 79/80 |
| Judge 无效结果 | 1/80 |

自动通过数不是人工核验的回答准确率；复核发现一条被判通过的答案仍漏必要步骤。该结果为固定自建集的历史运行记录，不代表生产效果或未来版本表现。完整口径与已知问题见 [评测说明](docs/evaluation.md)。

## 开发与测试

后端使用 Python 3.12。准备虚拟环境，在 `backend` 目录安装依赖后运行测试：

```bash
pip install -r requirements.txt
pip install pytest
python -m pytest tests -q
```

前端使用 Node.js 22，在 `frontend` 目录运行：

```bash
npm ci
npm run dev
```

开发服务器默认将 API 请求代理到 `http://localhost:18000`。构建前端：

```bash
npm run build
```

Windows 本地安装部分依赖可能需要 C++ 构建工具，完整服务可通过 Docker Compose 运行。离线测试不需要模型密钥；实际对话和模型评测需要配置可用的模型服务。

## 部署脚本

在项目根目录使用 Bash 运行：

- `bash build-image.sh help`：后端镜像构建、标签与推送命令。
- `bash run-image.sh help`：单独运行后端镜像及查看容器状态。
- `bash docker-deploy.sh help`：Compose 部署、状态、日志、备份和恢复命令。

完整前后端服务仍使用 `docker compose up -d --build`。脚本生成的 `backups/` 包含本地配置，不纳入 Git。
