# Nexo

企业 IT 支持智能体，提供办公软件咨询、技术故障排查、账号与权限申请指引，以及人工交接信息整理。

基于 Python、FastAPI 和 Vue 3，结合多 Agent 编排、知识库检索与会话记忆，支持在工作台中查看意图、路由、工具调用和运行状态。

## 功能

- **意图识别**：融合模型语义判断、向量相似度与关键词规则，识别 IT 咨询、技术故障、账号权限和转人工诉求。
- **多 Agent 协作**：根据意图、关键词和实体选择主处理角色；复合问题可由多个角色协作，并合并回答。
- **知识库检索**：支持文档导入、语义检索、查询改写、重排与缓存。
- **会话记忆**：使用 Redis 管理工作记忆，使用 ChromaDB 存储情景记忆和用户画像。
- **业务 Skills**：按角色和关键词加载 IT 支持规范，支持运行时重新加载。
- **监控与评测**：提供 Agent 运行指标、工具调用记录，以及意图和对话评测入口。

## Agent 角色

| 角色 | 职责 |
| --- | --- |
| GeneralAgent | IT 咨询、软件使用指引与需求澄清 |
| TechnicalAgent | 登录、网络与办公软件故障排查 |
| AccessAgent | 账号与权限申请指引、申请信息完整性检查 |
| EscalationAgent | 整理已知信息，提供人工交接说明 |

权限信息检查只核对目标系统和申请用途是否齐全。项目当前不连接身份、审批或工单系统，不执行授权、账号修改或工单创建。内置知识库为演示用 IT SOP。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | Vue 3、Vite、Nginx |
| 后端 | Python 3.12、FastAPI |
| 模型接口 | Anthropic Messages / Tool Use 兼容接口 |
| 数据存储 | Redis、ChromaDB |
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

启动服务：

```bash
docker compose up -d --build
```

| 服务 | 地址 |
| --- | --- |
| 工作台 | http://localhost:18080 |
| 后端 API | http://localhost:18000 |
| API 文档 | http://localhost:18000/docs |
| 健康检查 | http://localhost:18000/health |

端口可通过 `.env` 中的 `FRONTEND_PORT` 和 `BACKEND_PORT` 调整。会话与知识库数据保存在 Docker 数据卷中。

```bash
# 查看服务状态
docker compose ps

# 查看后端日志
docker compose logs -f backend

# 停止服务，保留数据
docker compose stop
```

## 使用示例

- 办公软件应该怎么安装？
- 今天 VPN 无法登录，错误码 401。
- 我要申请项目系统只读权限，用于查看需求。
- VPN 登录失败，同时需要申请项目权限。
- 转 IT 人工处理。

工作台同时提供知识库维护、Skills 查看、运行监控与评测功能。人工交接整理当前请求中提取到的信息，不自动创建工单。

## 项目结构

```text
backend/
  core/         意图识别、模型响应解析、Skills 加载
  agents/       Agent 角色、路由编排与工具
  mcp/          检索工具管理与知识库
  memory/       会话记忆与用户画像
  monitor/      运行指标与告警
  evaluation/   意图与对话评测
  api/          FastAPI 入口
  skills/       IT 业务规范
  tests/        后端测试
frontend/       Vue 工作台
compose.yaml    部署配置
```

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
