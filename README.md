# Nexo

企业 IT 支持智能体，Python / FastAPI 后端与 Vue 前端。

## 启动

需要 Docker Compose。首次使用将 `.env.example` 复制为 `.env`，填写可用的模型配置：

```powershell
Copy-Item .env.example .env
# 编辑 .env 中的 ANTHROPIC_API_KEY、ANTHROPIC_BASE_URL、ANTHROPIC_MODEL
# 已配置好的 .env 请勿覆盖。
docker compose up -d --build
```

- 工作台：http://localhost:18080
- API：http://localhost:18000
- Swagger：http://localhost:18000/docs
- 健康检查：http://localhost:18000/health

Nexo 使用独立的 Compose 项目、网络与数据卷，Redis 和 ChromaDB 分别存储会话与知识数据。默认端口可在 `.env` 中修改。查看运行状态使用 `docker compose ps`，查看后端日志使用 `docker compose logs backend`，停止使用 `docker compose stop`。不要随意删除卷，里面保存会话与知识库。

模型配置需要支持Anthropic Messages / tool-use 协议。服务能启动并不代表外部模型一定可用。

## 角色与演示

| 角色 | 职责 |
| --- | --- |
| GeneralAgent | IT 咨询、软件安装指引与需求澄清 |
| TechnicalAgent | 登录、网络与办公软件排查建议 |
| AccessAgent | 账号与权限申请流程、申请字段检查 |
| EscalationAgent | 需要人工处理时提供交接说明 |

可以依次尝试：“办公软件怎么安装？”、“VPN 登录报错 401”、“我要申请项目系统权限”、“VPN 登录报错，而且需要申请项目权限”、“转 IT 人工处理”。

当前不连接真实身份、权限或工单系统，不代为申请或执行授权。默认知识库是标注为演示资料的 IT SOP，不能当成真实公司制度。Skills 是模型提示规则，不是程序级授权系统。

## 目录

```text
backend/
  core/         意图识别、LLM 文本解析、Skills 加载
  agents/       角色、编排与工具
  mcp/          检索工具层和知识库
  memory/       会话记忆
  monitor/      监控
  evaluation/   评测框架和 IT 示例用例
  api/          FastAPI 入口
  skills/       IT 业务规则
  tests/        后端与权限工具测试
frontend/       Vue 工作台
compose.yaml    独立部署入口
```

## 开发与测试

后端离线测试（在 `backend` 目录，开发环境需要 pytest 和 requirements 对应依赖）：

```powershell
python -m pytest tests -q
```

前端使用 Node 22，在 `frontend` 目录执行 `npm ci`、`npm run build`。开发模式 `npm run dev`，Vite 默认代理到 `http://localhost:18000`。Windows 安装 ChromaDB 可能需要 C++ 构建工具，完整运行优先使用Docker 方式。
