# RAG 混合检索：运行与验证

本次只调整知识检索链路。Agent 仍自主调用 `search_knowledge_base`；意图识别、路由、实体提取和记忆流程保持原样。

## 检索与配置

- **关键词通道**：Elasticsearch 与 IK 均固定为 **8.15.3**，原始问题检索 BM25 Top 20。镜像构建文件为 `config/elasticsearch/Dockerfile`，使用独立的 ES 数据卷。
- **向量通道**：保留 ChromaDB 及现有默认 Embedding；原始问题和最多 3 条改写分别召回 Top 20，没有更换向量模型。
- **两级 RRF**：先融合各向量查询，保留 20 个候选；再与 BM25 等权融合，保留 20 个候选。按 `chunk_id` 去重，RRF 常数为 60，避免多条向量查询直接拥有多倍通道权重。
- **独立重排**：通过 HTTP 重排接口处理候选，默认模型名为 `BAAI/bge-reranker-v2-m3`。这只是请求配置，并不会自动下载或部署模型；服务必须支持对应模型及 `results/index/relevance_score` 返回格式。
- **证据整理**：先选择核心片段，再对最多 3 个核心取前后各 1 个邻居；限定同一文档、版本和章节，去除重复片段。保留完整片段及其 ID，默认预算为证据列表序列化后的 **8000 个 JSON 字符**，不是模型 Token 数，也不是整个请求的上下文上限。

`.env` 中的相关参数：

| 参数 | 默认值 / 用途 |
| --- | --- |
| `ELASTICSEARCH_URL` | `http://elasticsearch:9200`，Compose 内部地址 |
| `ELASTICSEARCH_INDEX` | `nexo_knowledge` |
| `RAG_CANDIDATE_K` / `RAG_RRF_K` | `20` / `60` |
| `RAG_NEIGHBOR_WINDOW` / `RAG_EXPANSION_SEEDS` | `1` / `3` |
| `RAG_EVIDENCE_MAX_CHARS` | `8000` |
| `RERANK_URL` | 完整重排端点；留空则禁用重排 |
| `RERANK_API_KEY` | 按服务要求填写，不写入报告 |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `RERANK_TIMEOUT_SECONDS` | `8` |

重排超时、请求失败或返回格式错误时保留 RRF 顺序，轨迹标记 `failed`；未配置端点时标记 `disabled`。某个检索通道失败时可使用另一个通道，错误说明不会被当成知识证据。查看 `/trace/tool/{request_id}` 中的 `tool_calls[].retrieval`，确认通道状态、重排状态和实际证据，而不是仅凭接口响应成功判断整条链路正常。

## 新库与旧库升级

以下命令在 Nexo 项目根目录运行。

**新库**：配置 `.env` 后运行 `docker compose up -d --build`。后端会调用 `initialize_keyword_index()`，为新建或空的关键词索引同步已有 `ready` 知识片段。ES 尚未就绪时会记录降级日志；服务就绪后重启后端，再检查实际工具轨迹。

**旧库**：旧切片缺少版本元数据时仍可只读向量检索，但必须显式迁移后才能继续导入和完整启用混合检索。迁移前暂停知识库写入，并备份目标 Chroma `knowledge_base` 集合的 ID、正文、元数据和向量；确认连接的是预期实例。备份应保存在独立位置，不覆盖原库。

```sh
docker compose build backend
docker compose up -d --build elasticsearch
docker compose stop backend
docker compose run --rm --no-deps backend python -m mcp.rebuild_knowledge_index --migrate-legacy
docker compose up -d backend
```

迁移保留旧切片 ID、正文及切片边界，补齐元数据并重建 ES 索引。脚本只处理知识集合，不操作情景记忆或用户画像集合。默认不允许远端 Chroma 连接失败时静默迁移本地副本。已经迁移过的库，单独重建关键词索引可省略 `--migrate-legacy`；外部重建或迁移完成后必须重启后端，刷新进程中的元数据与检索缓存。

导入经现有知识库 API 完成；两个存储全部完成前不会报告完整成功。失败可能已有部分文档发布，可以使用相同输入重试；这不等于跨数据库事务。不要通过删除整个 Chroma 数据卷来重建知识索引。

## 运行原用例与新增用例

默认评测集现为 `rag_support_v2.json`：保留原 24 个问题及知识片段，只澄清权限材料题与排查记录题的判定要求。旧 `rag_support_v1.json` 保留，可通过 `--dataset` 显式选择；比较不同版本的成绩时应区分标注变化。

RAG Judge 使用现有 Anthropic SDK，通过指定的 `submit_evaluation` 工具提交结构化判定，并在本地校验字段。缺字段、类型错误或截断保留为无效结果，不补默认分；报告记录评分协议及其哈希，旧协议结果不能直接用新协议补评。结构完整不代表语义判分正确，仍需人工复核。

Technical Skill 优先使用适用的知识库 SOP，保留必要检查项与顺序；通用诊断工具只在没有适用 SOP 时兜底。工具的确定性函数实现未变。此约束依赖模型遵循，不保证每次回答都完整准确。

原集 `rag_support_v1.json` 保留 **24 题**；新增 `rag_support_hybrid_v1.json` 为 **8 题**，包括精确错误码、多资料组合和公司特定事实缺失。两者使用同一份冻结的 7 块语料。脚本会先核验实时语料与冻结语料是否一致，不一致就停止；应在准备好对应语料的测试环境执行，不要为通过评测覆盖正常业务库。

```sh
docker compose exec backend python -m evaluation.run_rag_eval --validate-only
docker compose exec backend python -m evaluation.run_rag_eval --dataset /app/evaluation/datasets/rag_support_hybrid_v1.json --validate-only

docker compose exec backend python -m evaluation.run_rag_eval --report-name final-original-run1.json --application-revision feat/rag-hybrid-retrieval
docker compose exec backend python -m evaluation.run_rag_eval --dataset /app/evaluation/datasets/rag_support_hybrid_v1.json --report-name final-extra-run1.json --application-revision feat/rag-hybrid-retrieval
```

默认报告保存在宿主机 `data/eval/rag_hybrid/`。比较升级前后时，在各自实际运行的版本上用同一脚本、数据和模型配置执行，分别命名 `baseline-*` 与 `final-*`；应用版本最好填写实际提交号或明确的未提交工作区标识。重复运行使用 `run2` 等新名字，保留所有轮次，脚本拒绝覆盖已有报告。

默认评测真实 `/chat` 回答，并通过该请求的工具轨迹读取实际送给 Agent 的证据；不另外搜索来替代聊天证据。需要单独观察 `/search` 时可添加 `--standalone-retrieval`，其 Recall@K 与聊天证据覆盖率分别记录。

回答通过要求预先标注的必要要点全部满足、无实质错误且没有禁止宣称。无效 Judge 单独统计；报告同时保留全处理样本通过率和有效判定样本通过率、原始回答、证据、降级状态及耗时。旧应用没有证据轨迹时标记不可观测，不算成零召回。

这是一份小规模合成演示集，不是生产基准；7 块语料且每篇只有 1 块，不能用其结果证明长文扩窗收益或大库检索效果。Judge 使用应用模型家族，也不能代替独立人工标注。实际效果以保存的完整报告为准，本文不预填准确率或提升数字。
