# IT 支持场景评测 v1

本套件增加数据与独立验证入口，不修改应用路由、实体提取、模型提示词或默认评测阈值，也不覆盖前端内置评测的 baseline.json。

## 数据与边界

- 60 条单标签意图输入，覆盖现有 14 个标签；40 条开发集、20 条保留测试集。
- 10 个独立多轮场景，每个 3 轮，共 30 轮；7 个开发场景、3 个保留场景。
- 数据为助手根据已有业务定义编写的合成样例，不是生产流量，也没有独立人工标注验证。
- 每条意图用例具有独立 family；没有把同一原句的近似改写分到开发集与保留集。
- 当前范围：咨询、设备配置、账号管理、权限指引、故障排查、人工交接。真实授权、密码重置执行、工单发送不在能力范围。

## 预先确定的标签规则

- query：询问制度、文档、使用说明；request：明确请求协助配置或安装。
- account：修改普通账号资料；account_security：密码恢复、凭据泄露、异常登录和安全设置。
- permission_request：申请开通或变更访问权限，包括能登录但未获授权。
- technical_login：认证、验证码、登录循环等；technical_crash：明确崩溃/闪退，或错误后退出；technical：网络、设备、性能等其他技术故障。本版本不使用仅报 500 而未描述崩溃的歧义输入来区分后二者。
- complaint：表达不满但未要求负责人介入；escalation：明确要求主管/管理层升级处理；human_handoff：要求真人或人工渠道接手。
- greeting、feedback、other：分别为问候、正面服务反馈、非 IT 请求。
- 单标签精确匹配，不把预测错误事后改成“也算对”；混合诉求放入对话场景，检查参与 Agent。

## 执行

在已安装后端依赖的环境执行（Docker 后端已包含所需依赖）。脚本读取 Nexo .env 或进程环境，不输出密钥。

```shell
python backend/evaluation/run_support_eval.py --split dev --validate-only
python backend/evaluation/run_support_eval.py --split dev --report data/eval/support_v1/dev-run1.json
python backend/evaluation/run_support_eval.py --split holdout --report data/eval/support_v1/holdout-run1.json
```

Docker 中脚本路径为 /app/evaluation/run_support_eval.py，可传 --api-base http://localhost:8000。新增文件首次运行前需复制到容器，或重建后端。报告目录 /app/data/eval 对应宿主机 data/eval。

先跑开发集，检查数据/脚本问题，再跑保留集。本次应用代码冻结，不针对测试结果调路由或提示词。若发现标注问题，应保留旧报告，另建数据版本并说明变更；不能覆写原标签后只报告更好的结果。

## 指标口径

1. 意图：逐条准确率、Macro-F1、各类别支持数及 F1、模型失败降级次数。每次脚本启动使用新的识别器缓存。
2. 多轮：实际走 /chat，使用独立测试用户与会话，保留真实记忆和实体传递路径。不会修改已有用户会话；测试记录遵循应用原有存储与过期策略。
3. Judge：复用现有四维评分，阈值仍为 0.75；解析失败单列，不把默认 0.5 混入有效评分均值。
4. 行为断言：Agent 路由、协作、交接字段、最新更正、缺失字段和明确的未实现操作宣称。文字匹配只是有限检查，不能替代人工语义审阅。
5. 场景通过：所有轮次成功，行为断言全部通过，且每轮 Judge 有效并达标。单独保留行为与 Judge 成绩，不汇成一个混淆分母的“总准确率”。
6. 报告保留模型、SDK、开关、数据哈希、原始回答、耗时和失败项；文件已存在则拒绝覆盖。结果位于 Git 忽略的数据目录。

数字只代表固定小型合成集的一次运行。保留集经过查看后不能继续当作未见数据用于反复调参。简历应注明样本量和指标口径。

## 断言修订记录

断言 1.0 的未实现操作检查把“确认 VPN 权限是否已开通”误判为执行宣称。1.1 改为检查明确的执行表述，并增加询问、条件句和肯定宣称的正反例测试。原始开发集报告保留，使用同一批回答离线重算；未重新生成回答或修改标签。保留集使用 1.1。

```shell
python backend/evaluation/run_support_eval.py --split dev --rescore data/eval/support_v1/20261013-dev-1.json --report data/eval/support_v1/20261013-dev-1-reviewed.json
```
