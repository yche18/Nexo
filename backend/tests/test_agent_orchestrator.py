import asyncio
import json

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    AccessAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    TechnicalAgent,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "登录时报 401，同时需要申请项目权限",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.TECHNICAL_LOGIN,
        "intent_group": "technical",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"error_code": ["401"], "requested_access": ["只读权限"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TechnicalAgent.profile.role
    assert TechnicalAgent.profile.workflow != AccessAgent.profile.workflow
    assert TechnicalAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "lookup_error_code" in TechnicalAgent.profile.tool_scope
    assert "check_access_fields" in AccessAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    technical_packet = TechnicalAgent(FakeClient(), "test-model")._build_role_packet(req)
    access_packet = AccessAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "diagnostic_fields" in technical_packet
    assert "application_fields" in access_packet
    assert general_packet != technical_packet != access_packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "标记为 IT 人工处理" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.TECHNICAL, "先排查 Token 是否过期。", True),
        AgentResponse(AgentType.ACCESS, "请提供目标系统和申请用途。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("先排查 Token 是否过期。")
    assert "补充说明" in content
    assert "目标系统" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_composite_request_routes_explicit_access_signal_as_supporting_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [object()],
        AgentType.TECHNICAL: [object()],
        AgentType.ACCESS: [object()],
    }

    decision = orchestrator._route_decision(make_request())

    assert decision.primary_agent is AgentType.TECHNICAL
    assert decision.supporting_agents == [AgentType.ACCESS]
    assert decision.multi_agent is True


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    technical_tools = set(TechnicalAgent(FakeClient(), "test-model").get_tools())
    access_tools = set(AccessAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields"}
    assert technical_tools == {"lookup_error_code", "build_diagnostic_plan"}
    assert access_tools == {"check_access_fields"}
    assert escalation_tools == {"create_handoff_summary"}
    assert not general_tools & technical_tools
    assert not technical_tools & access_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "权限申请规范", "content": "申请需说明目标系统和用途"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    access = AccessAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, technical, access, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools


def test_tool_input_validation_rejects_unknown_fields():
    agent = TechnicalAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["lookup_error_code"]

    try:
        agent._validate_tool_input(spec, {"error_code": "401", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "lookup_error_code"
        input = {"error_code": "401"}

    class TextBlock:
        type = "text"
        text = "已根据 401 错误码给出排查建议。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TechnicalAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["lookup_error_code"]
    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool_name"] == "lookup_error_code"
    assert response.tool_traces[0]["input"] == {"error_code": "401"}
    assert response.tool_traces[0]["success"] is True
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "lookup_error_code",
        "build_diagnostic_plan",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_concurrent_requests_keep_tool_traces_isolated():
    class ToolUseBlock:
        type = "tool_use"

        def __init__(self, code):
            self.id = f"toolu_{code}"
            self.name = "lookup_error_code"
            self.input = {"error_code": code}

    class TextBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class ConcurrentClient:
        class Messages:
            async def create(inner, **kwargs):
                messages = kwargs["messages"]
                last_content = messages[-1]["content"]
                if isinstance(last_content, list):
                    code = "500" if "500" in str(last_content) else "401"
                    await asyncio.sleep(0)
                    return type("Response", (), {"content": [TextBlock(f"{code} 排查完成")]})()
                code = "500" if "500" in str(last_content) else "401"
                await asyncio.sleep(0.01 if code == "401" else 0)
                return type("Response", (), {"content": [ToolUseBlock(code)]})()

        def __init__(self):
            self.messages = self.Messages()

    async def run_both():
        agent = TechnicalAgent(ConcurrentClient(), "test-model")
        return await asyncio.gather(
            agent.handle(make_request(message="登录报 401", entities={"error_code": ["401"]})),
            agent.handle(make_request(message="页面报 500", entities={"error_code": ["500"]})),
        )

    first, second = asyncio.run(run_both())
    assert first.tool_traces[0]["input"] == {"error_code": "401"}
    assert second.tool_traces[0]["input"] == {"error_code": "500"}


def test_monitor_penalty_falls_back_to_general_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    general = GeneralAgent(FakeClient(), "test-model")
    technical = TechnicalAgent(FakeClient(), "test-model")
    access = AccessAgent(FakeClient(), "test-model")
    technical.stats.monitor_penalty = 0.5
    orchestrator._pool = {
        AgentType.GENERAL: [general],
        AgentType.TECHNICAL: [technical],
        AgentType.ACCESS: [access],
    }

    decision = orchestrator._route_decision(make_request(
        message="登录报 401",
        entities={"error_code": ["401"]},
    ))

    assert decision.primary_agent is AgentType.GENERAL
    assert decision.supporting_agents == []
    assert "monitor_fallback=[technical]" in decision.reason


def test_descriptive_entities_do_not_add_domain_scores_or_change_routing():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {kind: [object()] for kind in AgentType}
    for intent, message in (
        (IntentCategory.TECHNICAL_LOGIN, "VPN 登录报错 401"),
        (IntentCategory.PERMISSION_REQUEST, "申请项目权限"),
        (IntentCategory.GREETING, "你好"),
        (IntentCategory.HUMAN_HANDOFF, "转人工"),
    ):
        scored = {"error_code": ["401"], "requested_access": ["只读权限"], "target_system": ["VPN"]}
        original = make_request(intent=intent, message=message, entities=scored)
        adapted = make_request(intent=intent, message=message, entities={
            **scored, "issue_description": ["登录失败"], "date": ["今天"], "purpose": ["查看需求"],
        })
        assert orchestrator._domain_scores(original) == orchestrator._domain_scores(adapted)
        assert orchestrator._route_decision(original) == orchestrator._route_decision(adapted)


def test_three_entity_bonuses_match_original_weights_and_only_apply_once():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    baseline = orchestrator._domain_scores(make_request(entities={}))
    for field, target, weight in (
        ("error_code", AgentType.TECHNICAL, 0.2),
        ("requested_access", AgentType.ACCESS, 0.15),
        ("target_system", AgentType.GENERAL, 0.1),
    ):
        assert orchestrator._domain_scores(make_request(entities={field: []})) == baseline
        for values in (["value"], ["first", "second"]):
            actual = orchestrator._domain_scores(make_request(entities={field: values}))
            expected = dict(baseline)
            expected[target] = round(expected[target] + weight, 3)
            assert actual == expected
    all_present = orchestrator._domain_scores(make_request(entities={
        "error_code": ["401"], "requested_access": ["只读权限"], "target_system": ["VPN"],
    }))
    assert all_present == {
        AgentType.GENERAL: round(baseline[AgentType.GENERAL] + 0.1, 3),
        AgentType.TECHNICAL: round(baseline[AgentType.TECHNICAL] + 0.2, 3),
        AgentType.ACCESS: round(baseline[AgentType.ACCESS] + 0.15, 3),
    }


def test_routes_with_extracted_it_entities_and_fixed_intents():
    from core.intent_recognizer import IntentRecognizer

    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {kind: [object()] for kind in AgentType}
    cases = (
        ("VPN 登录失败，错误码 401", IntentCategory.TECHNICAL_LOGIN, UrgencyLevel.LOW, AgentType.TECHNICAL, []),
        ("申请项目系统只读权限，用于查看需求", IntentCategory.PERMISSION_REQUEST, UrgencyLevel.LOW, AgentType.ACCESS, []),
        ("VPN 登录失败，同时申请只读权限", IntentCategory.TECHNICAL_LOGIN, UrgencyLevel.LOW, AgentType.TECHNICAL, [AgentType.ACCESS]),
        ("VPN 怎么使用", IntentCategory.QUERY, UrgencyLevel.LOW, AgentType.GENERAL, []),
        ("账号登录失败", IntentCategory.TECHNICAL_LOGIN, UrgencyLevel.LOW, AgentType.TECHNICAL, []),
        ("VPN 登录失败，请转人工", IntentCategory.HUMAN_HANDOFF, UrgencyLevel.LOW, AgentType.ESCALATION, []),
        ("紧急 VPN 登录失败", IntentCategory.TECHNICAL_LOGIN, UrgencyLevel.CRITICAL, AgentType.ESCALATION, []),
    )
    for message, intent, urgency, primary, supporting in cases:
        req = make_request(message=message, intent=intent, urgency=urgency,
                           entities=recognizer._extract_entities(message))
        decision = orchestrator._route_decision(req)
        assert decision.primary_agent is primary, message
        assert decision.supporting_agents == supporting, message


def test_handoff_displays_known_it_fields_without_empty_json():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request(entities={
        "target_system": ["VPN"], "issue_description": ["登录失败"],
        "date": [], "error_code": ["401"], "requested_access": [], "purpose": [],
    })))
    assert "目标系统：VPN" in response.content
    assert "问题现象：登录失败" in response.content
    assert "错误码：401" in response.content
    assert "发生时间：" not in response.content
    assert "[]" not in response.content
    assert "尚未创建工单或通知 IT 人员" in response.content
    assert client.calls == []


def test_handoff_with_no_extracted_information_asks_for_details():
    client = FakeClient()
    response = asyncio.run(EscalationAgent(client, "test-model").handle(
        make_request(entities={"target_system": [], "error_code": []})
    ))
    assert "暂未提取到具体信息" in response.content
    assert "[]" not in response.content
    assert response.escalate is True
    assert client.calls == []


def test_account_words_do_not_trigger_extra_access_collaboration():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {kind: [object()] for kind in AgentType}
    for message in ("账号登录失败", "账户登录失败", "account 登录失败", "账号账户 account 登录失败"):
        req = make_request(message=message, entities={})
        decision = orchestrator._route_decision(req)
        assert decision.primary_agent is AgentType.TECHNICAL
        assert decision.supporting_agents == []
        assert orchestrator._domain_scores(req)[AgentType.ACCESS] == 0
    for intent in (IntentCategory.ACCOUNT, IntentCategory.ACCOUNT_SECURITY):
        decision = orchestrator._route_decision(make_request(
            message="账号信息", intent=intent, entities={},
        ))
        assert decision.primary_agent is AgentType.ACCESS


def test_access_packet_lists_missing_fields_from_entities():
    agent = AccessAgent(FakeClient(), "test-model")
    for entities, expected in (
        ({}, ["目标系统", "申请用途"]),
        ({"target_system": ["VPN"]}, ["申请用途"]),
        ({"target_system": ["VPN"], "purpose": ["远程办公"]}, []),
    ):
        packet = json.loads(agent._build_role_packet(make_request(entities=entities)))
        assert packet["application_fields"]["missing_fields"] == expected
        assert packet["application_fields"]["target_system"] == entities.get("target_system", [])
        assert packet["application_fields"]["purpose"] == entities.get("purpose", [])


def test_it_handoff_wording_matches_existing_escalation_detection():
    agent = GeneralAgent(FakeClient(), "test-model")
    assert agent._needs_escalation("请转人工（IT 支持）确认。") is True
    assert agent._needs_escalation("请先检查网络连接。") is False


HANDOFF_COLLECTED_INFORMATION = """好的，信息已补充完整。我帮您把交接说明整理成最终版本。

## 一、账号恢复（忘记密码，手机号已停用）

- 问题：忘记密码，自助重置时绑定手机号已停用
- 处理方式：需 **人工（IT 支持）** 核验身份后重置，我无法代您处理，也不会索要密码或验证码。

## 二、项目系统权限申请（信息已齐全）

| 项目 | 内容 |
|------|------|
| 目标系统 | 项目系统（SAP 开发） |
| 申请用途 | 查看需求 |
| 期望权限 | 读写 + 编辑 |
| 权限范围 | 全部项目 |
| 部门 | dev 部门 |

## 下一步：转人工（IT 支持）

请携带以下完整信息联系 **IT 支持**，一并处理两件事：

1. **账号恢复**：忘记密码，绑定手机号已停用，需人工核验身份后重置。
2. **权限申请**：
   - 目标系统 = 项目系统（SAP 开发）
   - 申请用途 = 查看需求
   - 期望权限 = 读写 + 编辑
   - 权限范围 = 全部项目
   - 部门 = dev 部门

---

**需要提醒您：**

- “读写 + 编辑”属于可修改数据的权限，且申请范围为**全部项目**，审批会比只读、限定项目更严格，材料齐全不等于审批通过。
- 我无法代您提交申请、查询审批状态或执行授权，实际开通需由 IT 支持按审批流程处理。
- 登录报 401 属于认证问题，建议在联系 IT 时一并说明，由 IT 排查认证与授权两个环节。

您的交接信息已整理完整，可直接用于联系 IT 支持。"""


def test_handoff_preserves_collected_information_with_empty_current_entities():
    from copy import deepcopy

    client = FakeClient()
    req = make_request(
        message="那你帮我转交给人工吧",
        intent=IntentCategory.HUMAN_HANDOFF,
        entities={key: [] for key in (
            "target_system", "issue_description", "date",
            "error_code", "requested_access", "purpose",
        )},
        history=[{"role": "assistant", "content": HANDOFF_COLLECTED_INFORMATION}],
    )
    before = deepcopy(req)
    result = asyncio.run(EscalationAgent(client, "unused").handle(req))

    for value in ("忘记密码", "绑定手机号已停用", "SAP 开发", "查看需求",
                  "读写 + 编辑", "401"):
        assert value in result.content
    assert "dev 部门" not in result.content
    assert "全部项目" not in result.content
    assert "近期会话" not in result.content
    assert "**助手" not in result.content
    assert "请携带" not in result.content
    assert "审批会比" not in result.content
    assert len(result.content) < 500
    field_lines = [line for line in result.content.splitlines() if line.startswith("- ")]
    assert len(field_lines) == 5
    assert all(line.split("：", 1)[0] in {
        "- 目标系统", "- 问题现象", "- 发生时间", "- 错误码", "- 申请权限", "- 申请用途",
    } for line in field_lines)
    assert "暂未提取到具体信息" not in result.content
    assert result.content.startswith("我已将您的问题标记为 IT 人工处理。")
    assert "尚未创建工单或通知 IT 人员" in result.content
    assert req == before
    assert result.escalate and result.success
    assert result.tools_used == []
    assert client.calls == []


def test_handoff_keeps_session_summary_but_excludes_other_history_and_profile():
    context = (
        "[会话摘要]\n问题现象：忘记密码且手机号停用；目标系统：SAP 开发。\n\n"
        "[相关历史]\n- 另一个会话的订单号 OTHER-123\n\n"
        '[用户画像]\n{"department": "不属于本次申请的部门"}'
    )
    result = asyncio.run(EscalationAgent(FakeClient(), "unused").handle(make_request(
        message="请转人工", intent=IntentCategory.HUMAN_HANDOFF,
        entities={}, history=None, context=context,
    )))
    assert "问题现象：忘记密码且手机号停用" in result.content
    assert "目标系统：SAP 开发" in result.content
    assert "OTHER-123" not in result.content
    assert "不属于本次申请的部门" not in result.content
    assert "暂未提取到具体信息" not in result.content


def test_handoff_displays_latest_confirmed_entity_fields_only():
    req = make_request(
        message="部门更正为 QA，请转人工。", intent=IntentCategory.HUMAN_HANDOFF,
        entities={}, history=[
            {"role": "assistant", "content": "申请权限：读写权限；权限范围：全部项目；部门：dev。"},
            {"role": "user", "content": "范围更正为项目 A，权限只需要只读。"},
        ],
    )
    result = asyncio.run(EscalationAgent(FakeClient(), "unused").handle(req))
    assert "申请权限：只读" in result.content
    for old_or_unrelated in ("读写权限", "全部项目", "dev", "QA", "项目 A", "更正为"):
        assert old_or_unrelated not in result.content
    assert req.entities == {}


def test_handoff_requests_details_only_when_no_session_information_exists():
    req = make_request(
        message="转人工", intent=IntentCategory.HUMAN_HANDOFF, entities={}, history=[],
        context='[相关历史]\n- 其他会话的 SAP 信息\n\n[用户画像]\n{"system": "SAP"}',
    )
    result = asyncio.run(EscalationAgent(FakeClient(), "unused").handle(req))
    assert "暂未提取到具体信息" in result.content
    assert "无需重复提供" not in result.content
    assert "SAP" not in result.content


def test_handoff_uses_current_context_messages_when_history_is_not_provided():
    for prefix in ("", "[会话摘要]\n此前账号无法登录。\n\n",
                   '[相关历史]\n- OTHER-456\n\n[用户画像]\n{"department": "其他部门"}\n\n'):
        context = prefix + "[最近对话]\n\nuser: 已确认：\n目标系统：SAP 开发系统；申请用途：查看需求。"
        result = asyncio.run(EscalationAgent(FakeClient(), "unused").handle(make_request(
            message="请转人工", intent=IntentCategory.HUMAN_HANDOFF,
            entities={}, history=None, context=context,
        )))
        assert "目标系统：SAP 开发系统" in result.content
        assert "申请用途：查看需求" in result.content
        assert "OTHER-456" not in result.content
        assert "其他部门" not in result.content
        assert "暂未提取到具体信息" not in result.content


def test_handoff_displays_only_nonempty_current_entity_fields_without_mutating_them():
    from copy import deepcopy

    entities = {
        "target_system": ["DemoPortal"], "issue_description": ["无法登录"],
        "date": ["今天"], "error_code": ["401"],
        "requested_access": ["只读权限"], "purpose": ["查看需求"],
    }
    req = make_request(message="转人工", intent=IntentCategory.HUMAN_HANDOFF, entities=entities)
    before = deepcopy(req)
    result = asyncio.run(EscalationAgent(FakeClient(), "unused").handle(req))
    assert len([line for line in result.content.splitlines() if line.startswith("- ")]) == 6
    for values in entities.values():
        assert values[0] in result.content
    assert req == before
