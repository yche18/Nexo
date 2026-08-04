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
    assert "人工升级" in result.content
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
