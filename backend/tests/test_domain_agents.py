import asyncio
import json
from pathlib import Path

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, DeviceAgent, EscalationAgent,
    NetworkAgent, Request, TechnicalAgent, build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.skill_loader import SkillManager


def request(message, intent, **kwargs):
    values = dict(message=message, user_id="domain-test", conv_id="domain-test",
                  intent=intent, intent_group="", intent_confidence=0.9,
                  urgency=UrgencyLevel.MEDIUM, entities={})
    values.update(kwargs)
    return Request(**values)


def orchestrator():
    result = AgentOrchestrator.__new__(AgentOrchestrator)
    result._pool = {kind: [object()] for kind in AgentType}
    return result


@pytest.mark.parametrize("message,intent,primary", [
    ("VPN连接超时报错", IntentCategory.NETWORK_ISSUE, AgentType.NETWORK),
    ("电脑连不上公司Wi-Fi，Wi-Fi连不上", IntentCategory.NETWORK_ISSUE, AgentType.NETWORK),
    ("内网无法访问，DNS解析失败", IntentCategory.NETWORK_ISSUE, AgentType.NETWORK),
    ("电脑无法开机", IntentCategory.DEVICE_FAILURE, AgentType.DEVICE),
    ("打印机卡纸并显示报错", IntentCategory.DEVICE_FAILURE, AgentType.DEVICE),
    ("笔记本上的应用崩溃", IntentCategory.TECHNICAL_CRASH, AgentType.TECHNICAL),
    ("安装办公软件", IntentCategory.SOFTWARE_INSTALL, AgentType.TECHNICAL),
    ("如何开通VPN权限", IntentCategory.PERMISSION_REQUEST, AgentType.ACCESS),
    ("网络使用规范", IntentCategory.NETWORK, AgentType.NETWORK),
    ("办公设备保养规范", IntentCategory.DEVICE, AgentType.DEVICE),
])
def test_domains_do_not_spawn_unrelated_support(message, intent, primary):
    decision = orchestrator()._route_decision(request(message, intent))
    assert decision.primary_agent == primary
    assert decision.supporting_agents == []


@pytest.mark.parametrize("message,intent,primary,support", [
    ("VPN连不上，而且办公软件崩溃", IntentCategory.NETWORK_ISSUE,
     AgentType.NETWORK, AgentType.TECHNICAL),
    ("VPN连接超时，同时申请项目权限", IntentCategory.NETWORK_ISSUE,
     AgentType.NETWORK, AgentType.ACCESS),
    ("打印机卡纸，另外VPN连接失败", IntentCategory.DEVICE_FAILURE,
     AgentType.DEVICE, AgentType.NETWORK),
])
def test_independent_problems_keep_collaboration(message, intent, primary, support):
    decision = orchestrator()._route_decision(request(message, intent))
    assert decision.primary_agent == primary
    assert decision.supporting_agents == [support]


@pytest.mark.parametrize("intent,urgency", [
    (IntentCategory.HUMAN_HANDOFF, UrgencyLevel.MEDIUM),
    (IntentCategory.NETWORK_ISSUE, UrgencyLevel.CRITICAL),
    (IntentCategory.DEVICE_FAILURE, UrgencyLevel.CRITICAL),
])
def test_handoff_still_precedes_domain_scoring(intent, urgency):
    decision = orchestrator()._route_decision(request(
        "VPN连接超时，电脑无法开机，请转人工", intent, urgency=urgency))
    assert decision.primary_agent == AgentType.ESCALATION
    assert decision.supporting_agents == []


def test_environment_entities_do_not_add_domain_scores():
    router = orchestrator()
    base = request("请帮我看看", IntentCategory.QUERY)
    expected = router._domain_scores(base)
    base.entities = {
        "operating_system": ["Windows 11"], "client_version": ["5.2"],
        "network_type": ["公司Wi-Fi"], "impact_scope": ["多人"],
        "device_type": ["笔记本"], "device_model": ["T14"],
    }
    assert router._domain_scores(base) == expected


def test_device_tools_exclude_software_network_diagnostics():
    class RagManager:
        pass
    shared = build_shared_rag_tools(RagManager())
    network = NetworkAgent(None, "offline")
    device = DeviceAgent(None, "offline")
    for agent in (network, device):
        agent.set_shared_tools(shared)
    assert set(network.get_tools()) == {"search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"}
    assert set(device.get_tools()) == {"search_knowledge_base"}
    assert network.profile.tool_scope == TechnicalAgent.profile.tool_scope


@pytest.mark.parametrize("kind,agent_class,intent", [
    (AgentType.NETWORK, NetworkAgent, IntentCategory.NETWORK_ISSUE),
    (AgentType.DEVICE, DeviceAgent, IntentCategory.DEVICE_FAILURE),
])
def test_new_agents_respect_monitor_fallback(kind, agent_class, intent):
    from agents.agent_orchestrator import GeneralAgent
    agent = agent_class(None, "offline")
    agent.stats.monitor_penalty = 0.5
    router = orchestrator()
    router._pool = {AgentType.GENERAL: [GeneralAgent(None, "offline")], kind: [agent]}
    decision = router._route_decision(request("请继续处理", intent))
    assert decision.primary_agent == AgentType.GENERAL
    assert "monitor_fallback" in decision.reason


def test_new_fields_survive_confirmed_handoff_without_llm():
    req = request("请转人工", IntentCategory.HUMAN_HANDOFF, harness_mode="enforce")
    req.entities = {"operating_system": ["Windows 11"], "client_version": ["5.2"],
                    "network_type": ["公司Wi-Fi"], "impact_scope": ["同部门多人"],
                    "device_type": ["笔记本"], "device_model": ["T14"]}
    response = asyncio.run(EscalationAgent(None, "offline").handle(req))
    for label in ("操作系统", "客户端版本", "网络环境", "影响范围", "设备类型", "设备型号"):
        assert label in response.content
    assert "Windows 11" in response.content
    assert "T14" in response.content


def test_network_device_skills_are_isolated_and_keep_sop_priority():
    manager = SkillManager(str(Path(__file__).resolve().parents[1] / "skills"))
    manager.load()
    assert manager.errors == []
    network = manager.prompt_for("VPN连接超时", "network")
    device = manager.prompt_for("打印机卡纸", "device")
    assert "网络支持处理规范" in network
    assert "设备支持处理规范" not in network
    assert "不再调用 build_diagnostic_plan" in network
    assert "设备支持处理规范" in device
    assert "网络支持处理规范" not in device
    assert "不套用软件网络通用诊断工具" in device
    for skill in manager.skills:
        if skill.agents in (["network"], ["device"]):
            assert len(skill.content) <= 3200
            for section in ("角色定位", "必须优先收集的信息", "标准排查流程", "常见场景处理要求", "回复格式要求", "禁止事项", "示例表达"):
                assert section in skill.content


@pytest.mark.parametrize("message,intent,primary", [
    ("电脑无法开机，另外企业邮箱登录失败401", IntentCategory.DEVICE_FAILURE, AgentType.DEVICE),
    ("VPN登录失败，验证码收不到，而且公司Wi-Fi连不上", IntentCategory.NETWORK_ISSUE, AgentType.NETWORK),
])
def test_independent_authentication_issue_is_not_suppressed(message, intent, primary):
    decision = orchestrator()._route_decision(request(message, intent))
    assert decision.primary_agent == primary
    assert decision.supporting_agents == [AgentType.TECHNICAL]


@pytest.mark.parametrize("message", [
    "不是打印机卡纸，是电脑软件崩溃",
    "不是VPN连接超时，是电脑软件崩溃",
    "如果打印机卡纸怎么办，我现在是办公软件崩溃",
])
def test_denied_or_hypothetical_domain_does_not_spawn_support(message):
    req = request(message, IntentCategory.TECHNICAL_CRASH)
    router = orchestrator()
    decision = router._route_decision(req)
    assert decision.primary_agent == AgentType.TECHNICAL
    assert decision.supporting_agents == []
    scores = router._domain_scores(req)
    assert scores[AgentType.NETWORK] == scores[AgentType.DEVICE] == 0


@pytest.mark.parametrize("mode", ["observe", "enforce"])
def test_handoff_recovers_user_domain_details_from_history(mode):
    req = request("请转人工", IntentCategory.HUMAN_HANDOFF, harness_mode=mode,
                  history=[{"role": "user", "content": "打印机卡纸，设备型号：HP M404"},
                           {"role": "user", "content": "我用Windows 11，连接公司Wi-Fi"},
                           {"role": "assistant", "content": "设备型号：Wrong model；客户端版本：9.9"}])
    response = asyncio.run(EscalationAgent(None, "offline").handle(req))
    for value in ("打印机卡纸", "HP M404", "Windows 11", "公司Wi-Fi"):
        assert value in response.content
    assert "Wrong model" not in response.content
    assert "9.9" not in response.content
    assert "我用" not in response.content
    assert req.entities == {}


def test_handoff_new_fields_keep_candidate_and_conflict_protection():
    req = request("请转人工", IntentCategory.HUMAN_HANDOFF, harness_mode="enforce",
                  entities={"device_model": ["HP M404"], "network_type": ["公司Wi-Fi"]},
                  history=[{"role": "user", "content": "设备型号：HP M404，连接公司Wi-Fi"}],
                  case_state={"candidates": [{"field": "device_model", "value": "HP M404"}],
                              "conflicts": [{"field": "network_type", "value": "公司Wi-Fi"}]})
    response = asyncio.run(EscalationAgent(None, "offline").handle(req))
    assert "设备型号：" not in response.content
    assert "网络环境：" not in response.content


def test_handoff_new_topic_does_not_restore_previous_device_details():
    req = request("请转人工", IntentCategory.HUMAN_HANDOFF, harness_mode="enforce",
                  history=[{"role": "user", "content": "打印机卡纸，设备型号：HP M404"}],
                  case_state={"new_topic": True})
    response = asyncio.run(EscalationAgent(None, "offline").handle(req))
    assert "HP M404" not in response.content
    assert "打印机卡纸" not in response.content


@pytest.mark.parametrize("mode", ["observe", "enforce"])
@pytest.mark.parametrize("message,expected", [
    ("忘记密码，打印机卡纸", ["忘记密码", "打印机卡纸"]),
    ("VPN连接失败，打印机卡纸", ["连接失败", "打印机卡纸"]),
])
def test_handoff_keeps_mixed_short_issue_points_without_progress(mode, message, expected):
    req = request("请转人工", IntentCategory.HUMAN_HANDOFF, harness_mode=mode,
                  history=[{"role": "user", "content": message},
                           {"role": "user", "content": "打印机卡纸"}])
    response = asyncio.run(EscalationAgent(None, "offline").handle(req))
    line = next(line for line in response.content.splitlines() if line.startswith("- 问题现象："))
    for issue in expected:
        assert issue in line
    assert line.count("打印机卡纸") == 1
    assert "请转人工" not in response.content
    assert req.entities == {}
