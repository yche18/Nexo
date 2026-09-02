import asyncio
from pathlib import Path

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, Request, TechnicalAgent,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from core.skill_loader import SkillManager


MIXED_QUESTION = "VPN已连接，项目报403且我有批准通知；办公软件又提示未激活，这两件事分别请IT核验什么？"


def request(message, **kwargs):
    values = dict(message=message, user_id="offline", conv_id="offline",
                  intent=IntentCategory.OTHER, intent_confidence=0.434,
                  urgency=UrgencyLevel.MEDIUM, harness_mode="enforce", entities={})
    values.update(kwargs)
    return Request(**values)


def router():
    result = AgentOrchestrator.__new__(AgentOrchestrator)
    result._pool = {kind: [object()] for kind in AgentType}
    return result


@pytest.mark.parametrize("message", [
    "Mac提示应用来源无法验证，能关掉安全校验安装吗？",
    "Mac的新版本应用来源校验不过，我又想回退旧版本但还没备份，这两步分别有什么前置条件？",
    "一台电脑有灯黑屏，另一台能进系统但某应用闪退，两台可以都直接重装吗？",
])
def test_technical_rules_reach_actual_system_prompt(message):
    manager = SkillManager(str(Path(__file__).resolve().parents[1] / "skills"))
    manager.load()
    agent = TechnicalAgent(None, "offline", skill_manager=manager)
    prompt = agent._build_system_prompt(request(message))
    assert "[动态 Skills]" in prompt
    assert "先查询知识库" in prompt
    assert "不要自行建议放行应用、关闭安全校验或移除隔离标记" in prompt
    assert "有灯黑屏不能直接排除软件原因或断定硬件故障" in prompt
    technical = next(skill for skill in manager.skills if skill.agents == ["technical"])
    assert not technical.matches(message, "access")


@pytest.mark.parametrize("message,entities", [
    (MIXED_QUESTION, {"target_system": ["VPN"], "error_code": []}),
    ("项目页面显示错误", {"error_code": ["403"]}),
    ("需要申请项目权限", {}),
    ("VPN连接超时", {}),
    ("电脑无法开机", {}),
])
def test_known_it_clues_continue_without_reclassifying(message, entities):
    req = request(message, entities=entities)
    assert router()._needs_clarification(req) is False
    assert req.intent is IntentCategory.OTHER
    assert req.intent_confidence == 0.434


@pytest.mark.parametrize("message", [
    "帮我处理一下，有问题。", "还是不行，帮我看看。", "我需要帮助", "今天有什么好吃的？",
])
def test_vague_or_non_it_requests_keep_clarification(message):
    assert router()._needs_clarification(request(message)) is True


@pytest.mark.parametrize("confidence,expected", [(0.499, True), (0.5, False), (0.51, False)])
def test_original_clarification_threshold_is_unchanged(confidence, expected):
    assert router()._needs_clarification(request("帮我处理一下", intent_confidence=confidence)) is expected


def test_known_intent_and_short_message_behavior_is_unchanged():
    assert not router()._needs_clarification(request("帮我处理一下", intent=IntentCategory.REQUEST))
    assert not router()._needs_clarification(request("嗯"))


@pytest.mark.parametrize("message,entities,should_execute", [
    (MIXED_QUESTION, {"target_system": ["VPN"], "error_code": []}, True),
    ("帮我处理一下，有问题。", {}, False),
    ("还是不行，帮我看看。", {}, False),
])
def test_run_uses_existing_route_only_when_clues_exist(message, entities, should_execute):
    instance = router()
    req = request(message, entities=entities)
    expected = instance._route_decision(req)
    calls = []

    async def execute(current, agent_type):
        calls.append(agent_type)
        return AgentResponse(agent_type, "已根据现有信息处理两个问题", True)

    async def finalize(current, result):
        return result

    instance._execute = execute
    instance._finalize_result = finalize
    result = asyncio.run(instance.run(req))
    if should_execute:
        assert calls == [expected.primary_agent]
        assert result.response == "已根据现有信息处理两个问题"
        assert result.routing_reason == expected.reason
    else:
        assert calls == []
        assert result.routing_reason == "低置信度 OTHER 意图，先澄清用户需求"