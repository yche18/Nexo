import pytest

from agents.agent_orchestrator import BaseAgent, Request
from agents.tools import access_tools, check_access_fields


def test_missing_information_is_reported_without_granting_access():
    req = Request(message="我要申请权限", user_id="demo", conv_id="test",
                  entities={"target_system": ["项目系统"], "purpose": []})
    result = check_access_fields(req, {})
    assert result["missing_fields"] == ["purpose"]
    assert result["can_grant_access"] is False


def test_complete_entities_need_no_model_arguments_and_still_require_human_approval():
    req = Request(message="申请项目系统用于查看需求", user_id="demo", conv_id="test",
                  entities={"target_system": ["项目系统"], "purpose": ["查看需求"]})
    result = check_access_fields(req, {})
    assert result["missing_fields"] == []
    assert result["can_grant_access"] is False


def test_missing_entities_are_not_filled_by_model_arguments():
    req = Request(message="我要申请权限", user_id="demo", conv_id="test")
    result = check_access_fields(req, {"target_system": "模型猜测", "purpose": "模型猜测"})
    assert result["missing_fields"] == ["target_system", "purpose"]


def test_access_tool_requires_no_arguments_and_rejects_unexpected_fields():
    spec = access_tools()["check_access_fields"]
    BaseAgent._validate_tool_input(spec, {})
    for args in ({"password": "unexpected"}, {"purpose": "查看需求"}):
        with pytest.raises(ValueError):
            BaseAgent._validate_tool_input(spec, args)
