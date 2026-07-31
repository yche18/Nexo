import pytest

from agents.agent_orchestrator import BaseAgent, Request
from agents.tools import access_tools, check_access_fields


def test_missing_information_is_reported_without_granting_access():
    req = Request(message="我要申请权限", user_id="demo", conv_id="test")
    result = check_access_fields(req, {"target_system": "项目系统", "purpose": "  "})
    assert result["missing_fields"] == ["purpose"]
    assert result["can_grant_access"] is False


def test_complete_information_still_requires_human_approval():
    req = Request(message="申请项目系统用于查看需求", user_id="demo", conv_id="test")
    result = check_access_fields(req, {"target_system": "项目系统", "purpose": "查看需求"})
    assert result["missing_fields"] == []
    assert result["can_grant_access"] is False


def test_access_tool_rejects_unexpected_or_non_string_parameters():
    spec = access_tools()["check_access_fields"]
    for args in ({"password": "unexpected"}, {"purpose": 123}):
        with pytest.raises(ValueError):
            BaseAgent._validate_tool_input(spec, args)
