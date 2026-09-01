"""Agent 工具定义与实现。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求分析
  - 技术排障建议
  - 权限申请字段检查
  - 人工升级摘要
  - 共享知识库 RAG

身份查询、权限开通、密码重置等真实业务操作不在这里执行。
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


DEMO_ASSETS_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "demo_assets.json"


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


def parse_ticket_action(message: str) -> Optional[Dict[str, str]]:
    """Only complete explicit commands authorize local ticket operations."""
    text = (message or "").strip().rstrip("。！!").strip()
    if text in {
        "请为当前问题创建工单", "请为当前问题创建报修单",
        "帮我创建报修单", "帮我创建工单", "请创建工单", "创建工单",
    }:
        return {"action": "create_ticket"}
    match = re.fullmatch(r"(?:请)?查询工单\s+(NEXO-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)", text, re.I)
    if match:
        return {"action": "get_ticket", "ticket_id": match.group(1).upper()}
    return None


def _ticket_snapshot(case: Dict[str, Any]) -> Dict[str, Any]:
    """Save concise source-separated case information, never dialogue history."""
    from harness.state import FACT_FIELDS
    facts = {}
    for key, item in case.get("facts", {}).items():
        if key in FACT_FIELDS and isinstance(item, dict) and item.get("source") == "user_reported":
            facts[key] = {name: copy.deepcopy(item[name]) for name in
                          ("value", "source", "source_quote", "request_id") if name in item}
    steps = [{key: copy.deepcopy(step[key]) for key in
              ("step_id", "description", "status", "source", "source_quote", "result", "request_id")
              if key in step} for step in case.get("steps", []) if isinstance(step, dict)]
    return {
        "user_facts": facts,
        "asset_lookup": copy.deepcopy(case.get("asset_lookup") or {}),
        "steps": steps,
        "pending_questions": copy.deepcopy(case.get("pending_questions") or []),
        "conflicts": copy.deepcopy(case.get("conflicts") or []),
    }


async def create_ticket(req: Request, store: Any) -> Dict[str, Any]:
    """Backend-only action; not exposed through model tool schemas."""
    from harness.context import harness_mode
    action = parse_ticket_action(req.message)
    if (req.harness_mode or harness_mode()) != "enforce" or not action or action["action"] != "create_ticket":
        return {"success": False, "error": "尚未收到明确的工单创建指令"}
    case = req.case_state or {}
    if (not req.user_id or not req.conv_id or not case.get("case_id")
            or case.get("user_id") != req.user_id or case.get("conv_id") != req.conv_id):
        return {"success": False, "error": "当前问题状态尚未就绪，请先描述问题后再创建工单"}
    issue = case.get("facts", {}).get("issue_description") or {}
    summary = issue.get("value") if issue.get("source") == "user_reported" else None
    if not isinstance(summary, str) or not summary.strip():
        return {"success": False, "error": "当前尚无已记录的问题现象，请先说明具体故障，再明确请求创建工单"}
    if any(item.get("field") == "issue_description" for item in case.get("conflicts", [])):
        return {"success": False, "error": "已记录的问题现象存在冲突，请先确认后再创建工单"}
    if store is None:
        return {"success": False, "error": "本地工单存储尚未配置，未创建工单"}
    try:
        return await asyncio.to_thread(
            store.create_ticket, req.user_id, req.conv_id, case["case_id"],
            summary.strip(), _ticket_snapshot(case), req.request_id, req.message,
        )
    except Exception:
        return {"success": False, "error": "本地工单存储暂不可用，请稍后重试；若已写入，重试将返回原工单"}


async def get_ticket(req: Request, store: Any) -> Dict[str, Any]:
    """Query only this user/conversation, without changing the active case."""
    from harness.context import harness_mode
    action = parse_ticket_action(req.message)
    if (req.harness_mode or harness_mode()) != "enforce" or not action or action["action"] != "get_ticket":
        return {"success": False, "error": "尚未收到明确的工单查询指令"}
    if not req.user_id or not req.conv_id:
        return {"success": False, "error": "缺少当前用户或会话信息，无法查询工单"}
    if store is None:
        return {"success": False, "error": "本地工单存储尚未配置，无法查询工单"}
    try:
        return await asyncio.to_thread(store.get_ticket, action["ticket_id"], req.user_id, req.conv_id)
    except Exception:
        return {"success": False, "error": "本地工单存储暂不可用，请稍后重试"}


def inspect_request_context(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：返回脱敏后的当前请求快照。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
    }


def suggest_required_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """通用客服工具：按业务类型计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent == "permission_request":
        fields = ["目标系统", "申请用途"]
    elif intent in {"account", "account_security"}:
        fields = ["登录方式或账号标识", "问题发生时间"]
    elif intent in {"complaint", "request"}:
        fields = ["事件时间", "期望处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
    }


def lookup_error_code(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：解释常见错误码的排查方向，不声称读取了服务端日志。"""
    code = str(args.get("error_code", "")).upper().strip()
    mapping = {
        "401": ("认证失败", ["确认 Token/API Key 是否过期", "确认请求时间戳和签名", "确认账号登录状态"]),
        "403": ("权限不足", ["确认账号或套餐权限", "确认资源权限和 IP 白名单"]),
        "404": ("资源或路径不存在", ["确认接口路径和环境", "确认资源标识是否正确"]),
        "500": ("服务端处理异常", ["记录 request_id 和发生时间", "检查依赖服务、参数格式和服务端日志"]),
    }
    meaning, steps = mapping.get(
        code,
        ("暂未识别的错误码", ["补充完整错误信息、发生时间和运行环境"]),
    )
    return {
        "error_code": code,
        "meaning": meaning,
        "next_steps": steps,
        "server_log_checked": False,
    }


def build_diagnostic_plan(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """技术工具：生成低风险排障顺序。"""
    environment = str(args.get("environment", "unknown"))[:80]
    reproduced = bool(args.get("reproduced", False))
    steps = [
        "复现并记录完整错误信息",
        "确认网络、DNS、代理和证书",
        "确认版本、配置和权限",
    ]
    if reproduced:
        steps.append("用最小请求复现并记录 request_id")
    return {
        "environment": environment,
        "reproduced": reproduced,
        "diagnostic_steps": steps,
    }


def lookup_asset(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """设备工具：精确查询公开虚构资产，不查询真实企业资产或执行身份核验。"""
    asset_id = args.get("asset_id") if isinstance(args, dict) else None
    if not isinstance(asset_id, str) or not asset_id.strip():
        return {"success": False, "source": "demo", "error": "asset_id 必须是非空字符串"}
    asset_id = asset_id.strip().upper()
    fields = ("asset_id", "device_type", "device_model", "operating_system",
              "configuration", "warranty_until")
    try:
        assets = json.loads(DEMO_ASSETS_PATH.read_text(encoding="utf-8"))
        if not isinstance(assets, list) or any(
            not isinstance(asset, dict)
            or any(not isinstance(asset.get(field), str) or not asset[field].strip() for field in fields)
            for asset in assets
        ):
            raise ValueError("invalid demo assets")
    except (OSError, ValueError):
        return {"success": False, "source": "demo", "error": "演示资产数据暂不可用，请稍后重试"}

    for asset in assets:
        if asset["asset_id"] == asset_id:
            return {
                "success": True,
                "found": True,
                "source": "demo",
                "asset_id": asset_id,
                "asset": {field: asset[field] for field in fields},
                "notice": "仅为公开虚构演示资产，不代表真实企业资产或保修状态",
            }
    return {
        "success": True,
        "found": False,
        "source": "demo",
        "asset_id": asset_id,
        "asset": None,
        "notice": "未在公开虚构演示资产目录中找到该编号，不代表真实资产不存在",
    }


def check_access_fields(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """检查申请信息是否齐全，不查询身份或执行授权。"""
    fields = {
        "target_system": bool(req.entities.get("target_system")),
        "purpose": bool(req.entities.get("purpose")),
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_grant_access": False,
        "reason": "当前工具只检查申请信息；实际开通需转人工（IT 支持）按审批流程处理",
    }


def create_handoff_summary(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成可交给IT 人员的结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要IT 人员继续核验"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。"""

    async def search_knowledge_base(req: Request, args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = int(args.get("top_k", 5) or 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
                "retrieval": getattr(result, "retrieval", {}),
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
            "retrieval": getattr(result, "retrieval", {}),
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "检索知识库并返回最相关的文档片段；可用于通用、技术、权限和升级场景。",
            {
                "query": {"type": "string", "description": "用户问题或检索关键词"},
                "top_k": {"type": "integer", "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        )
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向用户补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def technical_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_error_code": make_tool(
            "lookup_error_code",
            "解释常见 HTTP 错误码的可能含义和低风险排查方向；不会读取服务端日志。",
            {"error_code": {"type": "string", "description": "例如 401、403、500"}},
            lookup_error_code,
            required=["error_code"],
        ),
        "build_diagnostic_plan": make_tool(
            "build_diagnostic_plan",
            "提供通用低风险排查清单，仅在没有适用的知识库 SOP 时使用；不读取设备状态，不执行修复操作。",
            {
                "environment": {"type": "string", "description": "App、浏览器、服务端或 Docker 等"},
                "reproduced": {"type": "boolean", "description": "问题是否可以稳定复现"},
            },
            build_diagnostic_plan,
            required=["environment", "reproduced"],
        ),
    }


def device_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_asset": make_tool(
            "lookup_asset",
            "按明确的资产编号查询公开虚构演示资产的设备型号、配置和保修日期；不是企业资产系统，不核验身份或归属。未知编号返回未找到，不应编造或猜测资产编号。",
            {"asset_id": {"type": "string", "description": "用户提供的资产编号，例如 NB-001"}},
            lookup_asset,
            required=["asset_id"],
        ),
    }


def access_tools() -> Dict[str, AgentToolSpec]:
    return {
        "check_access_fields": make_tool(
            "check_access_fields",
            "检查当前请求实体中的目标系统和申请用途是否齐全；无需传参，不查询或修改权限。",
            {},
            check_access_fields,
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给IT 人员的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }
