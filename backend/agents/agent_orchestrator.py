"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 GeneralAgent

并行协作：
  - 复杂问题（如"技术问题 + 权限申请"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import copy
import inspect
import json
import logging
import os
import time
import uuid
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from agents.tools import (
    AgentToolSpec,
    build_shared_rag_tools,
    access_tools,
    device_tools,
    escalation_tools,
    general_tools,
    technical_tools,
)
from core.intent_recognizer import (
    IntentCategory, IntentRecognizer, UrgencyLevel,
    NETWORK_ISSUE_KEYWORDS, DEVICE_FAILURE_KEYWORDS, SOFTWARE_INSTALL_KEYWORDS,
    match_domain_keywords,
)
from core.llm_utils import extract_text_content, thinking_request_kwargs

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    GENERAL   = "general"    # 通用客服
    TECHNICAL = "technical"  # 技术支持
    NETWORK   = "network"    # 网络连接与 VPN 故障
    DEVICE    = "device"     # 设备硬件故障
    ACCESS    = "access"     # 账号与权限申请指引
    ESCALATION = "escalation" # 人工升级与交接


@dataclass(frozen=True)
class AgentProfile:

    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


def _env_float(name: str, default: float) -> float:
    """读取可选浮点配置；错误配置不应阻塞服务启动。"""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法浮点配置 %s=%r", name, os.getenv(name))
        return default


def _env_int(name: str, default: int) -> int:
    """读取可选整数配置；错误配置不应阻塞服务启动。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("忽略非法整数配置 %s=%r", name, os.getenv(name))
        return default


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    harness_events: List[Dict[str, Any]] = field(default_factory=list)
    state_updates: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentCallResult:
    """一次 LLM 调用的请求级结果，不在共享 Agent 实例上保存状态。"""
    content: str
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    harness_events: List[Dict[str, Any]] = field(default_factory=list)
    state_updates: List[Dict[str, Any]] = field(default_factory=list)


class AgentCallError(RuntimeError):
    """保留失败前已经发生的工具调用，供请求 trace 记录。"""

    def __init__(self, message: str, tools_used: List[str], tool_traces: List[Dict[str, Any]]):
        super().__init__(message)
        self.tools_used = list(tools_used)
        self.tool_traces = list(tool_traces)
        self.harness_events = []
        self.state_updates = []


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    entities:    Dict[str, List[str]] = field(default_factory=dict)
    intent:      Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency:     Optional[UrgencyLevel]   = None
    intent_confidence: float = 1.0
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    case_state: Dict[str, Any] = field(default_factory=dict)
    harness_mode: Optional[str] = None


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    harness_events: List[Dict[str, Any]] = field(default_factory=list)
    state_updates: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    case_state: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    execution_success: bool = True


@dataclass
class RoutingDecision:
    """一次请求的结构化路由决策。"""
    primary_agent: AgentType
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0

    @property
    def agent_types(self) -> List[AgentType]:
        return [self.primary_agent] + self.supporting_agents

    @property
    def multi_agent(self) -> bool:
        return bool(self.supporting_agents)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用、角色契约和统计。"""

    agent_type: AgentType
    system_prompt: str
    profile: AgentProfile

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        profile: Optional[AgentProfile] = None,
    ):
        self._client = client
        self.profile = profile or self.profile
        self._model  = self.profile.model or model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()
        self._shared_tools: Dict[str, AgentToolSpec] = {}

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        """返回该角色真实可调用的工具白名单。"""
        return dict(self._shared_tools)

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        self._shared_tools = dict(tools or {})

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        try:
            call_result = await self._call_llm(req)
            ms = (time.monotonic() - t0) * 1000
            escalate = self._needs_escalation(call_result.content)
            response = AgentResponse(
                agent_type=self.agent_type,
                content=call_result.content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=call_result.tools_used,
                tool_traces=call_result.tool_traces,
                harness_events=call_result.harness_events,
                state_updates=call_result.state_updates,
            )
            # 只统计已完成请求，避免 Monitor 把正在执行的请求当成失败。
            self.stats.total += 1
            self.stats.success += 1
            self.stats.total_ms += ms
            return response
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total += 1
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            tools_used = ex.tools_used if isinstance(ex, AgentCallError) else []
            tool_traces = ex.tool_traces if isinstance(ex, AgentCallError) else []
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理您的请求时出现问题，请稍后重试。",
                success=False,
                latency_ms=ms,
                tools_used=list(tools_used),
                tool_traces=list(tool_traces),
                harness_events=getattr(ex, "harness_events", []),
                state_updates=getattr(ex, "state_updates", []),
            )

    async def _call_llm(self, req: Request) -> AgentCallResult:
        from harness.runtime import run_agent
        return await run_agent(self, req)

    @staticmethod
    def _block_type(block: Any) -> Optional[str]:
        if isinstance(block, dict):
            return block.get("type")
        return getattr(block, "type", None)

    @staticmethod
    def _block_value(block: Any, key: str) -> Any:
        if isinstance(block, dict):
            return block.get(key)
        return getattr(block, key, None)

    @staticmethod
    def _validate_tool_input(spec: AgentToolSpec, args: Any) -> None:
        if not isinstance(args, dict):
            raise ValueError("工具参数必须是 JSON 对象")
        schema = spec.input_schema
        for field_name in schema.get("required", []):
            if field_name not in args:
                raise ValueError(f"缺少必需参数: {field_name}")
        properties = schema.get("properties", {})
        unknown = set(args) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"不允许的工具参数: {', '.join(sorted(unknown))}")
        type_map = {"string": str, "number": (int, float), "integer": int, "boolean": bool}
        for key, value in args.items():
            expected = properties.get(key, {}).get("type")
            if expected in type_map and not isinstance(value, type_map[expected]):
                raise ValueError(f"参数 {key} 类型错误，期望 {expected}")

    def _build_system_prompt(self, req: Request) -> str:
        """把角色契约和动态 Skills 拼入 system prompt。"""
        profile_prompt = (
            f"\n\n[角色契约]\n"
            f"角色：{self.profile.role}\n"
            f"职责：{self.profile.mission}\n"
            f"处理流程：{' -> '.join(self.profile.workflow)}\n"
            f"可用输入：{'；'.join(self.profile.input_contract)}\n"
            f"输出要求：{'；'.join(self.profile.output_contract)}\n"
            f"升级条件：{'；'.join(self.profile.handoff_conditions) or '无，按通用客服规则处理'}\n"
            f"允许的数据/工具范围：{'、'.join(self.profile.tool_scope) or '仅使用当前请求上下文'}\n"
            "不要声称执行了未提供的查询、修改或授权操作；缺少证据时明确说明需要核验。\n"
            "[IT 支持实体] target_system=目标系统，issue_description=问题现象，date=发生时间，"
            "error_code=错误码，requested_access=申请权限，purpose=申请用途，"
            "operating_system=操作系统，client_version=客户端版本，network_type=网络环境，"
            "impact_scope=影响范围，device_type=设备类型，device_model=设备型号。"
            "实体仅由本轮消息的规则匹配得到，空值表示未提取到，不代表用户没有该信息；结合原文和已有上下文理解。"
        )
        base_prompt = f"{self.system_prompt}{profile_prompt}"
        if self._skill_manager is None:
            return base_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return base_prompt
        return f"{base_prompt}\n\n[动态 Skills]\n{skill_prompt}"

    def _build_role_packet(self, req: Request) -> str:
        """给子 Agent 的确定性输入包；子类可补充领域字段。"""
        packet = {
            "agent_type": self.agent_type.value,
            "intent": req.intent.value if req.intent else None,
            "intent_group": req.intent_group,
            "urgency": req.urgency.name if req.urgency else None,
            "intent_confidence": round(req.intent_confidence, 4),
            "available_entities": req.entities or {},
        }
        return json.dumps(packet, ensure_ascii=False)

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "人工客服", "escalate", "specialist", "无法处理"]
        return any(kw in content for kw in keywords)


class GeneralAgent(BaseAgent):
    agent_type    = AgentType.GENERAL
    profile = AgentProfile(
        role="IT 咨询分诊与首轮接待",
        mission="快速回答基础问题，澄清不完整需求，并识别是否需要专业 Agent 或人工处理。",
        workflow=("复述诉求", "判断业务范围", "直接回答或补充必要信息", "给出下一步"),
        input_contract=("对话历史", "用户画像", "意图与紧急度", "知识库上下文"),
        output_contract=("先回应核心问题", "信息不足时只询问必要字段", "明确下一步和边界"),
        handoff_conditions=("涉及实际权限操作、隐私或复杂投诉", "用户明确要求人工"),
        tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
        temperature=0.3,
        max_tokens=900,
    )
    system_prompt = (
        "你是 Nexo 企业 IT 支持助手。友好、简洁地回答软件使用、IT 制度与基础咨询。非 IT 问题说明服务范围。"
        "如果问题超出你的能力范围，明确说明并建议转人工（IT 支持），不声称已代办。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["triage_targets"] = ["technical", "access", "network", "device", "escalation"]
        packet["response_mode"] = "answer_or_clarify"
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(general_tools())
        return tools


class TechnicalAgent(BaseAgent):
    agent_type    = AgentType.TECHNICAL
    profile = AgentProfile(
        role="技术故障诊断与排障",
        mission="基于错误码、环境和复现信息缩小根因范围，给出低风险、可验证的排查步骤。",
        workflow=("确认现象", "判断影响范围", "按网络/权限/配置/依赖排查", "给出验证方式", "判断升级条件"),
        input_contract=("错误码", "问题发生时间", "运行环境", "影响范围", "最近变更", "知识库上下文"),
        output_contract=("现象复述", "可能原因", "编号排查步骤", "验证结果", "需要补充的信息"),
        handoff_conditions=("生产大面积不可用", "数据丢失或权限异常", "需要后台日志、数据库或人工操作"),
        tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = (
        "你是 Nexo 的 IT 技术支持专家。专注于办公软件安装更新、登录、应用崩溃和通用技术问题的故障排查、错误诊断与配置指引。明确的网络连接故障由 Network 支持，设备硬件故障由 Device 支持。"
        "提供清晰的步骤化解决方案。遇到需要后台操作的问题，说明需要升级处理。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["diagnostic_fields"] = {
            "error_codes": req.entities.get("error_code", []),
            "environment_hint": "请从用户消息和背景中确认设备、系统、版本、网络",
            "risk_boundary": "不得要求密码、验证码、完整密钥；不得建议破坏性操作",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(technical_tools())
        return tools


class NetworkAgent(TechnicalAgent):
    agent_type = AgentType.NETWORK
    profile = AgentProfile(
        role="网络连接与 VPN 故障排查",
        mission="区分本机、网络环境和服务端问题，按知识库 SOP 给出低风险且可验证的连接排查步骤。",
        workflow=("确认连接现象", "确认网络环境和影响范围", "查阅适用 SOP", "按已有排查结果继续", "验证或交接"),
        input_contract=("目标系统", "问题现象", "错误码", "操作系统", "客户端版本", "网络环境", "影响范围", "知识库上下文"),
        output_contract=("问题判断", "适用 SOP 的必要步骤", "验证方式", "尚缺信息", "升级条件"),
        handoff_conditions=("多名员工持续无法连接", "需要网络管理员操作", "完成适用排查后仍未恢复"),
        tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = (
        "你是 Nexo 的网络支持专家，负责 Wi-Fi、有线网络、DNS 和 VPN 连接故障。"
        "先查询知识库，找到适用 SOP 后保留必要步骤，不再调用通用诊断工具生成另一套流程。"
        "结合用户已执行的操作继续排查，不把未执行的建议记作完成。"
        "申请 VPN 使用权限属于 Access；软件崩溃属于 Technical；硬件损坏属于 Device。"
        "没有实际网络探测或后台管理工具，不声称已测试连接、修改配置或创建工单。"
    )


class DeviceAgent(BaseAgent):
    agent_type = AgentType.DEVICE
    profile = AgentProfile(
        role="办公设备故障排查",
        mission="根据设备类型、型号和故障现象查阅设备 SOP，先区分软硬件与连接原因，再提供低风险检查与人工维修交接指引。",
        workflow=("确认设备和现象", "检查安全风险", "查阅适用 SOP", "给出外部可操作检查", "验证或交接"),
        input_contract=("设备类型", "设备型号", "问题现象", "发生时间", "影响范围", "知识库上下文"),
        output_contract=("现象确认", "必要检查步骤", "验证方式", "需补充信息", "维修交接边界"),
        handoff_conditions=("冒烟、异味、电池鼓包或漏电", "需要拆机或更换部件", "适用检查后仍未恢复"),
        tool_scope=("search_knowledge_base", "lookup_asset"),
        temperature=0.1,
        max_tokens=1200,
    )
    system_prompt = (
        "你是 Nexo 的办公设备支持专家，负责电脑、手机和打印机等设备的故障初诊与硬件排查。"
        "原因尚不明确的设备异常先区分软硬件与连接问题，不预设硬件损坏。"
        "先查询知识库，按适用设备 SOP 回答，信息不足时只询问本轮必要字段。"
        "不要把软件网络通用诊断流程套用到硬件故障；已明确的应用安装和崩溃由 Technical 处理，已明确的网络连接故障由 Network 处理。"
        "只提供安全的外部检查，不要求拆机、带电维修或反复强制启动。"
        "可通过 lookup_asset 查询公共演示资产资料；没有远程控制和报修工具，不声称已检查设备或安排维修。"
    )

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(device_tools())
        return tools


class AccessAgent(BaseAgent):
    agent_type = AgentType.ACCESS
    profile = AgentProfile(
        role="账号与权限申请指引",
        mission="解释企业账号与权限申请流程，检查申请信息是否齐全，明确人工审批和实际操作边界。",
        workflow=("确认账号或权限诉求", "收集目标系统与申请用途", "查阅申请规范", "说明下一步", "判断是否升级"),
        input_contract=("目标系统", "申请用途", "账号问题描述", "知识库上下文"),
        output_contract=("需要补充的信息", "适用申请流程", "下一步处理路径", "人工审批边界"),
        handoff_conditions=("实际开通或撤销权限", "重置密码或处理账号锁定", "疑似账号被盗", "需要身份核验"),
        tool_scope=("search_knowledge_base", "check_access_fields"),
        temperature=0.0,
        max_tokens=1100,
    )
    system_prompt = (
        "你是 Nexo 的账号与权限支持专家，负责企业账号咨询与权限申请指引。"
        "按知识库说明流程；使用 check_access_fields 检查已提取的目标系统与用途，按缺失字段补问。"
        "你未接入身份、审批或权限系统，不得声称已查询权限、提交申请、完成审批或开通账号。"
        "登录故障由 Technical 排查，网络连接故障由 Network 排查；需要实际操作时建议转人工（IT 支持），不索要密码和验证码。"
    )

    def _build_role_packet(self, req: Request) -> str:
        packet = json.loads(super()._build_role_packet(req))
        packet["application_fields"] = {
            "target_system": req.entities.get("target_system", []),
            "purpose": req.entities.get("purpose", []),
            "missing_fields": [
                field for field, values in (
                    ("目标系统", req.entities.get("target_system", [])),
                    ("申请用途", req.entities.get("purpose", [])),
                ) if not values
            ],
            "risk_boundary": "只提供指引，不查询身份、不执行授权、不承诺审批结果",
        }
        return json.dumps(packet, ensure_ascii=False)

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(access_tools())
        return tools


class EscalationAgent(BaseAgent):
    """人工升级节点。

    升级不是一个普通问答 Prompt：它应该生成标准化的交接信息并停止普通
    Agent 继续编造答案。生产环境可在这里接工单系统、人工队列或 Webhook。
    """

    agent_type = AgentType.ESCALATION
    profile = AgentProfile(
        role="人工升级与交接",
        mission="确认升级原因，整理已知上下文，告知用户下一步，不执行未经授权的业务操作。",
        workflow=("确认升级原因", "整理已知信息", "标记优先级", "生成交接摘要"),
        input_contract=("用户消息", "意图", "紧急度", "结构化实体", "对话背景"),
        output_contract=("升级原因", "已知信息摘要", "还需补充的信息", "保守的后续说明"),
        handoff_conditions=("用户明确要求人工", "紧急或高风险场景"),
        tool_scope=("search_knowledge_base", "create_handoff_summary"),
        temperature=0.0,
        max_tokens=500,
    )
    system_prompt = "你负责 IT 人工升级交接，不要模拟已完成的后台操作。"

    def get_tools(self) -> Dict[str, AgentToolSpec]:
        tools = super().get_tools()
        tools.update(escalation_tools())
        return tools

    @staticmethod
    def latest_user_report(req):
        """Keep the user's latest report as a quote, not an executed-step claim."""
        import re
        message = (req.message or "").strip()
        handoff_only = re.fullmatch(
            r"(?:那你|那|请你|请|麻烦你|麻烦)?(?:帮我)?(?:转(?:接|交)?给?人工(?:客服)?|联系人工(?:客服)?)(?:吧|一下|处理)?[。！!？?\s]*",
            message,
        )
        if message and not handoff_only:
            return {"content": message, "source": "user_reported", "request_id": req.request_id}
        if not req.case_state.get("new_topic"):
            previous = req.case_state.get("latest_user_report", {})
            if previous.get("source") == "user_reported":
                return dict(previous)
        return {}

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        import re

        entity_labels = {
            "target_system": "目标系统",
            "issue_description": "问题现象",
            "date": "发生时间",
            "error_code": "错误码",
            "requested_access": "申请权限",
            "purpose": "申请用途",
            "operating_system": "操作系统",
            "client_version": "客户端版本",
            "network_type": "网络环境",
            "impact_scope": "影响范围",
            "device_type": "设备类型",
            "device_model": "设备型号",
        }
        # 仅整理交接展示字段，不修改 req.entities 或全局实体提取规则。
        aliases = {
            "target_system": ("目标系统", "涉及系统", "系统名称", "应用名称"),
            "issue_description": ("问题现象", "故障现象", "账号恢复", "问题"),
            "date": ("发生时间", "故障时间"),
            "error_code": ("错误码", "状态码"),
            "requested_access": ("申请权限", "期望权限", "所需权限", "权限类型", "权限"),
            "purpose": ("申请用途", "用途"),
            "operating_system": ("操作系统", "系统版本"),
            "client_version": ("客户端版本", "软件版本"),
            "network_type": ("网络环境", "网络类型"),
            "impact_scope": ("影响范围",),
            "device_type": ("设备类型",),
            "device_model": ("设备型号", "电脑型号", "打印机型号"),
        }
        new_fields = ("operating_system", "client_version", "network_type",
                      "impact_scope", "device_type", "device_model")
        user_sources = [message.get("content") or "" for message in req.history or []
                        if message.get("role") == "user"]
        from harness.context import harness_mode
        if (req.harness_mode or harness_mode()) == "enforce" and req.case_state.get("new_topic"):
            user_sources = []
        user_sources.append(req.message)
        context = req.context or ""
        sources = []
        if context.startswith("[会话摘要]\n"):
            summary = context[len("[会话摘要]\n"):]
            for header in ("[相关历史]", "[用户画像]", "[最近对话]"):
                summary = summary.split("\n\n" + header, 1)[0]
            sources.append(summary)
        sources.extend(
            message.get("content") or ""
            for message in req.history or []
            if message.get("role") in ("user", "assistant")
        )
        if not req.history:
            marker = "[最近对话]\n\n"
            if context.startswith(marker):
                sources.append(context[len(marker):])
            elif "\n\n" + marker in context:
                sources.append(context.split("\n\n" + marker, 1)[1])

        from harness.context import harness_mode
        if (req.harness_mode or harness_mode()) == "enforce":
            # 助手建议与自动摘要不能作为用户已确认的交接事实。
            sources = [message.get("content") or "" for message in req.history or []
                       if message.get("role") == "user"]
            if req.case_state.get("new_topic"):
                sources = []
            sources.append(req.message)
        fields = {}
        for source in sources:
            text = source.replace("**", "").replace("`", "")
            for line in re.split(r"[\n；;]", text):
                line = re.sub(r"^\s*(?:[-*>]+|\d+[.)、])\s*", "", line).strip()
                for key, names in aliases.items():
                    if key in new_fields and source not in user_sources:
                        continue
                    labels = "|".join(re.escape(name) for name in names)
                    # 已确认的 Markdown 表格、字段列表及显式更正，后出现的值优先。
                    match = re.match(rf"^\|\s*(?:{labels})\s*\|\s*([^|]+)\|", line)
                    if not match:
                        separator = r"(?:[:：=]|更正为|修改为|调整为|改为|改成|只需要|需要|是|为)"
                        match = re.search(rf"(?:^|[，,。])\s*(?:{labels})\s*{separator}\s*(.+)", line)
                    if match:
                        value = match.group(1).strip().strip("。 ")
                        if key != "issue_description":
                            value = re.split(r"[，,。]", value, maxsplit=1)[0].strip()
                        if value and not re.search(r"请提供|请补充|待补充|待确认|未提供|未知", value):
                            fields[key] = value[:120]
            # 为常见的非字段式问题描述保留简短要点，不转贴说明和排查步骤。
            if "issue_description" not in fields:
                issues = re.findall(r"忘记密码|绑定手机号已停用|手机号已停用|手机号停用|无法登录|登录失败|连接超时|连接失败|无法连接|崩溃|闪退", text)
                if issues:
                    fields["issue_description"] = "，".join(dict.fromkeys(issues))
            if "purpose" not in fields:
                purpose = re.search(r"(?:用于|用来)\s*([^，。；;！？!?\n]+)", text)
                if purpose:
                    fields["purpose"] = purpose.group(1).strip()[:120]
            codes = re.findall(r"(?:错误码|状态码|报错?|HTTP)\s*[:：#]?\s*([45]\d{2})\b", text, re.I)
            if codes:
                fields["error_code"] = "、".join(dict.fromkeys(codes))
        # 新领域字段仅从用户原话提取，复用同一套规则，不把助手建议当成环境事实。
        recognizer = IntentRecognizer.__new__(IntentRecognizer)
        for source in user_sources:
            extracted = recognizer._extract_entities(source)
            for key in new_fields:
                if extracted.get(key):
                    fields[key] = "、".join(extracted[key])[:120]
            issues = match_domain_keywords(source, NETWORK_ISSUE_KEYWORDS + DEVICE_FAILURE_KEYWORDS)
            if issues:
                existing = re.split(r"[，、]", fields.get("issue_description", ""))
                issues = list(dict.fromkeys(item for item in existing + issues if item))
                fields["issue_description"] = "，".join(issues)[:120]
        from harness.context import harness_mode
        blocked_new_fields = set()
        if (req.harness_mode or harness_mode()) == "enforce" and req.case_state:
            # 未确认候选不能通过旧助手回复或摘要重新成为交接事实。
            for candidate in req.case_state.get("candidates", []):
                fields.pop(candidate.get("field"), None)
                if candidate.get("field") in new_fields:
                    blocked_new_fields.add(candidate["field"])
        for key in entity_labels:
            if key in blocked_new_fields:
                continue
            if (req.entities or {}).get(key):
                fields[key] = "、".join(req.entities[key])[:120]

        if (req.harness_mode or harness_mode()) == "enforce" and req.case_state:
            from harness.state import handoff_fields
            fields.update(handoff_fields(req.case_state))
            for conflict in req.case_state.get("conflicts", []):
                fields.pop(conflict.get("field"), None)
        details = [f"- {label}：{fields[key]}" for key, label in entity_labels.items() if fields.get(key)]
        sections = ["我已将您的问题标记为 IT 人工处理。"]
        if details:
            sections.append("**交接信息**\n" + "\n".join(details))
        else:
            sections.append("暂未提取到具体信息，请补充涉及的系统、问题现象或申请需求。")
        if (req.harness_mode or harness_mode()) == "enforce" and req.case_state:
            asset_lookup = req.case_state.get("asset_lookup") or {}
            if asset_lookup.get("source") == "demo":
                asset_id = str(asset_lookup.get("asset_id") or "未提供有效编号")[:80]
                if asset_lookup.get("status") == "found":
                    asset_model = str((asset_lookup.get("asset") or {}).get("device_model") or "")[:120]
                    asset_lines = [f"- 演示资产编号：{asset_id}", f"- 演示资料型号：{asset_model}",
                                   "- 来源：lookup_asset 返回的公开虚构演示资料，不是用户陈述或真实资产核验结果。"]
                    user_model = fields.get("device_model")
                    if user_model and user_model.strip().lower() != asset_model.strip().lower():
                        asset_lines.append("- 用户提供的设备型号与演示资料型号不一致，需由 IT 核对，不能用演示资料覆盖用户描述。")
                else:
                    status = "未找到该编号" if asset_lookup.get("status") == "not_found" else "本次查询不可用"
                    asset_lines = [f"- 演示资产编号：{asset_id}", f"- 演示查询结果：{status}，没有可用型号资料。",
                                   "- 来源：公开虚构演示资产查询，不代表真实企业资产状态。"]
                sections.append("**演示资产资料（工具返回）**\n" + "\n".join(asset_lines))
            completed = [step for step in req.case_state.get("steps", [])
                         if step.get("status") == "user_reported_done"
                         and step.get("source") == "user_reported"
                         and step.get("source_quote") and step.get("result")]
            if completed:
                step_lines = [f"- {str(step.get('description', ''))[:120]}：{str(step['result'])[:120]}"
                              for step in completed[-4:]]
                sections.append("**已尝试及结果（用户反馈）**\n" + "\n".join(step_lines))
        if (req.harness_mode or harness_mode()) == "enforce":
            report = self.latest_user_report(req)
            if report.get("content"):
                quoted = report["content"][:300]
                if len(report["content"]) > 300:
                    quoted += "…"
                sections.append("**最新反馈（用户原话）**\n" + quoted
                                + "\n以上是用户反馈，待 IT 人员核验，不代表系统已代为执行或确认解决。")
        sections.append("请不要发送密码、短信验证码或完整密钥；以上仅为交接说明，尚未创建工单或通知 IT 人员。")
        content = "\n\n".join(sections)
        ms = (time.monotonic() - t0) * 1000
        self.stats.success += 1
        self.stats.total_ms += ms
        return AgentResponse(
            agent_type=self.agent_type,
            content=content,
            success=True,
            latency_ms=ms,
            escalate=True,
            tools_used=[],
        )


class ResponseComposer:
    """多 Agent 汇总节点，统一主次、去重和输出边界。"""

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model = model
        self._skill_manager = skill_manager

    async def compose(self, req: Request, responses: List[AgentResponse]) -> str:
        successful = [response for response in responses if response.success and response.content.strip()]
        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        if len(successful) == 1:
            return successful[0].content

        evidence = "\n\n".join(
            f"[{response.agent_type.value} Agent 输出]\n{response.content}"
            for response in successful
        )
        prompt = (
            "你是客服 Response Composer，负责把多个专业 Agent 的结果合并成一条最终回复。\n"
            "要求：以主 Agent 的结论为主，按用户问题优先级组织内容；去掉重复和冲突表述；"
            "不能补造权限、审批、工单或后台查询结果；如果结论冲突，明确说明需要核验；"
            "保留必要的排查步骤、核验字段和升级边界。只输出给用户看的中文回复，不要提及 Agent。\n\n"
            f"主 Agent：{successful[0].agent_type.value}\n"
            f"用户问题：{req.message}\n"
            f"候选结果：\n{evidence}"
        )
        from harness.context import case_context, harness_mode
        if (req.harness_mode or harness_mode()) == "enforce" and req.case_state:
            prompt += "\n\n" + case_context(req.case_state)
        if self._skill_manager is not None:
            skill = self._skill_manager.prompt_for(req.message, "general")
            if skill:
                prompt += f"\n\n[通用客服输出边界]\n{skill}"
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=_env_int("NEXO_COMPOSER_MAX_TOKENS", 1000),
                temperature=_env_float("NEXO_COMPOSER_TEMPERATURE", 0.1),
                messages=[{"role": "user", "content": prompt}],
                **thinking_request_kwargs(),
            )
            content = extract_text_content(response.content).strip()
            if content:
                return content
        except Exception as ex:
            logger.warning("Response Composer 失败，使用确定性合并: %s", ex)

        # 汇总节点不可用时保留主次标签，避免丢失某个专业 Agent 的结论。
        return "\n\n".join(
            f"{response.content}" if index == 0 else f"补充说明：\n{response.content}"
            for index, response in enumerate(successful)
        )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 GeneralAgent
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.TECHNICAL:  AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN: AgentType.TECHNICAL,
        IntentCategory.TECHNICAL_CRASH: AgentType.TECHNICAL,
        IntentCategory.SOFTWARE_INSTALL: AgentType.TECHNICAL,
        IntentCategory.NETWORK: AgentType.NETWORK,
        IntentCategory.NETWORK_ISSUE: AgentType.NETWORK,
        IntentCategory.DEVICE: AgentType.DEVICE,
        IntentCategory.DEVICE_FAILURE: AgentType.DEVICE,
        IntentCategory.ACCOUNT:    AgentType.ACCESS,
        IntentCategory.ACCOUNT_SECURITY: AgentType.ACCESS,
        IntentCategory.PERMISSION_REQUEST: AgentType.ACCESS,
        IntentCategory.ESCALATION: AgentType.ESCALATION,
        IntentCategory.HUMAN_HANDOFF: AgentType.ESCALATION,
        # 其余意图 → GENERAL（默认）
    }

    # Monitor 的降权达到该值时，专业 Agent 暂停接收新请求并回退到 GeneralAgent。
    # 使用环境变量可按线上数据调整，默认值与 Monitor 的严重成功率惩罚上限一致。
    MONITOR_FALLBACK_PENALTY = 0.5

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        rag_tool_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._composer = ResponseComposer(client, model, skill_manager)
        self._shared_tools: Dict[str, AgentToolSpec] = {}
        self._recent_tool_traces = deque(maxlen=_env_int("NEXO_TOOL_TRACE_MAX", 200))

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.GENERAL: [self._make_agent(GeneralAgent, client, model, skill_manager)],
            AgentType.TECHNICAL: [self._make_agent(TechnicalAgent, client, model, skill_manager)],
            AgentType.ACCESS: [self._make_agent(AccessAgent, client, model, skill_manager)],
            AgentType.NETWORK: [self._make_agent(NetworkAgent, client, model, skill_manager)],
            AgentType.DEVICE: [self._make_agent(DeviceAgent, client, model, skill_manager)],
            AgentType.ESCALATION: [self._make_agent(EscalationAgent, client, model, skill_manager)],
        }
        self.set_shared_tools(build_shared_rag_tools(rag_tool_manager))

    @staticmethod
    def _make_agent(
        agent_cls: type[BaseAgent],
        client: AsyncAnthropic,
        default_model: str,
        skill_manager: Optional[Any],
    ) -> BaseAgent:
        """按角色创建 Agent，并允许用环境变量覆盖该角色的模型。

        可使用更强模型，通用接待可使用更快模型，升级节点本身不需要调用 LLM。
        """
        profile = agent_cls.profile
        env_name = f"NEXO_{agent_cls.agent_type.value.upper()}_MODEL"
        model = os.getenv(env_name, "").strip() or profile.model
        configured_profile = replace(profile, model=model) if model else profile
        return agent_cls(client, default_model, skill_manager, profile=configured_profile)

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        self._composer._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_shared_tools(self, tools: Optional[Dict[str, AgentToolSpec]]) -> None:
        """更新所有 Agent 共享的工具白名单。"""
        self._shared_tools = dict(tools or {})
        for agents in self._pool.values():
            for agent in agents:
                agent.set_shared_tools(self._shared_tools)

    async def recognize_intent(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """对外暴露意图识别，供 API 层先判断是否需要 RAG 等前置能力。"""
        return await self._intent_recognizer.recognize(message, history=history)

    def _record_tool_trace(self, result: OrchestratorResult) -> None:
        trace = {
            "request_id": result.request_id,
            "timestamp": datetime.now().isoformat(),
            "intent": result.intent.value if result.intent else None,
            "primary_agent": result.primary_agent.value if result.primary_agent else None,
            "supporting_agents": [agent.value for agent in result.supporting_agents],
            "tools_used": list(result.tools_used),
            "tool_calls": list(result.tool_traces),
            "escalated": result.escalated,
            "latency_ms": round(result.latency_ms, 1),
            "harness_events": list(result.harness_events),
            "validation": dict(result.validation),
            "case_state": dict(result.case_state),
        }
        self._recent_tool_traces.append(trace)

    def get_tool_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        for trace in reversed(self._recent_tool_traces):
            if trace.get("request_id") == request_id:
                return trace
        return None

    def get_recent_tool_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._recent_tool_traces:
            return []
        limit = max(1, min(int(limit or 20), len(self._recent_tool_traces)))
        return list(reversed(list(self._recent_tool_traces)[-limit:]))

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence

        if self._needs_clarification(req):
            result = OrchestratorResult(
                request_id=req.request_id,
                response="我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？",
                agent_type=AgentType.GENERAL,
                intent=req.intent,
                escalated=False,
                latency_ms=(time.monotonic() - t0) * 1000,
                agent_types=[AgentType.GENERAL],
                primary_agent=AgentType.GENERAL,
                routing_reason="低置信度 OTHER 意图，先澄清用户需求",
                routing_confidence=req.intent_confidence,
            )
            return await self._finalize_result(req, result)

        # 复杂问题自动并行协作，例如同一句同时涉及登录故障和扣款/退款。
        decision = self._route_decision(req)
        if decision.multi_agent:
            return await self.run_parallel(req, decision)

        # 2. 执行主 Agent（含降级）
        response = await self._execute(req, decision.primary_agent)

        # 4. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION,
            IntentCategory.HUMAN_HANDOFF,
        ):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知人工客服

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(response.tools_used),
            tool_traces=list(response.tool_traces),
            execution_success=response.success,
            harness_events=list(response.harness_events),
            state_updates=list(response.state_updates),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )
        return await self._finalize_result(req, result)

    async def run_parallel(self, req: Request, decision: RoutingDecision) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        t0 = time.monotonic()
        agent_types = decision.agent_types
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        valid_responses = [r for r in responses if isinstance(r, AgentResponse)]
        from harness.context import harness_mode
        mode = req.harness_mode or harness_mode()
        merged_state = {}
        merge_events = []
        composer_req = req
        if mode == "enforce":
            merged_state = self._merge_case_state(
                req, [update for response in valid_responses for update in response.state_updates],
                [trace for response in valid_responses for trace in response.tool_traces],
                merge_events,
            )
            composer_req = replace(req, case_state=merged_state)
        combined = await self._composer.compose(composer_req, valid_responses)
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)
        tools_used = list(dict.fromkeys(
            tool_name
            for response in valid_responses
            for tool_name in response.tools_used
        ))
        tool_traces = [
            trace
            for response in valid_responses
            for trace in response.tool_traces
        ]
        result = OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=decision.primary_agent,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[
                r.agent_type for r in responses
                if isinstance(r, AgentResponse) and r.success
            ] or agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tools_used=tools_used,
            tool_traces=tool_traces,
            harness_events=[event for response in valid_responses for event in response.harness_events] + merge_events,
            state_updates=[update for response in valid_responses for update in response.state_updates],
            case_state=merged_state,
            execution_success=any(response.success and response.content.strip() for response in valid_responses),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
        )
        return await self._finalize_result(req, result)

    @staticmethod
    def _merge_case_state(req, updates, traces, events):
        from harness.state import apply_progress, apply_asset_lookup, new_case
        from harness.runtime import evidence_ids
        state = req.case_state or new_case(req.user_id, req.conv_id)
        topic_reset = False
        for update in updates:
            candidate = dict(update)
            if candidate.get("new_case") and topic_reset:
                candidate.pop("new_case")
            try:
                state = apply_progress(state, candidate, req.message, req.request_id,
                                       evidence_refs=evidence_ids(traces))
                topic_reset = topic_reset or bool(candidate.get("new_case"))
            except ValueError as ex:
                events.append({"event": "state_update_rejected", "reason": str(ex)})
        for trace in traces:
            state = apply_asset_lookup(state, trace, req.request_id)
        return state

    async def _finalize_result(self, req: Request, result: OrchestratorResult) -> OrchestratorResult:
        """One delivery boundary for single, composed and fallback answers."""
        finalize_started = time.monotonic()
        from harness.context import harness_mode
        from harness.runtime import evidence_ids
        from harness.state import apply_progress, new_case
        from harness.validation import ValidationResult, finalize_answer

        mode = req.harness_mode or harness_mode()
        if mode == "enforce":
            if not result.case_state:
                result.case_state = self._merge_case_state(req, result.state_updates,
                                                          result.tool_traces, result.harness_events)
        else:
            result.case_state = dict(req.case_state or {})
        if mode == "enforce" and not result.execution_success:
            checked = ValidationResult(
                content=result.response, status="runtime_failed", escalated=True,
                issues=["Agent 执行失败，未生成可核验的最终回答"],
                events=[{"event": "runtime_failure_handoff", "reason": "all_agent_attempts_failed"}],
            )
        else:
            checked = await finalize_answer(
                req.message, result.response, result.tool_traces,
                self._composer._client, self._composer._model, mode,
                case_state=result.case_state,
            )
        result.response = checked.content
        result.validation = {"status": checked.status, "issues": checked.issues,
                             "extra_calls": checked.extra_calls, "repair_count": checked.repair_count}
        result.harness_events.extend(checked.events)
        result.escalated = result.escalated or checked.escalated
        if mode == "enforce" and checked.status in {"needs_handoff", "runtime_failed"}:
            handoff = self._best_agent(AgentType.ESCALATION)
            if handoff is not None:
                handoff_req = replace(req, case_state=result.case_state)
                handoff_result = await handoff.handle(handoff_req)
                result.response = handoff_result.content
                result.agent_type = AgentType.ESCALATION
                if AgentType.ESCALATION not in result.agent_types:
                    result.agent_types.append(AgentType.ESCALATION)
                result.harness_events.append({"event": "validation_handoff"})
        explicit_handoff = (
            req.intent in {IntentCategory.HUMAN_HANDOFF, IntentCategory.ESCALATION}
            or result.agent_type == AgentType.ESCALATION
            or result.primary_agent == AgentType.ESCALATION
            or checked.status in {"needs_handoff", "runtime_failed"}
        )
        if mode == "enforce" and explicit_handoff and result.case_state:
            result.case_state = copy.deepcopy(result.case_state)
            report = EscalationAgent.latest_user_report(replace(req, case_state=result.case_state))
            if report:
                result.case_state["latest_user_report"] = report
            result.case_state["status"] = "handoff_ready"
        result.latency_ms += (time.monotonic() - finalize_started) * 1000
        self._record_tool_trace(result)
        return result

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射
          2. 紧急度覆盖（CRITICAL 直接升级）
          3. 默认 GENERAL
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.GENERAL

    def _route_decision(self, req: Request) -> RoutingDecision:
        """
        结构化路由决策。

        先处理紧急/转人工，再用领域分数决定主 Agent 和辅助 Agent。
        这样可以表达“主处理 + 辅助诊断”，避免关键词命中后无主次地拼接。
        """
        if req.urgency == UrgencyLevel.CRITICAL:
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason="紧急度为 CRITICAL，触发升级路由",
                confidence=1.0,
            )

        if req.intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return RoutingDecision(
                primary_agent=AgentType.ESCALATION,
                reason=f"意图为 {req.intent.value if req.intent else 'unknown'}，触发升级路由",
                confidence=max(req.intent_confidence, 0.8),
            )

        scores = self._domain_scores(req)
        degraded_agents = {
            agent_type
            for agent_type in (AgentType.TECHNICAL, AgentType.ACCESS, AgentType.NETWORK, AgentType.DEVICE)
            if scores.get(agent_type, 0.0) > 0 and self._monitor_degraded(agent_type)
        }
        available_scores = {
            agent_type: score
            for agent_type, score in scores.items()
            if agent_type == AgentType.GENERAL
            or (self._pool.get(agent_type) and agent_type not in degraded_agents)
        }
        if not available_scores:
            return RoutingDecision(
                primary_agent=AgentType.GENERAL,
                reason="无可用专属 Agent，降级到 GeneralAgent",
                confidence=0.1,
            )

        ordered = sorted(available_scores.items(), key=lambda item: item[1], reverse=True)
        primary_agent, primary_score = ordered[0]

        collaboration_targets = self._collaboration_targets(req)
        supporting_agents = [
            agent_type
            for agent_type in collaboration_targets
            if agent_type != primary_agent and agent_type in available_scores
        ]

        if not supporting_agents:
            supporting_agents = [
                agent_type
                for agent_type, score in ordered[1:]
                if agent_type != AgentType.GENERAL
                and score >= 0.45
                and score >= primary_score * 0.55
            ]

        reason = self._routing_reason(req, available_scores, primary_agent, supporting_agents)
        if degraded_agents:
            degraded_text = ", ".join(sorted(agent.value for agent in degraded_agents))
            reason += f", monitor_fallback=[{degraded_text}]"
        return RoutingDecision(
            primary_agent=primary_agent,
            supporting_agents=supporting_agents,
            reason=reason,
            confidence=round(min(primary_score, 1.0), 3),
        )

    @staticmethod
    def _technical_keywords(req: Request, include_verification: bool = False) -> List[str]:
        """明确网络/设备故障时，通用报错和连接词不再重复触发技术协作。"""
        msg = req.message.lower()
        specialised = req.intent in (
            IntentCategory.NETWORK, IntentCategory.NETWORK_ISSUE,
            IntentCategory.DEVICE, IntentCategory.DEVICE_FAILURE,
        ) or bool(match_domain_keywords(msg, NETWORK_ISSUE_KEYWORDS + DEVICE_FAILURE_KEYWORDS))
        if specialised:
            keywords = ["崩溃", "闪退", "crash", "应用报错", "页面报错", "接口报错", "软件报错",
                        "无法登录", "登录失败", "401", "验证码"]
        else:
            keywords = ["崩溃", "报错", "error", "crash", "无法登录", "登录失败", "500", "401", "连不上", "连接失败"]
        if include_verification and not specialised:
            keywords.append("验证码")
        return keywords + match_domain_keywords(msg, SOFTWARE_INSTALL_KEYWORDS)

    def _domain_scores(self, req: Request) -> Dict[AgentType, float]:
        """按意图、关键词和实体为各领域 Agent 打分。"""
        msg = req.message.lower()
        scores = {
            AgentType.GENERAL: 0.1,
            AgentType.TECHNICAL: 0.0,
            AgentType.ACCESS: 0.0,
            AgentType.NETWORK: 0.0,
            AgentType.DEVICE: 0.0,
        }

        if req.intent in (
            IntentCategory.QUERY,
            IntentCategory.REQUEST,
            IntentCategory.COMPLAINT,
            IntentCategory.GREETING,
            IntentCategory.FEEDBACK,
            IntentCategory.OTHER,
        ):
            scores[AgentType.GENERAL] += 0.55

        if req.intent in (
            IntentCategory.TECHNICAL,
            IntentCategory.TECHNICAL_LOGIN,
            IntentCategory.TECHNICAL_CRASH,
            IntentCategory.SOFTWARE_INSTALL,
        ):
            scores[AgentType.TECHNICAL] += 0.75

        if req.intent in (
            IntentCategory.ACCOUNT,
            IntentCategory.ACCOUNT_SECURITY,
            IntentCategory.PERMISSION_REQUEST,
        ):
            scores[AgentType.ACCESS] += 0.75

        technical_kws = self._technical_keywords(req, include_verification=True)
        access_kws = ["申请权限", "项目权限", "访问权限", "权限申请", "开通", "permission", "只读权限"]
        general_kws = ["安装", "办公软件", "打印机", "操作手册", "咨询", "帮助"]

        technical_hits = sum(1 for kw in technical_kws if kw in msg)
        access_hits = sum(1 for kw in access_kws if kw in msg)
        general_hits = sum(1 for kw in general_kws if kw in msg)

        scores[AgentType.TECHNICAL] += min(0.45, technical_hits * 0.18)
        scores[AgentType.ACCESS] += min(0.45, access_hits * 0.18)
        scores[AgentType.GENERAL] += min(0.35, general_hits * 0.12)

        if req.intent in (IntentCategory.NETWORK, IntentCategory.NETWORK_ISSUE):
            scores[AgentType.NETWORK] += 0.75
        if req.intent in (IntentCategory.DEVICE, IntentCategory.DEVICE_FAILURE):
            scores[AgentType.DEVICE] += 0.75
        network_hits = len(match_domain_keywords(msg, NETWORK_ISSUE_KEYWORDS))
        device_hits = len(match_domain_keywords(msg, DEVICE_FAILURE_KEYWORDS))
        scores[AgentType.NETWORK] += min(0.45, network_hits * 0.18)
        scores[AgentType.DEVICE] += min(0.45, device_hits * 0.18)

        entities = req.entities or {}
        if entities.get("error_code"):
            scores[AgentType.TECHNICAL] += 0.2
        if entities.get("requested_access"):
            scores[AgentType.ACCESS] += 0.15
        if entities.get("target_system"):
            scores[AgentType.GENERAL] += 0.1

        return {agent_type: round(score, 3) for agent_type, score in scores.items()}

    @staticmethod
    def _routing_reason(
        req: Request,
        scores: Dict[AgentType, float],
        primary_agent: AgentType,
        supporting_agents: List[AgentType],
    ) -> str:
        score_text = ", ".join(
            f"{agent_type.value}={score:.2f}"
            for agent_type, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )
        support_text = ", ".join(agent.value for agent in supporting_agents) or "none"
        intent = req.intent.value if req.intent else "unknown"
        return (
            f"intent={intent}, group={req.intent_group or 'unknown'}, "
            f"primary={primary_agent.value}, supporting={support_text}, scores=[{score_text}]"
        )

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"登录报错且申请项目权限"需要技术和权限 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        technical_kws = self._technical_keywords(req)
        access_kws = ["申请权限", "项目权限", "访问权限", "权限申请", "开通", "permission", "只读权限"]

        if req.intent in (
            IntentCategory.TECHNICAL,
            IntentCategory.TECHNICAL_LOGIN,
            IntentCategory.TECHNICAL_CRASH,
            IntentCategory.SOFTWARE_INSTALL,
        ) or any(kw in msg for kw in technical_kws):
            targets.append(AgentType.TECHNICAL)
        if req.intent in (
            IntentCategory.ACCOUNT,
            IntentCategory.ACCOUNT_SECURITY,
            IntentCategory.PERMISSION_REQUEST,
        ) or any(kw in msg for kw in access_kws):
            targets.append(AgentType.ACCESS)

        if req.intent in (IntentCategory.NETWORK, IntentCategory.NETWORK_ISSUE) or match_domain_keywords(
            msg, NETWORK_ISSUE_KEYWORDS
        ):
            targets.append(AgentType.NETWORK)
        if req.intent in (IntentCategory.DEVICE, IntentCategory.DEVICE_FAILURE) or match_domain_keywords(
            msg, DEVICE_FAILURE_KEYWORDS
        ):
            targets.append(AgentType.DEVICE)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    @staticmethod
    def _needs_clarification(req: Request) -> bool:
        """低置信度且无明确意图时，先追问，避免误路由。"""
        if req.intent != IntentCategory.OTHER:
            return False
        text = (req.message or "").strip()
        if len(text) <= 2:
            return False
        return req.intent_confidence < 0.5

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    def _monitor_degraded(self, agent_type: AgentType) -> bool:
        """专业 Agent 降权过高时由路由层回退，确保单实例部署也能生效。"""
        if agent_type in (AgentType.GENERAL, AgentType.ESCALATION):
            return False
        agents = self._pool.get(agent_type, [])
        if not agents:
            return False
        threshold = _env_float(
            "NEXO_MONITOR_FALLBACK_PENALTY",
            self.MONITOR_FALLBACK_PENALTY,
        )
        penalties = [
            float(getattr(getattr(agent, "stats", None), "monitor_penalty", 0.0) or 0.0)
            for agent in agents
        ]
        # 只要还有一个健康实例就继续使用该专业类型；全部高降权时才回退。
        return min(penalties) >= min(max(threshold, 0.0), 0.9)

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
        """执行 Agent，失败时降级到 GeneralAgent。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.GENERAL)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.GENERAL,
                content="服务暂时不可用，请稍后重试。",
                success=False,
            )

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 GeneralAgent
        if not response.success and agent_type not in (AgentType.GENERAL, AgentType.ESCALATION):
            logger.warning(f"{agent_type.value} 失败，降级到 GeneralAgent")
            fallback = self._best_agent(AgentType.GENERAL)
            if fallback:
                previous = response
                from harness.context import harness_mode
                from harness.runtime import evidence_ids
                from harness.state import apply_progress, new_case
                mode = req.harness_mode or harness_mode()
                fallback_req = req
                if mode == "enforce":
                    state = self._merge_case_state(req, previous.state_updates,
                                                   previous.tool_traces, previous.harness_events)
                    evidence = [item for call in previous.tool_traces
                                if call.get("tool_name") == "search_knowledge_base"
                                and call.get("success") and call.get("result_success") is not False
                                for item in (call.get("retrieval") or {}).get("evidence", [])]
                    context = req.context
                    if evidence:
                        context += "\n\n[本次处理已经返回的知识库资料]\n" + json.dumps(evidence, ensure_ascii=False)
                    fallback_req = replace(req, case_state=state, context=context)
                response = await fallback.handle(fallback_req)
                if mode == "enforce":
                    response.tools_used = list(dict.fromkeys(previous.tools_used + response.tools_used))
                    response.tool_traces = previous.tool_traces + response.tool_traces
                    response.state_updates = previous.state_updates + response.state_updates
                if mode != "off":
                    response.harness_events = previous.harness_events + [
                        {"event": "agent_fallback", "from": agent_type.value, "to": "general"}
                    ] + response.harness_events

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "role": agent.profile.role,
                    "workflow": list(agent.profile.workflow),
                    "tool_scope": list(agent.profile.tool_scope),
                    "available_tools": list(agent.get_tools()),
                    "model": agent._model,
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
