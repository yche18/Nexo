"""Shared request-local Agent loop; business routing remains in the orchestrator."""
import copy
import os
import inspect
import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.llm_utils import extract_text_content, thinking_request_kwargs

logger = logging.getLogger(__name__)


async def run_agent(agent, req):
    from agents.agent_orchestrator import AgentCallResult, AgentCallError
    from harness.context import build_messages, harness_mode
    from harness.state import PROGRESS_SCHEMA, PROGRESS_DESCRIPTION, apply_progress, apply_asset_lookup, new_case
    from agents.tools import AgentToolSpec

    mode = req.harness_mode or harness_mode()
    messages = build_messages(req, agent._build_role_packet(req))
    events = []
    updates = []
    local_state = copy.deepcopy(req.case_state) if req.case_state else new_case(req.user_id, req.conv_id)

    def event(kind, **details):
        if mode != "off":
            events.append({"event": kind, "agent_type": agent.agent_type.value, **details})

    def record_progress(request, args):
        nonlocal local_state
        local_state = apply_progress(local_state, args, request.message, request.request_id,
                                     evidence_refs=evidence_ids(tool_traces))
        updates.append(copy.deepcopy(args))
        event("state_update_staged", status=local_state.get("status"))
        warnings = local_state.get("progress_warnings", [])
        return {"success": True, "status": local_state.get("status"), "staged": True,
                "warnings": warnings,
                "next_action": "已处理本次进度记录；请查看警告，未核实条目未写入确认状态，无需重复提交，继续回答或澄清。" if warnings else "继续处理并回答用户"}

    tools = agent.get_tools()
    if mode == "enforce":
        tools["record_support_progress"] = AgentToolSpec(
            name="record_support_progress", description=PROGRESS_DESCRIPTION,
            input_schema=PROGRESS_SCHEMA, handler=record_progress,
        )
    tools_used: List[str] = []
    tool_traces: List[Dict[str, Any]] = []
    event("run_started", mode=mode)
    tool_count = 0
    try:
        for turn in range(3):
            event("model_started", round=turn + 1)
            request_kwargs: Dict[str, Any] = {
                "model": agent._model,
                "max_tokens": agent.profile.max_tokens,
                "temperature": agent.profile.temperature,
                "system": agent._build_system_prompt(req),
                "messages": messages,
            }
            if mode == "enforce":
                request_kwargs["system"] += (
                    "\n[任务进度] 多轮排障、权限信息收集时，使用 record_support_progress 记录本轮有原文依据的事实和步骤。"
                    "建议执行不等于用户已经执行；工具信息只表明其实际返回结果。无需为问候调用进度工具。"
                )
            request_kwargs.update(thinking_request_kwargs())
            if tools:
                request_kwargs["tools"] = [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "input_schema": spec.input_schema,
                    }
                    for spec in tools.values()
                ]
            model_started = time.monotonic()
            resp = await agent._client.messages.create(**request_kwargs)
            model_latency_ms = (time.monotonic() - model_started) * 1000
            usage = getattr(resp, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump(mode="json")
            elif not isinstance(usage, dict):
                usage = {key: getattr(usage, key) for key in (
                    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"
                ) if getattr(usage, key, None) is not None}
            event("model_finished", round=turn + 1, stop_reason=getattr(resp, "stop_reason", None),
                  latency_ms=round(model_latency_ms, 1), usage=usage)
            tool_uses = [block for block in (resp.content or []) if agent._block_type(block) == "tool_use"]
            if not tool_uses:
                intent = getattr(req.intent, "value", "")
                business_intent = intent.startswith(("technical", "access", "permission", "account", "network", "device", "software"))
                business_agent = agent.agent_type.value in {"technical", "access", "network", "device"}
                if mode == "enforce" and (business_intent or business_agent) and not updates:
                    event("progress_not_recorded", reason="no_valid_progress_update", status="unknown")
                event("run_finished", reason="answer_ready", rounds=turn + 1)
                return AgentCallResult(
                    content=extract_text_content(resp.content),
                    tools_used=list(tools_used),
                    tool_traces=list(tool_traces),
                    harness_events=list(events), state_updates=list(updates),
                )

            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in tool_uses:
                name = agent._block_value(block, "name")
                tool_use_id = agent._block_value(block, "id")
                args = agent._block_value(block, "input") or {}
                spec = tools.get(name)
                tool_t0 = time.monotonic()
                call_success = True
                result_success: Optional[bool] = None
                error_text = ""
                tool_executed = False
                if spec is None:
                    call_success = False
                    result: Any = {"success": False, "error": f"工具不在 {agent.agent_type.value} Agent 白名单中"}
                    error_text = result["error"]
                else:
                    try:
                        agent._validate_tool_input(spec, args)
                        if name == "build_diagnostic_plan":
                            from harness.validation import applicable_vpn_evidence
                            if applicable_vpn_evidence(req.message, tool_traces, local_state):
                                event("tool_policy", tool_name=name, reason="applicable_vpn_sop", mode=mode)
                                if mode == "enforce":
                                    raise ValueError("已有适用 VPN SOP，请依据已返回的知识库步骤继续，不要用通用诊断清单替代")
                        tool_count += 1
                        if mode == "enforce" and tool_count > tool_limit():
                            event("tool_blocked", tool_name=name, reason="tool_budget_exhausted")
                            raise ValueError("本轮工具调用预算已用完，请整理已有结果或说明待确认的信息")
                        tool_executed = True
                        result = spec.handler(req, args)
                        if inspect.isawaitable(result):
                            result = await result
                        tools_used.append(name)
                        if isinstance(result, dict) and "success" in result:
                            result_success = bool(result.get("success"))
                    except Exception as ex:
                        call_success = False
                        logger.warning("Agent 工具 %s 执行失败: %s", name, ex)
                        error_text = str(ex)
                        result = {"success": False, "error": error_text}
                        if mode == "enforce" and name == "record_support_progress":
                            tools.pop(name, None)
                            result["next_action"] = "本次未记录该更新。不要重试进度工具；请根据已知信息继续回答或向用户澄清。"
                            event("progress_tool_retired", reason="invalid_update", error=error_text)
                tool_latency_ms = (time.monotonic() - tool_t0) * 1000
                if not error_text and isinstance(result, dict):
                    error_text = str(result.get("error", "") or "")
                tool_traces.append(
                    {
                        "agent_type": agent.agent_type.value,
                        "tool_name": name,
                        "tool_use_id": tool_use_id,
                        "input": dict(args),
                        "success": call_success,
                        "result_success": result_success,
                        "latency_ms": round(tool_latency_ms, 1),
                        "cached": bool(result.get("cached")) if isinstance(result, dict) else False,
                        "reranked": bool(result.get("reranked")) if isinstance(result, dict) else False,
                        "retrieval": result.get("retrieval", {}) if isinstance(result, dict) else {},
                        "error": error_text,
                    }
                )
                if mode == "enforce" and name == "lookup_asset":
                    trace = tool_traces[-1]
                    if tool_executed and isinstance(result, dict):
                        snapshot = {key: copy.deepcopy(result[key]) for key in
                                    ("success", "source", "asset_id", "found") if key in result}
                        if isinstance(result.get("asset"), dict):
                            snapshot["asset"] = {key: copy.deepcopy(result["asset"][key]) for key in
                                                 ("asset_id", "device_type", "device_model", "operating_system",
                                                  "configuration", "warranty_until") if key in result["asset"]}
                        trace["asset_lookup"] = snapshot
                    local_state = apply_asset_lookup(local_state, trace, req.request_id)
                event("tool_finished", tool_name=name, success=call_success,
                      result_success=result_success, error=error_text)
                # 完整证据已在 results 中；trace 保留证据，模型只接收一份。
                if isinstance(result, dict) and result.get("retrieval"):
                    result = dict(result)
                    result["retrieval"] = {
                        key: value for key, value in result["retrieval"].items()
                        if key != "evidence"
                    }
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            messages.append({"role": "user", "content": tool_results})

        if mode == "enforce":
            # 工具阶段仍只有三轮；额外一次只整理已有结果，不允许继续执行工具。
            finish_kwargs = dict(request_kwargs)
            finish_kwargs["messages"] = messages
            finish_kwargs["system"] += (
                "\n[回答收尾] 工具阶段已结束。本次只根据已有上下文和工具结果回答用户，不再请求任何工具。"
                "说明已确认的信息、可执行的下一步和仍待核实的内容；不能把失败的工具操作说成成功。"
            )
            # 当前 SDK 0.40 未声明 none 类型，通过原有 extra_body 透传，保留历史工具定义。
            finish_kwargs["extra_body"] = dict(finish_kwargs.get("extra_body") or {})
            finish_kwargs["extra_body"]["tool_choice"] = {"type": "none"}
            finish_client = agent._client.with_options(max_retries=0) if hasattr(agent._client, "with_options") else agent._client
            event("answer_finish_started", max_extra_calls=1)
            event("model_started", round=4, phase="answer_finish")
            model_started = time.monotonic()
            resp = await finish_client.messages.create(**finish_kwargs)
            model_latency_ms = (time.monotonic() - model_started) * 1000
            usage = getattr(resp, "usage", None)
            if hasattr(usage, "model_dump"):
                usage = usage.model_dump(mode="json")
            elif not isinstance(usage, dict):
                usage = {key: getattr(usage, key) for key in (
                    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"
                ) if getattr(usage, key, None) is not None}
            event("model_finished", round=4, phase="answer_finish", stop_reason=getattr(resp, "stop_reason", None),
                  latency_ms=round(model_latency_ms, 1), usage=usage)
            content = extract_text_content(resp.content)
            has_tool_use = any(agent._block_type(block) == "tool_use" for block in (resp.content or []))
            truncated = getattr(resp, "stop_reason", None) == "max_tokens"
            if has_tool_use or truncated or not content.strip():
                reason = "tool_use" if has_tool_use else "truncated" if truncated else "empty"
                event("answer_finish_rejected", reason=reason)
                event("run_finished", reason="model_round_budget_exhausted", rounds=4)
                raise AgentCallError("回答收尾未生成有效最终文本", tools_used, tool_traces)
            if not updates:
                event("progress_not_recorded", reason="no_valid_progress_update", status="unknown")
            event("run_finished", reason="answer_ready", rounds=4, phase="answer_finish")
            return AgentCallResult(
                content=content, tools_used=list(tools_used), tool_traces=list(tool_traces),
                harness_events=list(events), state_updates=list(updates),
            )

        event("run_finished", reason="model_round_budget_exhausted", rounds=3)
        raise AgentCallError(
            f"{agent.agent_type.value} 工具调用超过最大轮数",
            tools_used,
            tool_traces,
        )
    except AgentCallError as ex:
        ex.harness_events = list(events)
        ex.state_updates = list(updates)
        raise
    except Exception as ex:
        event("run_finished", reason="model_or_runtime_error", error=type(ex).__name__)
        error = AgentCallError(str(ex), tools_used, tool_traces)
        error.harness_events = list(events)
        error.state_updates = list(updates)
        raise error from ex



def tool_limit():
    """Per-Agent tool cap; enforce may add one tool-free answer after three rounds."""
    try:
        return max(1, int(os.getenv("NEXO_HARNESS_MAX_TOOL_CALLS", "12")))
    except ValueError:
        return 12


def evidence_ids(tool_traces):
    refs = []
    for trace in tool_traces:
        if trace.get("tool_name") != "search_knowledge_base" or not trace.get("success"):
            continue
        if trace.get("result_success") is False:
            continue
        for item in trace.get("retrieval", {}).get("evidence", []):
            if isinstance(item, dict):
                value = item.get("id") or item.get("chunk_id") or item.get("document_id")
                if value and str(value) not in refs:
                    refs.append(str(value))
    return refs
