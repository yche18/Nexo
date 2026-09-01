"""Per-conversation IT case state, separate from chat summaries and routing entities."""
import copy
import hashlib
import json
import re
import uuid


FACT_FIELDS = (
    "target_system", "issue_description", "date", "error_code",
    "requested_access", "purpose", "operating_system", "client_version",
    "network_type", "impact_scope", "device_type", "device_model",
)
ASSET_FIELDS = ("asset_id", "device_type", "device_model", "operating_system",
                "configuration", "warranty_until")
STATUSES = ("collecting", "waiting_user", "handoff_ready", "resolved")
CORRECTION = re.compile(r"说错|更正|改为|改成|不是.+是|实际是|纠正")
NEW_TOPIC = re.compile(r"^\s*(?:(?:好[的吧]?|那|现在|我们|我想问|我要问|我想)[，,\s]*)?(?:换个问题|换一个问题|另一个问题|新的问题|重新开始)")
PROGRESS_DESCRIPTION = (
    "记录当前 IT 问题的处理进度。只记录本轮用户明确提供的事实和操作结果，"
    "事实值尽量使用原话；source_quote 必须逐字引用本轮用户消息，user_reported_done 步骤引用应包含动作及结果。"
    "result_quote 可单独引用同一句中的结果原文，结果未知可留空；建议执行不代表已经执行。"
    "已保存的事实不需重复提交，不得用历史消息冒充本轮来源。"
    "steps 只记用户实际操作或待执行建议，不记 Agent 自己的工具调用。"
    "proposed 步骤只填 step_id、description、status，不带 source_quote、result 或 result_quote。"
    "资产工具结果由后端独立保存，不要作为用户事实重复记录，也不能提交 asset_lookup。"
    "工单回执也由后端独立保存，不得提交 ticket_receipt。"
    "evidence_refs 只填检索实际返回的 chunk_id，不拼接标题。"
    "只在用户明确更正时设置 correction，明确换问题时设置 new_case。"
    "不要记录密码、验证码或密钥。此工具不创建工单、不执行授权或设备操作。"
)
PROGRESS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "facts": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "field": {"type": "string", "enum": list(FACT_FIELDS),
                          "description": "network_type 为故障网络环境，impact_scope 为影响范围，device_type/device_model 为设备种类和型号；只记录用户明确提供的信息。"},
                "value": {"type": "string"}, "source_quote": {"type": "string"},
                "correction": {"type": "boolean"},
            }, "required": ["field", "value", "source_quote"]}},
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "step_id": {"type": "string"}, "description": {"type": "string"},
                "status": {"type": "string", "enum": ["proposed", "user_reported_done"]},
                "result": {"type": "string"}, "source_quote": {"type": "string"},
                "result_quote": {"type": "string"},
            }, "required": ["step_id", "description", "status"]}},
        "pending_questions": {"type": "array", "items": {"type": "string"}},
        "next_action": {"type": "string"},
        "status": {"type": "string", "enum": list(STATUSES)},
        "status_quote": {"type": "string"},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "new_case": {"type": "boolean"},
    },
}


def new_case(user_id, conv_id):
    return {
        "case_id": uuid.uuid4().hex[:12], "user_id": user_id, "conv_id": conv_id,
        "version": 0, "facts": {}, "steps": [], "evidence_refs": [],
        "pending_questions": [], "next_action": "", "status": "collecting",
        "conflicts": [], "candidates": [], "progress_warnings": [], "asset_lookup": {}, "ticket_receipt": {},
    }


def prepare_case(case, user_message):
    """Explicit new-topic markers clear active facts before even direct handoff.

    Preserve the slot version for CAS; this is not an intent classifier.
    """
    if not NEW_TOPIC.search(user_message):
        continued = copy.deepcopy(case)
        continued.pop("new_topic", None)
        asset_lookup = continued.get("asset_lookup") or {}
        asset_id = str(asset_lookup.get("asset_id") or "").upper()
        mentioned_ids = re.findall(r"(?<![A-Za-z0-9-])(?:NB|PC|PR)-\d+(?![A-Za-z0-9-])", user_message, re.I)
        # Explicitly supplied unknown identifiers also invalidate the previous device.
        mentioned_ids += re.findall(
            r"资产编号\s*(?:[:：=]|更正为|修改为|调整为|改为|改成|是|为)\s*([A-Za-z0-9][A-Za-z0-9._-]*)",
            user_message,
        )
        if asset_id and any(value.upper() != asset_id for value in mentioned_ids):
            continued["asset_lookup"] = {}
        return continued
    replacement = new_case(case["user_id"], case["conv_id"])
    replacement["version"] = case.get("version", 0)
    replacement["new_topic"] = True
    replacement["isolate_history"] = True
    return replacement


def apply_asset_lookup(case, trace, request_id):
    """Copy only backend lookup results into a separate demo-asset snapshot."""
    result = copy.deepcopy(case)
    if not isinstance(trace, dict) or trace.get("tool_name") != "lookup_asset":
        return result
    inputs = trace.get("input") or {}
    value = inputs.get("asset_id") if isinstance(inputs, dict) else None
    asset_id = value.strip().upper() if isinstance(value, str) else ""
    snapshot = {"asset_id": asset_id, "source": "demo", "status": "unavailable",
                "asset": {}, "request_id": request_id}
    # Every lookup attempt replaces the old asset, including schema/service failures.
    result["asset_lookup"] = snapshot
    payload = trace.get("asset_lookup")
    if not isinstance(payload, dict):
        return result
    if (trace.get("success") is not True or trace.get("result_success") is not True
            or payload.get("success") is not True or payload.get("source") != "demo"):
        return result
    returned_id = payload.get("asset_id")
    if not asset_id or not isinstance(returned_id, str) or returned_id.strip().upper() != asset_id:
        return result
    if payload.get("found") is False:
        snapshot["status"] = "not_found"
        return result
    asset = payload.get("asset")
    if payload.get("found") is not True or not isinstance(asset, dict):
        return result
    if any(not isinstance(asset.get(field), str) or not asset[field].strip() for field in ASSET_FIELDS):
        return result
    if asset["asset_id"].strip().upper() != asset_id:
        return result
    snapshot["status"] = "found"
    snapshot["asset"] = {field: asset[field] for field in ASSET_FIELDS}
    snapshot["asset"]["asset_id"] = asset_id
    return result


def _text(value, name, limit=400, required=True):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f"{name} 必须是非空字符串且不超过 {limit} 字")
    return value.strip()


def _quoted(value, message):
    quote = _text(value, "source_quote")
    if quote not in message:
        raise ValueError("来源必须逐字引用本轮用户消息")
    return quote


def _items(value, name, limit=12):
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{name} 必须是数组且最多 {limit} 项")
    return value


def _source_context(quote, message):
    """Keep the sentence around a literal quote, including omitted qualifiers."""
    start = message.index(quote)
    end = start + len(quote)
    left = re.search(r"[^。！？!?\n]*$", message[:start]).group()
    right = re.match(r"[^。！？!?\n]*", message[end:]).group()
    return left + quote + right


def _completed_action(quote, context, description):
    # A later failed outcome is different from negating the action itself.
    if re.search(r"如果|假如|要是", context):
        raise ValueError("条件描述不能标记为已执行")
    actions = r"重启|登录|登陆|切换|更换|换|执行|操作|尝试|试|检查|关闭|打开|安装|卸载|更新|重置|清理|断开|连接|重连|禁用|启用|取纸|取出卡纸|清除卡纸|拔下电源|拔掉电源|拔电源|断电|插拔"
    matches = list(re.finditer(actions, quote))
    selected = [match for match in matches if match.group() in description]
    if selected:
        matches = selected
    completed = False
    offset = context.index(quote)
    for match in matches:
        position = offset + match.start()
        prefix = re.split(r"[，,；;]|但是|但", context[:position])[-1]
        suffix = re.split(r"[，,；;]|但是|但", context[position:])[0]
        if re.search(r"没(?:有)?|未曾|尚未|还未|并未|不曾|准备|打算|建议|请先|是否|要不要", prefix):
            raise ValueError("计划、询问或未执行描述不能标记为已执行")
        if re.search(r"吗|么|是不是|有没有|是否|怎么|如何", prefix + suffix):
            raise ValueError("询问操作不能标记为已执行")
        if re.search(r"已经|已|刚刚|刚", prefix) or re.search(r"^.{0,16}(?:了|过|后)", suffix):
            completed = True
    return completed


def _step_result(step, quote, context, message):
    """Only report a literal outcome; never substitute an action for a result."""
    outcome = _text(step.get("result", ""), "result", 200, required=False)
    explicit = step.get("result_quote", "")
    if explicit:
        explicit = _quoted(explicit, message)
        if explicit not in context:
            raise ValueError("结果引用必须来自步骤来源的同一句用户消息")
    # This is deliberately conservative. Unrecognized results remain unknown.
    outcome_pattern = r"超时|报错|错误|失败|成功|恢复|正常|异常|无效|有效|不行|不通|没变化|没有变化|无变化|没改善|没有改善|仍|还是|依旧|依然|不能|无法|可以|能够"
    result_text = explicit or outcome
    if result_text and result_text in context and re.search(outcome_pattern, result_text):
        position = context.index(result_text)
        prefix = re.split(r"[，,；;]", context[:position])[-1]
        clause = prefix + re.split(r"[，,；;]", context[position:])[0]
        if re.search(r"如果|假如|要是|是否|吗|么|怎么|如何", clause):
            return ""
        if re.search(r"没|未|不|仍|还是|依旧|依然", prefix):
            return clause
        return result_text
    reports = [clause for clause in re.split(r"[，,；;]", context)
               if not re.search(r"如果|假如|要是|是否|吗|么|怎么|如何", clause)]
    if any(re.search(outcome_pattern, clause) for clause in reports):
        # Preserve all qualifiers rather than invent a normalized outcome.
        return context
    return ""


def _apply_fact(result, fact, user_message, request_id):
    if not isinstance(fact, dict) or set(fact) - {"field", "value", "source_quote", "correction"}:
        raise ValueError("事实格式错误")
    field = fact.get("field")
    if field not in FACT_FIELDS:
        raise ValueError("不支持的事实字段")
    value = _text(fact.get("value"), "value", 200)
    quote = _quoted(fact.get("source_quote"), user_message)
    model_label = ""
    if value not in quote and field == "issue_description":
        model_label = value
        value = quote
    if value not in quote:
        # Preserve the source and model interpretation without promoting it to
        # a fact. A paraphrase must not roll back other source-backed items.
        candidate = {"field": field, "value": value, "source": "model_interpretation",
                     "source_quote": quote, "request_id": request_id}
        if candidate not in result["candidates"]:
            result["candidates"].append(candidate)
        result["candidates"] = result["candidates"][-12:]
        result["progress_warnings"].append({"kind": "unverified_paraphrase", "field": field,
            "reason": "改写值仅作为待核验候选，未覆盖已确认字段"})
        return
    correction = fact.get("correction", False)
    if not isinstance(correction, bool):
        raise ValueError("correction 必须是布尔值")
    source_context = _source_context(quote, user_message)
    explicit_correction = bool(CORRECTION.search(source_context))
    if correction and not explicit_correction:
        raise ValueError("更正需要引用明确的更正原文")
    correction = correction or explicit_correction
    entry = {"value": value, "source": "user_reported", "source_quote": quote,
             "request_id": request_id}
    if correction:
        entry["correction_context"] = source_context
    if model_label:
        entry["model_label"] = model_label
    old = result["facts"].get(field)
    if field == "issue_description" and old:
        # A free-text problem report evolves as users report new observations.
        history = list(old.get("previous_reports", []))
        if old["value"] != value:
            history.append({"value": old["value"], "source_quote": old["source_quote"],
                            "request_id": old["request_id"]})
        entry["previous_reports"] = history[-4:]
    if field != "issue_description" and old and old["value"] != value and not correction:
        conflict = {"field": field, "previous": old["value"], **entry}
        if conflict not in result["conflicts"]:
            result["conflicts"].append(conflict)
        result["conflicts"] = result["conflicts"][-12:]
        return
    result["facts"][field] = entry
    if correction:
        result["conflicts"] = [item for item in result["conflicts"] if item["field"] != field]


def _apply_step(result, step, user_message, request_id):
    if not isinstance(step, dict) or set(step) - {"step_id", "description", "status", "result", "source_quote", "result_quote"}:
        raise ValueError("步骤格式错误")
    step_id = _text(step.get("step_id"), "step_id", 80)
    description = _text(step.get("description"), "description", 200)
    status = step.get("status")
    if status not in ("proposed", "user_reported_done"):
        raise ValueError("不能把模型建议标记为已验证执行")
    entry = {"step_id": step_id, "description": description, "status": status,
             "request_id": request_id}
    if status == "user_reported_done":
        quote = _quoted(step.get("source_quote"), user_message)
        source_context = _source_context(quote, user_message)
        completed = _completed_action(quote, source_context, description)
        outcome = _step_result(step, quote, source_context, user_message)
        # Model descriptions and results are labels, never factual evidence.
        if description not in quote:
            entry["model_label"] = description
            entry["description"] = quote
        if step.get("result") and step["result"] != outcome:
            entry["model_result_label"] = step["result"]
        if not completed:
            entry["status"] = "user_reported_observation"
            result["progress_warnings"].append({"kind": "observation_only", "step_id": step_id,
                "reason": "用户陈述的是现象，未记录为已执行操作"})
        if not outcome:
            result["progress_warnings"].append({"kind": "unknown_step_result", "step_id": step_id,
                "reason": "原文未明确报告操作结果，结果留空，不推断成功或失败"})
        entry.update(source="user_reported", source_quote=quote,
                     source_context=source_context, result=outcome)
    elif step.get("result") or step.get("source_quote") or step.get("result_quote"):
        raise ValueError("建议步骤不能携带已经完成的结果")
    old = next((item for item in result["steps"] if item["step_id"] == step_id), None)
    if old and old["status"] == "user_reported_done" and entry["status"] != "user_reported_done":
        return
    result["steps"] = [item for item in result["steps"] if item["step_id"] != step_id]
    result["steps"].append(entry)
    result["steps"] = result["steps"][-24:]


def apply_progress(case, update, user_message, request_id, evidence_refs=None):
    """Validate a staged update. Return a copy; never mutate shared request state.

    Quoted information is user-reported, not independently verified. Text matching
    establishes provenance only; it does not prove semantic correctness.
    """
    if not isinstance(update, dict) or set(update) - set(PROGRESS_SCHEMA["properties"]):
        raise ValueError("进度参数包含未知字段")
    result = copy.deepcopy(case)
    result["progress_warnings"] = []
    result.setdefault("candidates", [])
    if "new_case" in update and not isinstance(update["new_case"], bool):
        raise ValueError("new_case 必须是布尔值")
    if update.get("new_case"):
        if not NEW_TOPIC.search(user_message):
            raise ValueError("用户未明确要求换问题，不能清空已有处理进度")
        result = new_case(case["user_id"], case["conv_id"])
        result["version"] = case.get("version", 0)
        result["new_topic"] = True
        result["isolate_history"] = True
    for fact in _items(update.get("facts", []), "facts"):
        try:
            _apply_fact(result, fact, user_message, request_id)
        except ValueError as exc:
            result["progress_warnings"].append({"kind": "invalid_fact",
                "field": fact.get("field") if isinstance(fact, dict) else None,
                "reason": str(exc)})
    for step in _items(update.get("steps", []), "steps"):
        try:
            _apply_step(result, step, user_message, request_id)
        except ValueError as exc:
            result["progress_warnings"].append({"kind": "invalid_step",
                "step_id": step.get("step_id") if isinstance(step, dict) else None,
                "reason": str(exc)})
    if "pending_questions" in update:
        result["pending_questions"] = [_text(item, "pending_question", 200)
                                       for item in _items(update["pending_questions"], "pending_questions")]
    if "next_action" in update:
        result["next_action"] = _text(update["next_action"], "next_action", 300, required=False)
    for ref in _items(update.get("evidence_refs", []), "evidence_refs"):
        ref = _text(ref, "evidence_ref", 160)
        if ref not in (evidence_refs or []):
            result["progress_warnings"].append({"kind": "invalid_evidence_ref", "ref": ref,
                "reason": "引用不在本次工具返回的 chunk_id 中，已丢弃；其他合法进度继续保存"})
            continue
        if ref not in result["evidence_refs"]:
            result["evidence_refs"].append(ref)
    result["evidence_refs"] = result["evidence_refs"][-24:]
    if "status" in update:
        status = update["status"]
        if status not in STATUSES:
            raise ValueError("无效处理状态")
        if status == "resolved":
            quote = _quoted(update.get("status_quote"), user_message)
            if not re.search(r"问题解决了|已经恢复正常|已恢复正常|现在正常了|可以正常使用了", quote):
                raise ValueError("只有用户明确确认恢复，才能标记问题已解决")
            confirmation = _source_context(quote, user_message)
            end = user_message.index(confirmation) + len(confirmation)
            confirmation += user_message[end:end + 1]
            if re.search(r"没有|还没|并未|尚未|不是|如果|是否|吗|？|\?", confirmation):
                raise ValueError("用户没有明确确认问题已解决")
            result["resolution_source"] = {"source": "user_reported", "source_quote": quote,
                                           "request_id": request_id}
        result["status"] = status
    if result["conflicts"]:
        result["status"] = "collecting"
    return result


def handoff_fields(case):
    """Only compact business fields, excluding unresolved conflicting values."""
    if not case:
        return {}
    conflicts = {item["field"] for item in case.get("conflicts", [])}
    return {key: item["value"] for key, item in case.get("facts", {}).items()
            if key in FACT_FIELDS and key not in conflicts and item.get("source") == "user_reported"}


class CaseConflictError(RuntimeError):
    pass


class CaseStore:
    """Redis snapshot with optimistic concurrency, isolated by user and conversation."""
    def __init__(self, redis_client, ttl_seconds=86400):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def key(user_id, conv_id):
        identity = json.dumps([user_id, conv_id], ensure_ascii=False).encode("utf-8")
        return "nexo:case:" + hashlib.sha256(identity).hexdigest()

    async def load(self, user_id, conv_id):
        raw = await self.redis.get(self.key(user_id, conv_id))
        return json.loads(raw) if raw else new_case(user_id, conv_id)

    async def save(self, case, expected_version):
        from redis.exceptions import WatchError
        key = self.key(case["user_id"], case["conv_id"])
        snapshot = copy.deepcopy(case)
        snapshot["version"] = expected_version + 1
        try:
            async with self.redis.pipeline(transaction=True) as pipe:
                await pipe.watch(key)
                raw = await pipe.get(key)
                actual = json.loads(raw).get("version", 0) if raw else 0
                if actual != expected_version:
                    raise CaseConflictError("同一会话的处理进度已更新，请重试本轮消息")
                pipe.multi()
                pipe.set(key, json.dumps(snapshot, ensure_ascii=False), ex=self.ttl_seconds)
                await pipe.execute()
        except WatchError as exc:
            raise CaseConflictError("同一会话正在更新，请重试本轮消息") from exc
        return snapshot
