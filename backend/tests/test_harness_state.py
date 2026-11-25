import asyncio
import copy
import json
import sys
from types import SimpleNamespace

import pytest

from harness.context import build_messages, case_context, harness_mode
from harness.state import (
    CaseConflictError, CaseStore, apply_progress, handoff_fields, new_case, prepare_case,
)


def test_facts_have_user_provenance_without_mutating_input():
    original = new_case("u1", "c1")
    value = apply_progress(original, {"facts": [{"field": "target_system", "value": "SAP",
        "source_quote": "目标系统是 SAP"}]}, "目标系统是 SAP", "r1")
    assert original["facts"] == {}
    assert value["facts"]["target_system"]["source"] == "user_reported"
    assert value["facts"]["target_system"]["request_id"] == "r1"


@pytest.mark.parametrize("fact,message", [
    ({"field": "target_system", "value": "SAP", "source_quote": "目标系统是 SAP"}, "你好"),
    ({"field": "password", "value": "123", "source_quote": "密码123"}, "密码123"),
])
def test_unsubstantiated_facts_are_rejected(fact, message):
    rejected = apply_progress(new_case("u", "c"), {"facts": [fact]}, message, "r")
    assert rejected["facts"] == {}
    assert rejected["progress_warnings"][0]["kind"] == "invalid_fact"


def test_conflict_keeps_previous_value_until_explicit_correction():
    initial = apply_progress(new_case("u", "c"), {"facts": [
        {"field": "target_system", "value": "SAP", "source_quote": "SAP"}]}, "SAP", "r1")
    conflict = apply_progress(initial, {"facts": [
        {"field": "target_system", "value": "OA", "source_quote": "OA"}]}, "OA", "r2")
    assert conflict["facts"]["target_system"]["value"] == "SAP"
    assert conflict["conflicts"][0]["value"] == "OA"
    assert "target_system" not in handoff_fields(conflict)
    corrected = apply_progress(conflict, {"facts": [
        {"field": "target_system", "value": "OA", "source_quote": "说错了，是 OA", "correction": True}]},
        "说错了，是 OA", "r3")
    assert corrected["facts"]["target_system"]["value"] == "OA"
    assert corrected["conflicts"] == []


def test_suggestion_is_not_execution_and_user_result_is_preserved():
    initial = new_case("u", "c")
    proposed = apply_progress(initial, {"steps": [
        {"step_id": "restart", "description": "重启客户端", "status": "proposed"}]}, "VPN 超时", "r1")
    done = apply_progress(proposed, {"steps": [{"step_id": "restart", "description": "重启客户端",
        "status": "user_reported_done", "result": "仍然超时", "source_quote": "重启客户端后仍然超时"}]},
        "重启客户端后仍然超时", "r2")
    assert done["steps"][0]["result"] == "仍然超时"
    again = apply_progress(done, {"steps": [
        {"step_id": "restart", "description": "重启客户端", "status": "proposed"}]}, "接下来呢", "r3")
    assert again["steps"][0]["status"] == "user_reported_done"


@pytest.mark.parametrize("message", ["如果重启客户端后仍然超时怎么办", "准备重启客户端后看看仍然超时吗"])
def test_hypothetical_or_planned_steps_cannot_be_recorded_as_executed(message):
    rejected = apply_progress(new_case("u", "c"), {"steps": [{"step_id": "restart", "description": "重启客户端",
        "status": "user_reported_done", "result": "仍然超时", "source_quote": message}]}, message, "r")
    assert rejected["steps"] == []
    assert rejected["progress_warnings"][0]["kind"] == "invalid_step"


@pytest.mark.parametrize("message", ["问题解决了吗", "如果问题解决了", "不是问题解决了", "没有恢复"])
def test_unconfirmed_resolution_rejected(message):
    with pytest.raises(ValueError):
        apply_progress(new_case("u", "c"), {"status": "resolved", "status_quote": message}, message, "r")


def test_resolution_requires_explicit_user_confirmation():
    case = apply_progress(new_case("u", "c"), {"status": "resolved", "status_quote": "问题解决了"}, "问题解决了", "r")
    assert case["status"] == "resolved"
    assert case["resolution_source"]["source"] == "user_reported"


def test_evidence_must_have_been_returned_by_tools():
    rejected = apply_progress(new_case("u", "c"), {"evidence_refs": ["invented"]}, "VPN", "r", ["actual"])
    assert rejected["evidence_refs"] == []
    assert rejected["progress_warnings"][0]["kind"] == "invalid_evidence_ref"
    case = apply_progress(new_case("u", "c"), {"evidence_refs": ["actual"]}, "VPN", "r", ["actual"])
    assert case["evidence_refs"] == ["actual"]


def test_explicit_new_topic_clears_case_before_direct_handoff():
    old = new_case("u", "c")
    old["version"] = 7
    old["facts"]["target_system"] = {"value": "SAP", "source": "user_reported"}
    fresh = prepare_case(old, "换个问题，帮我转人工")
    assert fresh["case_id"] != old["case_id"]
    assert fresh["version"] == 7
    assert fresh["facts"] == {}
    assert fresh["new_topic"]
    assert prepare_case(old, "继续") == old


@pytest.mark.parametrize("mode", ["off", "observe"])
def test_baseline_prompt_identity_without_case_injection(monkeypatch, mode):
    monkeypatch.setenv("NEXO_HARNESS_MODE", "enforce")
    req = SimpleNamespace(context="旧上下文", entities={"target_system": ["SAP"]},
                          message="你好", harness_mode=mode, case_state=new_case("u", "c"))
    messages = build_messages(req, "角色包")
    assert messages == [
        {"role": "user", "content": "[背景信息]\n旧上下文"},
        {"role": "assistant", "content": "好的，我已了解背景信息。"},
        {"role": "user", "content": '[结构化实体]\n{"target_system": ["SAP"]}'},
        {"role": "assistant", "content": "好的，我会结合这些结构化实体处理。"},
        {"role": "user", "content": "[角色输入契约]\n角色包"},
        {"role": "assistant", "content": "好的，我会按照该角色的输入和输出契约处理。"},
        {"role": "user", "content": "你好"},
    ]


def test_case_context_is_independent_from_chat_summary_and_honors_request_mode(monkeypatch):
    monkeypatch.setenv("NEXO_HARNESS_MODE", "off")
    case = new_case("user-secret", "conv-secret")
    case["facts"]["target_system"] = {"value": "SAP", "source": "user_reported"}
    req = SimpleNamespace(context="已压缩摘要", entities={}, message="继续", harness_mode="enforce", case_state=case)
    text = json.dumps(build_messages(req, ""), ensure_ascii=False)
    assert "SAP" in text
    assert "user-secret" not in text and "conv-secret" not in text
    assert req.entities == {}


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.expiries = {}
        self.race = False
    async def get(self, key):
        return self.data.get(key)
    def pipeline(self, transaction=True):
        owner = self
        class Pipeline:
            async def __aenter__(self): return self
            async def __aexit__(self, *args): return False
            async def watch(self, key): self.key = key
            async def get(self, key): return owner.data.get(key)
            def multi(self): pass
            def set(self, key, value, ex): self.pending = (key, value, ex)
            async def execute(self):
                if owner.race:
                    from redis.exceptions import WatchError
                    raise WatchError()
                key, value, ex = self.pending
                owner.data[key] = value
                owner.expiries[key] = ex
        return Pipeline()


@pytest.fixture
def redis_stub(monkeypatch):
    # Keep pure helper tests runnable without installing the runtime Redis client.
    if "redis.exceptions" not in sys.modules:
        try:
            import redis.exceptions
        except ImportError:
            monkeypatch.setitem(sys.modules, "redis.exceptions", SimpleNamespace(WatchError=type("WatchError", (Exception,), {})))
    return FakeRedis()


def test_case_store_persists_isolated_snapshot_and_rejects_stale_version(redis_stub):
    async def run():
        store = CaseStore(redis_stub)
        case = await store.load("u1", "c")
        saved = await store.save(case, 0)
        assert saved["version"] == 1 and case["version"] == 0
        assert await store.load("u1", "c") == saved
        assert (await store.load("u2", "c"))["facts"] == {}
        assert (await store.load("u2", "c"))["case_id"] != saved["case_id"]
        assert redis_stub.expiries[store.key("u1", "c")] == 86400
        with pytest.raises(CaseConflictError):
            await store.save(case, 0)
        assert (await store.load("u1", "c"))["version"] == 1
    asyncio.run(run())


def test_case_store_watch_race_has_no_lost_update(redis_stub):
    async def run():
        store = CaseStore(redis_stub)
        redis_stub.race = True
        with pytest.raises(CaseConflictError):
            await store.save(new_case("u", "c"), 0)
        assert redis_stub.data == {}
    asyncio.run(run())

@pytest.mark.parametrize("message", ["不要换个问题", "我不想换个问题", "不是新的问题", "你说换个问题是什么意思"])
def test_topic_mentions_and_negations_do_not_reset_case(message):
    case = new_case("u", "c")
    assert prepare_case(case, message) == case


def test_new_case_does_not_receive_old_background(monkeypatch):
    monkeypatch.setenv("NEXO_HARNESS_MODE", "enforce")
    case = prepare_case(new_case("u", "c"), "换个问题，打印机坏了")
    req = SimpleNamespace(context="旧账号手机号停用", entities={}, message="打印机坏了", case_state=case)
    assert "旧账号手机号停用" not in json.dumps(build_messages(req, ""), ensure_ascii=False)


def test_topic_reset_is_transient_but_history_boundary_survives():
    from harness.context import case_history
    case = prepare_case(new_case("u", "c"), "换个问题，打印机坏了")
    continued = prepare_case(case, "还是不行")
    assert "new_topic" not in continued
    assert continued["isolate_history"]
    role = SimpleNamespace(value="user")
    memory = SimpleNamespace(recent_messages=[
        SimpleNamespace(role=role, content="旧账号密码问题", metadata={"case_id": "old"}),
        SimpleNamespace(role=role, content="打印机坏了", metadata={"case_id": case["case_id"]}),
        SimpleNamespace(role=role, content="旧会话未标记", metadata={}),
    ])
    context, history = case_history(memory, continued)
    assert "打印机坏了" in context and "旧账号" not in context and "未标记" not in context
    assert history == [{"role": "user", "content": "打印机坏了"}]
    req = SimpleNamespace(context=context, entities={}, message="还是不行", harness_mode="enforce", case_state=continued)
    assert "打印机坏了" in json.dumps(build_messages(req, ""), ensure_ascii=False)


def test_h01_actual_paraphrase_does_not_roll_back_valid_facts_or_observation():
    message = "普通网页可以打开，但 VPN 一直连接超时，排查顺序是什么？"
    update = {"facts": [
        {"field": "target_system", "value": "VPN", "source_quote": "VPN 一直连接超时"},
        {"field": "issue_description", "value": "连接超时", "source_quote": "VPN 一直连接超时"},
        {"field": "issue_description", "value": "普通网页可打开，VPN 连接超时", "source_quote": "普通网页可以打开，但 VPN 一直连接超时"},
    ], "steps": [{"step_id": "s1", "description": "确认基础网络（普通网页可访问）",
        "status": "user_reported_done", "result": "普通网页可以打开", "source_quote": "普通网页可以打开"}]}
    case = apply_progress(new_case("u", "c"), update, message, "r1")
    assert case["facts"]["target_system"]["value"] == "VPN"
    assert case["facts"]["issue_description"]["value"] == "普通网页可以打开，但 VPN 一直连接超时"
    assert case["facts"]["issue_description"]["model_label"] == "普通网页可打开，VPN 连接超时"
    assert case["candidates"] == []
    assert case["steps"][0]["status"] == "user_reported_observation"
    assert case["steps"][0]["description"] == "普通网页可以打开"
    assert len(case["progress_warnings"]) == 1
    assert handoff_fields(case)["issue_description"] == "普通网页可以打开，但 VPN 一直连接超时"


def test_h02_actual_completed_actions_keep_source_despite_model_paraphrases():
    message = "我已经换过网络，也重新登录了，还是超时"
    update = {"facts": [{"field": "issue_description", "value": "换网络、重新登录后仍连接超时", "source_quote": message}],
        "steps": [
            {"step_id": "s1", "description": "切换网络后重试 VPN 连接", "status": "user_reported_done", "result": "仍超时", "source_quote": message},
            {"step_id": "s2", "description": "重新登录 VPN 后重试", "status": "user_reported_done", "result": "仍超时", "source_quote": message},
        ]}
    case = apply_progress(new_case("u", "c"), update, message, "r2")
    assert case["facts"]["issue_description"]["value"] == message
    assert case["candidates"] == []
    assert all(step["status"] == "user_reported_done" for step in case["steps"])
    assert all(step["description"] == message and step["result"] == message for step in case["steps"])
    assert case["steps"][0]["model_label"] == "切换网络后重试 VPN 连接"
    assert case["steps"][0]["model_result_label"] == "仍超时"
    assert handoff_fields(case) == {"issue_description": message}


def test_unsupported_fact_value_remains_candidate_not_confirmed_fact():
    case = apply_progress(new_case("u", "c"), {"facts": [
        {"field": "target_system", "value": "SAP", "source_quote": "目标系统是 OA"}]}, "目标系统是 OA", "r")
    assert case["facts"] == {}
    assert case["candidates"][0]["value"] == "SAP"
    assert case["candidates"][0]["source"] == "model_interpretation"
    assert handoff_fields(case) == {}


def test_flash_h01_invalid_title_reference_keeps_valid_facts():
    message = "普通网页可以打开，但 VPN 一直连接超时，排查顺序是什么？"
    case = apply_progress(new_case("u", "c"), {
        "facts": [{"field": "target_system", "value": "VPN", "source_quote": "VPN 一直连接超时"}],
        "evidence_refs": ["VPN 连接排查（演示资料）:actual", "actual", "invented"],
    }, message, "r", ["actual"])
    assert case["facts"]["target_system"]["value"] == "VPN"
    assert case["evidence_refs"] == ["actual"]
    assert [item["kind"] for item in case["progress_warnings"]] == ["invalid_evidence_ref"] * 2


def test_flash_h02_clipped_quotes_keep_whole_outcome_sentence():
    message = "我已经换过网络，也重新登录了，还是超时，下一步怎么办？"
    case = apply_progress(new_case("u", "c"), {"steps": [
        {"step_id": "step2", "description": "换一个网络验证（如切换手机热点）后重连 VPN",
         "status": "user_reported_done", "result": "仍超时", "source_quote": "我已经换过网络"},
        {"step_id": "step6", "description": "退出 VPN 客户端后重新登录再连接",
         "status": "user_reported_done", "result": "仍超时", "source_quote": "也重新登录了，还是超时"},
    ]}, message, "r")
    for step in case["steps"]:
        assert step["status"] == "user_reported_done"
        assert step["result"] == message[:-1]
        assert step["source_context"] == message[:-1]
        assert step["model_result_label"] == "仍超时"
    assert not case["progress_warnings"]


@pytest.mark.parametrize("message,quote", [
    ("我还没重新登录了", "重新登录了"),
    ("如果我重新登录了，会恢复吗？", "重新登录了"),
    ("我换过网络吗？", "换过网络"),
    ("是否重新登录了？", "重新登录了"),
    ("准备重启客户端后再试", "重启客户端后"),
])
def test_clipped_execution_quote_cannot_hide_negative_conditional_or_question(message, quote):
    rejected = apply_progress(new_case("u", "c"), {"steps": [{"step_id": "s", "description": quote,
        "status": "user_reported_done", "source_quote": quote, "result": ""}]}, message, "r")
    assert rejected["steps"] == []
    assert rejected["progress_warnings"][0]["kind"] == "invalid_step"


def test_completed_action_without_outcome_does_not_invent_result():
    message = "我重新登录了"
    case = apply_progress(new_case("u", "c"), {"steps": [{"step_id": "s", "description": "重新登录",
        "status": "user_reported_done", "source_quote": message, "result": "我重新登录了"}]}, message, "r")
    assert case["steps"][0]["status"] == "user_reported_done"
    assert case["steps"][0]["result"] == ""
    assert case["progress_warnings"][0]["kind"] == "unknown_step_result"


def test_result_quote_must_be_literal_and_same_sentence():
    message = "我重新登录了，还是超时。打印机恢复正常了。"
    step = {"step_id": "s", "description": "重新登录", "status": "user_reported_done",
            "source_quote": "我重新登录了", "result_quote": "还是超时"}
    case = apply_progress(new_case("u", "c"), {"steps": [step]}, message, "r")
    assert case["steps"][0]["result"] == "还是超时"
    assert "打印机" not in case["steps"][0]["source_context"]
    for invalid in ["打印机恢复正常了", "已经恢复正常"]:
        rejected = apply_progress(new_case("u", "c"), {"steps": [{**step, "result_quote": invalid}]}, message, "r")
        assert rejected["steps"] == []
        assert rejected["progress_warnings"][0]["kind"] == "invalid_step"


def test_issue_report_tracks_latest_raw_words_and_bounded_history_only():
    case = new_case("u", "c")
    for index in range(7):
        quote = f"第{index}次尝试还是连接超时"
        case = apply_progress(case, {"facts": [{"field": "issue_description", "value": "VPN 无法连接",
            "source_quote": quote}]}, quote, str(index))
    fact = case["facts"]["issue_description"]
    assert fact["value"] == quote
    assert fact["model_label"] == "VPN 无法连接"
    assert len(fact["previous_reports"]) == 4
    assert fact["previous_reports"][0]["request_id"] == "2"
    assert not case["conflicts"]
    assert handoff_fields(case)["issue_description"] == quote


@pytest.mark.parametrize("message,quote,description", [
    ("已经重启了，但没有效果", "已经重启了", "重启"),
    ("重新登录了，还是无法连接", "重新登录了", "重新登录"),
    ("已经重启了但没重新登录", "已经重启了", "重启"),
])
def test_failure_or_another_unperformed_action_does_not_negate_completed_step(message, quote, description):
    case = apply_progress(new_case("u", "c"), {"steps": [{"step_id": "s", "description": description,
        "status": "user_reported_done", "source_quote": quote}]}, message, "r")
    assert case["steps"][0]["status"] == "user_reported_done"


@pytest.mark.parametrize("message,quote", [
    ("没有重启", "重启"),
    ("已经重启了但没重新登录", "重新登录"),
])
def test_negation_before_selected_action_is_preserved(message, quote):
    rejected = apply_progress(new_case("u", "c"), {"steps": [{"step_id": "s", "description": quote,
        "status": "user_reported_done", "source_quote": quote}]}, message, "r")
    assert rejected["steps"] == []
    assert rejected["progress_warnings"][0]["kind"] == "invalid_step"


@pytest.mark.parametrize("field", ["result", "result_quote"])
def test_clipped_positive_result_preserves_negative_context(field):
    message = "重启客户端后没有恢复正常"
    case = apply_progress(new_case("u", "c"), {"steps": [{"step_id": "s", "description": "重启客户端",
        "status": "user_reported_done", "source_quote": "重启客户端后", field: "恢复正常"}]}, message, "r")
    assert case["steps"][0]["result"] == message


@pytest.mark.parametrize("message", ["不是现在正常了", "如果现在正常了", "是不是现在正常了", "现在正常了？"])
def test_clipped_resolved_quote_cannot_hide_negation_condition_or_question(message):
    with pytest.raises(ValueError):
        apply_progress(new_case("u", "c"), {"status": "resolved", "status_quote": "现在正常了"}, message, "r")


def test_explicit_normal_confirmation_still_resolves():
    case = apply_progress(new_case("u", "c"), {"status": "resolved", "status_quote": "现在正常了"},
                          "现在正常了。谢谢", "r")
    assert case["status"] == "resolved"


def test_h03_facts_survive_invalid_agent_steps_then_explicit_correction_and_history_replay():
    first = "我需要 SAP 的只读权限，用于查看需求。"
    case = apply_progress(new_case("u", "c"), {
        "facts": [{"field": field, "value": value, "source_quote": first} for field, value in [
            ("target_system", "SAP"), ("requested_access", "只读权限"), ("purpose", "查看需求")]],
        "steps": [
            {"step_id": "check_fields", "description": "使用 check_access_fields 检查申请字段完整性",
             "status": "user_reported_done", "result": "工具返回目标系统未识别为已提供"},
            {"step_id": "kb_lookup", "description": "检索权限申请流程知识库",
             "status": "user_reported_done", "result": "检索到权限申请流程"},
        ], "status": "waiting_user",
    }, first, "r1")
    assert handoff_fields(case) == {"target_system": "SAP", "requested_access": "只读权限", "purpose": "查看需求"}
    assert case["steps"] == []
    assert [warning["step_id"] for warning in case["progress_warnings"]] == ["check_fields", "kb_lookup"]
    correction = "更正一下，我要的是 GitLab 的只读权限，用于查看项目代码。"
    quote = "我要的是 GitLab 的只读权限，用于查看项目代码。"
    update = {"facts": [
        {"field": "target_system", "value": "GitLab", "source_quote": quote, "correction": True},
        {"field": "requested_access", "value": "只读权限", "source_quote": quote},
        {"field": "purpose", "value": "查看项目代码", "source_quote": quote},
    ], "status": "collecting"}
    case = apply_progress(case, update, correction, "r2")
    assert handoff_fields(case) == {"target_system": "GitLab", "requested_access": "只读权限", "purpose": "查看项目代码"}
    assert case["facts"]["purpose"]["correction_context"] == correction
    assert case["conflicts"] == []
    previous_facts = copy.deepcopy(case["facts"])
    case = apply_progress(case, {**update, "status": "waiting_user"}, "请整理刚才的权限申请信息。", "r3")
    assert case["facts"] == previous_facts
    assert [warning["kind"] for warning in case["progress_warnings"]] == ["invalid_fact"] * 3


def test_invalid_fact_or_step_does_not_drop_valid_siblings():
    message = "目标系统是 SAP。我重启客户端后仍超时。"
    case = apply_progress(new_case("u", "c"), {
        "facts": [{"field": "target_system", "value": "OA", "source_quote": "虚构来源"},
                  {"field": "target_system", "value": "SAP", "source_quote": "目标系统是 SAP"}],
        "steps": [{"step_id": "bad", "description": "工具查询", "status": "user_reported_done"},
                  {"step_id": "restart", "description": "重启客户端", "status": "user_reported_done",
                   "source_quote": "我重启客户端后仍超时", "result": "仍超时"}],
    }, message, "r")
    assert handoff_fields(case)["target_system"] == "SAP"
    assert [step["step_id"] for step in case["steps"]] == ["restart"]
    assert [warning["kind"] for warning in case["progress_warnings"]] == ["invalid_fact", "invalid_step"]


def test_explicit_correction_does_not_spread_to_another_sentence():
    case = apply_progress(new_case("u", "c"), {"facts": [
        {"field": "purpose", "value": "查看需求", "source_quote": "查看需求"}]}, "查看需求", "r1")
    message = "更正一下，目标是 GitLab。用途是查看代码。"
    changed = apply_progress(case, {"facts": [
        {"field": "purpose", "value": "查看代码", "source_quote": "用途是查看代码"}]}, message, "r2")
    assert changed["facts"]["purpose"]["value"] == "查看需求"
    assert changed["conflicts"][0]["field"] == "purpose"


@pytest.mark.parametrize("update", [{"facts": {}}, {"steps": {}}, {"status": "invalid"}, {"new_case": "true"}])
def test_root_contract_remains_strict(update):
    with pytest.raises(ValueError):
        apply_progress(new_case("u", "c"), update, "VPN", "r")
