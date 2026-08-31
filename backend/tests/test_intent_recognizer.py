import pytest

from core.intent_recognizer import (
    IntentCategory, IntentRecognizer, match_domain_keywords,
    NETWORK_ISSUE_KEYWORDS, DEVICE_FAILURE_KEYWORDS, SOFTWARE_INSTALL_KEYWORDS,
)


def make_recognizer(threshold=0.5):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._embedding_enabled = True
    recognizer.threshold = threshold
    return recognizer


def vote(recognizer, llm, emb, pat):
    return recognizer._vote(
        {"intent": llm[0], "confidence": llm[1]},
        {"intent": emb[0], "confidence": emb[1]},
        {"intent": pat[0], "confidence": pat[1]},
    )


def test_pattern_cannot_refine_a_different_intent_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.ACCOUNT, 0.9),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.ACCOUNT
    assert confidence == pytest.approx(0.63)
    assert "refined_by_pattern" not in sources


def test_coarse_and_fine_scores_choose_the_same_group_together():
    intent, confidence, sources = vote(
        make_recognizer(threshold=0.2),
        (IntentCategory.ACCOUNT, 0.3),       # weighted score 0.21
        (IntentCategory.TECHNICAL_LOGIN, 0.9),  # weighted score 0.18
        (IntentCategory.TECHNICAL, 0.4),     # weighted score 0.04
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.22)
    assert "refined_by_pattern" not in sources


def test_pattern_can_still_refine_within_the_winning_group():
    intent, confidence, sources = vote(
        make_recognizer(),
        (IntentCategory.TECHNICAL, 0.8),
        (IntentCategory.OTHER, 0.0),
        (IntentCategory.TECHNICAL_LOGIN, 0.5),
    )

    assert intent is IntentCategory.TECHNICAL_LOGIN
    assert confidence == pytest.approx(0.61)
    assert sources["refined_by_pattern"] == pytest.approx(0.5)


def test_it_entity_extraction_for_diagnostic_and_access_requests():
    recognizer = make_recognizer()
    diagnostic = recognizer._extract_entities("今天 VPN 无法登录，错误码：401")
    assert diagnostic == {
        "target_system": ["VPN"], "issue_description": ["无法登录"],
        "date": ["今天"], "error_code": ["401"], "requested_access": [], "purpose": [],
        "operating_system": [], "client_version": [], "network_type": [],
        "impact_scope": [], "device_type": [], "device_model": [],
    }
    access = recognizer._extract_entities("目标系统：研发平台，申请权限：只读权限，用于查看需求。")
    assert access["target_system"] == ["研发平台"]
    assert access["requested_access"] == ["只读权限"]
    assert access["purpose"] == ["查看需求"]
    assert not access["issue_description"]


def test_entities_stay_empty_when_current_message_has_no_explicit_details():
    assert all(not values for values in make_recognizer()._extract_entities("转 IT 人工处理").values())


def test_existing_date_and_error_code_rules_keep_their_scope():
    entities = make_recognizer()._extract_entities("昨天 2026-07-01 HTTP 403，状态码：500，ERR_CONNECTION_RESET")
    assert entities["date"] == ["昨天", "2026-07-01"]
    assert entities["error_code"] == ["403", "500"]


@pytest.mark.parametrize("message, expected", [
    ("VPN 连接超时", IntentCategory.NETWORK_ISSUE),
    ("电脑连不上公司 Wi-Fi，Wi-Fi 无法连接", IntentCategory.NETWORK_ISSUE),
    ("内网无法访问，DNS 解析失败", IntentCategory.NETWORK_ISSUE),
    ("电脑无法开机", IntentCategory.DEVICE_FAILURE),
    ("打印机卡纸", IntentCategory.DEVICE_FAILURE),
    ("手机无法充电", IntentCategory.DEVICE_FAILURE),
    ("电脑上的软件闪退", IntentCategory.TECHNICAL_CRASH),
    ("帮我安装办公软件", IntentCategory.SOFTWARE_INSTALL),
    ("VPN 客户端安装失败", IntentCategory.SOFTWARE_INSTALL),
    ("申请 VPN 权限，开通访问权限", IntentCategory.PERMISSION_REQUEST),
    ("忘记密码，请重置密码", IntentCategory.ACCOUNT_SECURITY),
    ("VPN 连接超时，请转人工", IntentCategory.HUMAN_HANDOFF),
    ("打印机卡纸，请转人工", IntentCategory.HUMAN_HANDOFF),
])
def test_new_domain_patterns_respect_business_boundaries(message, expected):
    assert make_recognizer()._pattern_recognize(message)["intent"] is expected


@pytest.mark.parametrize("coarse, fine, group", [
    (IntentCategory.NETWORK, IntentCategory.NETWORK_ISSUE, "network"),
    (IntentCategory.DEVICE, IntentCategory.DEVICE_FAILURE, "device"),
    (IntentCategory.TECHNICAL, IntentCategory.SOFTWARE_INSTALL, "technical"),
])
def test_new_fine_intents_refine_only_their_own_coarse_group(coarse, fine, group):
    recognizer = make_recognizer()
    intent, confidence, sources = vote(
        recognizer, (coarse, 0.8), (IntentCategory.OTHER, 0), (fine, 0.5),
    )
    assert intent is fine
    assert confidence == pytest.approx(0.61)
    assert recognizer._intent_group(intent) == group
    assert sources["refined_by_pattern"] == 0.5


def test_network_votes_do_not_accumulate_into_technical_group():
    intent, confidence, sources = vote(
        make_recognizer(), (IntentCategory.TECHNICAL, 0.8),
        (IntentCategory.NETWORK_ISSUE, 0.9), (IntentCategory.NETWORK_ISSUE, 0.5),
    )
    assert intent is IntentCategory.TECHNICAL
    assert confidence == pytest.approx(0.56)
    assert "refined_by_pattern" not in sources


def test_environment_entities_preserve_user_values_without_fault_inference():
    entities = make_recognizer()._extract_entities(
        "我的笔记本用Windows 11系统，客户端版本：6.2.1，公司 Wi-Fi 连接超时，"
        "同部门多人也有问题。设备型号：ThinkPad T14。"
    )
    assert entities["operating_system"] == ["Windows 11"]
    assert entities["client_version"] == ["6.2.1"]
    assert entities["network_type"] == ["公司 Wi-Fi"]
    assert entities["impact_scope"] == ["同部门多人"]
    assert entities["device_type"] == ["笔记本"]
    assert entities["device_model"] == ["ThinkPad T14"]
    assert entities["issue_description"] == ["连接超时"]


@pytest.mark.parametrize("message", [
    "如果换成 Windows 11，使用手机热点会好吗？",
    "不是 Windows 11，也没有使用手机热点。",
    "建议试试公司 Wi-Fi，是否只有我出问题？",
    "设备型号：未知吗？",
])
def test_environment_entities_skip_hypotheses_negations_and_questions(message):
    entities = make_recognizer()._extract_entities(message)
    for field in ("operating_system", "client_version", "network_type", "impact_scope", "device_type", "device_model"):
        assert entities[field] == []


def test_device_name_or_network_name_alone_does_not_select_failure_intent():
    recognizer = make_recognizer()
    for message in ("VPN", "电脑", "打印机", "公司 Wi-Fi"):
        assert recognizer._pattern_recognize(message)["intent"] is IntentCategory.OTHER
    assert recognizer._extract_entities("电脑")["device_type"] == ["电脑"]


def test_environment_values_can_be_extracted_from_new_correction_clause():
    entities = make_recognizer()._extract_entities("不是 Windows 10，是 Windows 11；客户端版本：6.3.0")
    assert entities["operating_system"] == ["Windows 11"]
    assert entities["client_version"] == ["6.3.0"]


def test_software_update_background_does_not_override_crash_symptom():
    message = "软件更新后反复 crash，每次启动几秒就退出。"
    assert make_recognizer()._pattern_recognize(message)["intent"] is IntentCategory.TECHNICAL_CRASH


def test_phone_hotspot_is_network_environment_not_affected_device():
    entities = make_recognizer()._extract_entities("笔记本使用手机热点")
    assert entities["network_type"] == ["手机热点"]
    assert entities["device_type"] == ["笔记本"]


@pytest.mark.parametrize("message, keywords", [
    ("不是打印机卡纸，是电脑软件闪退", DEVICE_FAILURE_KEYWORDS),
    ("并非 VPN 连接超时，是密码忘了", NETWORK_ISSUE_KEYWORDS),
    ("如果软件安装失败怎么办", SOFTWARE_INSTALL_KEYWORDS),
    ("没有出现电脑无法开机的问题", DEVICE_FAILURE_KEYWORDS),
])
def test_new_domain_keywords_ignore_explicit_negation_and_hypothesis(message, keywords):
    assert match_domain_keywords(message, keywords) == []


def test_denied_device_fault_does_not_override_affirmed_software_crash():
    result = make_recognizer()._pattern_recognize("不是打印机卡纸，是电脑软件闪退")
    assert result["intent"] is IntentCategory.TECHNICAL_CRASH


def test_affirmed_domain_fault_after_a_correction_is_kept():
    assert match_domain_keywords("不是电脑无法开机，而是打印机卡纸", DEVICE_FAILURE_KEYWORDS) == ["打印机卡纸"]


def test_negated_device_possession_does_not_become_a_fact():
    entities = make_recognizer()._extract_entities("我没有电脑，只有手机热点")
    assert entities["device_type"] == []
    assert entities["network_type"] == ["手机热点"]


@pytest.mark.parametrize("message", [
    "我已经换成手机热点了，还是超时", "我换成手机热点后，还是超时",
])
def test_completed_network_switch_can_supply_environment(message):
    entities = make_recognizer()._extract_entities(message)
    assert entities["network_type"] == ["手机热点"]
    assert entities["device_type"] == []
