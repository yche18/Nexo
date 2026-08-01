import pytest

from core.intent_recognizer import IntentCategory, IntentRecognizer


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
