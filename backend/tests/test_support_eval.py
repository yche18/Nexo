import json
from pathlib import Path

from evaluation.run_support_eval import check_turn, dialog_metrics, intent_metrics, validate_dataset


def test_support_dataset_size_splits_and_labels():
    data = json.loads((Path(__file__).parents[1] / "evaluation/datasets/support_v1.json").read_text(encoding="utf-8"))
    validate_dataset(data)
    assert len(data["intent_cases"]) == 60
    assert sum(c["split"] == "dev" for c in data["intent_cases"]) == 40
    assert len({c["expected_intent"] for c in data["intent_cases"]}) == 14
    assert len(data["dialog_cases"]) == 10
    assert sum(len(c["turns"]) for c in data["dialog_cases"]) == 30


def test_intent_metrics_include_incorrect_and_failed_requests():
    rows = [
        {"expected":"query", "predicted":"query", "passed":True},
        {"expected":"query", "predicted":"request", "passed":False, "llm_failed":True},
        {"expected":"request", "predicted":None, "passed":False, "error":"timeout"},
    ]
    result = intent_metrics(rows)
    assert result["accuracy"] == 1 / 3
    assert result["request_errors"] == 1
    assert result["llm_fallback_count"] == 1
    assert result["per_class"]["request"]["recall"] == 0


def test_handoff_assertion_checks_the_field_not_an_unrelated_paragraph():
    spec = {"fields":{"申请权限":["只读"]}, "fields_absent":["申请用途"]}
    result = check_turn(spec, {"response":"之前说过只读。\n- 申请权限：读写\n- 申请用途：未说明"})
    assert not next(c["passed"] for c in result if c["name"] == "field:申请权限")
    assert not next(c["passed"] for c in result if c["name"] == "field_absent:申请用途")


def test_dialog_metrics_do_not_treat_failed_judges_as_valid_scores():
    scenes = [{"planned_turns":2, "behavior_passed":False, "passed":False,
               "turn_results":[{"wall_seconds":1.0,"checks":[{"passed":True}],
                                "judge":{"failed":True,"overall":0.5}}, {"error":"timeout"}]}]
    result = dialog_metrics(scenes)
    assert result["planned_turns"] == 2
    assert result["completed_turns"] == 1
    assert result["judge_failed"] == 1
    assert result["judge_valid"] == 0
    assert result["judge_scores_mean"] == {}


def test_action_claim_check_distinguishes_questions_from_execution_claims():
    def passed(text):
        return next(c["passed"] for c in check_turn({}, {"response":text}) if c["name"] == "no_unavailable_action_claim")
    assert passed("确认账号是否被锁定、VPN 权限是否已开通。")
    assert passed("如果 IT 已开通权限，请重新登录。")
    assert passed("尚未创建工单或通知 IT 人员。")
    assert not passed("我已为您开通权限。")
    assert not passed("已经帮你重置密码。")
    assert not passed("已创建工单，请等待处理。")


def test_rescore_preserves_saved_answers_judge_scores_and_original_file(tmp_path):
    import hashlib
    from evaluation.run_support_eval import rescore_report

    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"dialog_cases":[{"id":"d01","turns":[{"message":"test"}]}]}), encoding="utf-8")
    scores = {"failed":False,"overall":0.9,"relevance":0.9,"accuracy":0.9,"completeness":0.9,"helpfulness":0.9}
    turn = {"turn":1,"response":{"response":"确认权限是否已开通。"},"wall_seconds":1.0,
            "checks":[{"name":"no_unavailable_action_claim","passed":False}],"judge":scores}
    original = {"dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),"judge_threshold":0.75,
                "intent_results":[{"expected":"query","predicted":"query","passed":True}],
                "dialog_results":[{"id":"d01","planned_turns":1,"turn_results":[turn],"behavior_passed":False,"passed":False}]}
    source, output = tmp_path / "raw.json", tmp_path / "reviewed.json"
    source.write_text(json.dumps(original), encoding="utf-8")
    before = source.read_bytes()
    rescore_report(source, dataset, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert source.read_bytes() == before
    assert result["intent_results"] == original["intent_results"]
    assert result["dialog_results"][0]["turn_results"][0]["response"] == turn["response"]
    assert result["dialog_results"][0]["turn_results"][0]["judge"] == scores
    assert result["dialog_results"][0]["passed"] is True
