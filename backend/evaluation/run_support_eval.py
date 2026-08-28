"""Run the versioned support suite without changing the application's default baseline."""
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import anthropic
import httpx

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
sys.path.insert(0, str(BACKEND))
from core.intent_recognizer import IntentCategory, IntentRecognizer
from evaluation.evaluator import EndToEndEvaluator, LLMJudge


def validate_dataset(data):
    labels = {category.value for category in IntentCategory}
    ids, messages, families = set(), set(), set()
    for case in data["intent_cases"]:
        assert case["id"] not in ids and case["message"] not in messages
        assert case["family"] not in families, "Paraphrase families must not cross splits"
        assert case["expected_intent"] in labels
        assert case["split"] in ("dev", "holdout")
        ids.add(case["id"])
        messages.add(case["message"])
        families.add(case["family"])
    for case in data["dialog_cases"]:
        assert case["id"] not in ids
        assert case["split"] in ("dev", "holdout")
        assert 2 <= len(case["turns"]) <= 4
        assert all(turn["message"].strip() for turn in case["turns"])
        ids.add(case["id"])


def intent_metrics(rows):
    labels = sorted({r["expected"] for r in rows} | {r["predicted"] for r in rows if r.get("predicted")})
    per_class = {}
    for label in labels:
        tp = sum(r["expected"] == label and r.get("predicted") == label for r in rows)
        fp = sum(r["expected"] != label and r.get("predicted") == label for r in rows)
        fn = sum(r["expected"] == label and r.get("predicted") != label for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": tp + fn, "precision": precision, "recall": recall, "f1": f1}
    correct = sum(r["passed"] for r in rows)
    return {
        "total": len(rows), "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": sum(x["f1"] for x in per_class.values()) / len(labels) if labels else None,
        "llm_fallback_count": sum(r.get("llm_failed", False) for r in rows),
        "request_errors": sum(bool(r.get("error")) for r in rows),
        "per_class": per_class,
    }


def check_turn(spec, response):
    text = response.get("response", "")
    checks = [{"name": "nonempty_response", "passed": bool(text.strip())}]
    def add(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
    if "agent_in" in spec:
        add("agent_in:" + ",".join(spec["agent_in"]), response.get("agent_type") in spec["agent_in"])
    for agent in spec.get("agents_include", []):
        add("collaborating_agent:" + agent, agent in response.get("agent_types", []))
    for value in spec.get("contains_all", []):
        add("contains:" + value, value in text)
    for value in spec.get("not_contains", []):
        add("excludes:" + value, value not in text)
    for field, values in spec.get("fields", {}).items():
        match = re.search(r"(?m)^- " + re.escape(field) + r"：([^\n]+)", text)
        add("field:" + field, match is not None and all(v in match.group(1) for v in values))
    for field in spec.get("fields_absent", []):
        add("field_absent:" + field, re.search(r"(?m)^- " + re.escape(field) + r"：", text) is None)
    # A conservative literal check, not a full semantic guarantee. Review any hits in the report.
    actor = r"(?:(?:我|我们|系统)(?:已经|已)(?:为您|帮您|为你|帮你)?|(?:已经|已)(?:为您|帮您|为你|帮你)|(?:^|\n)\s*(?:[-*]\s*)?(?:已经|已))"
    action = r"(?:成功)?(?:开通|授予|创建工单|提交工单|重置密码|通知(?:了)?(?:IT|人工))"
    claim = re.search(actor + action, text)
    add("no_unavailable_action_claim", claim is None)
    return checks


def dialog_metrics(scenes):
    turns = [turn for scene in scenes for turn in scene["turn_results"]]
    valid_judges = [t["judge"] for t in turns if t.get("judge") and not t["judge"]["failed"]]
    assertions = [c for t in turns for c in t.get("checks", [])]
    times = sorted(t["wall_seconds"] for t in turns if "wall_seconds" in t)
    return {
        "scenarios": len(scenes), "planned_turns": sum(s["planned_turns"] for s in scenes),
        "completed_turns": sum(not t.get("error") for t in turns),
        "judge_valid": len(valid_judges),
        "judge_failed": sum(bool(t.get("judge", {}).get("failed")) for t in turns),
        "judge_passed": sum(j["overall"] >= EndToEndEvaluator.PASS_THRESHOLD for j in valid_judges),
        "judge_scores_mean": {k: sum(j[k] for j in valid_judges) / len(valid_judges) for k in ("relevance", "accuracy", "completeness", "helpfulness")} if valid_judges else {},
        "assertions": len(assertions), "assertions_passed": sum(c["passed"] for c in assertions),
        "behavior_scenarios_passed": sum(s["behavior_passed"] for s in scenes),
        "combined_scenarios_passed": sum(s["passed"] for s in scenes),
        "chat_latency_p50_seconds": times[len(times)//2] if times else None,
        "chat_latency_p95_seconds": times[max(0, int(len(times)*0.95 + 0.999999)-1)] if times else None,
    }


def rescore_report(source, dataset, output):
    """Recheck saved responses without new model calls; preserve the original report."""
    if output.exists():
        raise RuntimeError("Output report already exists")
    raw = dataset.read_bytes()
    data = json.loads(raw)
    report = json.loads(source.read_text(encoding="utf-8"))
    assert report["dataset_sha256"] == hashlib.sha256(raw).hexdigest(), "Dataset changed"
    specs = {case["id"]:case for case in data["dialog_cases"]}
    for scene in report["dialog_results"]:
        for turn in scene["turn_results"]:
            if "response" in turn:
                turn["checks"] = check_turn(specs[scene["id"]]["turns"][turn["turn"]-1], turn["response"])
        complete = len(scene["turn_results"]) == scene["planned_turns"]
        scene["behavior_passed"] = complete and all(not t.get("error") and all(c["passed"] for c in t.get("checks", [])) for t in scene["turn_results"])
        scene["passed"] = scene["behavior_passed"] and all(not t.get("judge",{}).get("failed", True) and t["judge"]["overall"] >= report["judge_threshold"] for t in scene["turn_results"])
    report["dialog_metrics"] = dialog_metrics(report["dialog_results"])
    report["assertion_version"] = "1.1"
    report["rescored_from"] = source.name
    report["original_report_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["rescored_at"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["dialog_metrics"], ensure_ascii=False))


async def run(args):
    raw = args.dataset.read_bytes()
    data = json.loads(raw)
    validate_dataset(data)
    intents = [c for c in data["intent_cases"] if c["split"] == args.split]
    dialogs = [c for c in data["dialog_cases"] if c["split"] == args.split]
    if args.validate_only:
        print(json.dumps({"split": args.split, "intent_count": len(intents), "dialog_count": len(dialogs), "dataset_sha256": hashlib.sha256(raw).hexdigest()}))
        return
    if args.rescore:
        rescore_report(args.rescore, args.dataset, args.report)
        return
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    if args.report.exists():
        raise RuntimeError("Report already exists; choose a new path to retain previous evidence")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    config = {"api_key": os.environ["ANTHROPIC_API_KEY"], "model": model}
    if os.getenv("ANTHROPIC_BASE_URL"):
        config["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    recognizer = IntentRecognizer(**config)
    client_config = {k:v for k,v in config.items() if k != "model"}
    judge_client = anthropic.AsyncAnthropic(**client_config, timeout=60.0, max_retries=0)
    judge = LLMJudge(judge_client, model)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        revision = None
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "dataset_version": data["version"], "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "assertion_version": "1.1",
        "split": args.split, "model": model, "thinking_env": os.getenv("LLM_THINKING_ENABLED"),
        "sdk_version": anthropic.__version__, "application_revision": revision,
        "judge_threshold": EndToEndEvaluator.PASS_THRESHOLD,
        "planned_intents": len(intents), "planned_dialogs": len(dialogs),
        "fresh_intent_cache": True, "intent_results": [], "dialog_results": [],
    }
    start = time.monotonic()
    def save():
        report["intent_metrics"] = intent_metrics(report["intent_results"])
        report["dialog_metrics"] = dialog_metrics(report["dialog_results"])
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for i, case in enumerate(intents):
            t0 = time.monotonic()
            row = {"id": case["id"], "message": case["message"], "expected": case["expected_intent"], "predicted": None, "passed": False}
            try:
                result = await recognizer.recognize(case["message"])
                row.update(predicted=result.intent.value, passed=result.intent.value == case["expected_intent"], confidence=result.confidence, reasoning=result.reasoning, source_scores=result.source_scores, llm_failed="LLM 失败" in result.reasoning)
            except Exception as ex:
                row["error"] = type(ex).__name__
            row["wall_seconds"] = round(time.monotonic()-t0, 3)
            report["intent_results"].append(row)
            save()
            if (i+1) % 5 == 0 or not row["passed"]:
                print(json.dumps({"phase":"intent", "done":i+1, "total":len(intents), "id":case["id"], "passed":row["passed"], "predicted":row["predicted"]}), flush=True)
        async with httpx.AsyncClient(timeout=120.0) as http:
            for case in dialogs:
                user_id = "eval_" + run_id + "_" + case["id"]
                history = []
                scene = {"id":case["id"], "name":case["name"], "planned_turns":len(case["turns"]), "turn_results":[], "behavior_passed":False, "passed":False}
                report["dialog_results"].append(scene)
                for index, spec in enumerate(case["turns"]):
                    row = {"turn":index+1, "question":spec["message"]}
                    t0 = time.monotonic()
                    try:
                        response = await http.post(args.api_base.rstrip("/") + "/chat", json={"message":spec["message"], "user_id":user_id, "conv_id":user_id})
                        response.raise_for_status()
                        body = response.json()
                        row["wall_seconds"] = round(time.monotonic()-t0, 3)
                        row["response"] = body
                        row["checks"] = check_turn(spec, body)
                        context = "\n".join(x["role"] + ": " + x["content"] for x in history[-6:])
                        scores = await judge.judge(spec["message"], body.get("response", ""), context=context or None)
                        row["judge"] = {"relevance":scores.relevance, "accuracy":scores.accuracy, "completeness":scores.completeness, "helpfulness":scores.helpfulness, "overall":scores.overall, "failed":scores.judge_failed, "error":scores.error}
                        history.extend([{"role":"user","content":spec["message"]},{"role":"assistant","content":body.get("response", "")}])
                    except Exception as ex:
                        row["error"] = type(ex).__name__
                    scene["turn_results"].append(row)
                    save()
                    print(json.dumps({"phase":"dialog", "id":case["id"], "turn":index+1, "error":row.get("error"), "checks_failed":sum(not c["passed"] for c in row.get("checks", [])), "judge_failed":row.get("judge",{}).get("failed")}), flush=True)
                    if row.get("error"):
                        break
                complete = len(scene["turn_results"]) == scene["planned_turns"]
                scene["behavior_passed"] = complete and all(not t.get("error") and all(c["passed"] for c in t.get("checks", [])) for t in scene["turn_results"])
                scene["passed"] = scene["behavior_passed"] and all(not t.get("judge",{}).get("failed", True) and t["judge"]["overall"] >= EndToEndEvaluator.PASS_THRESHOLD for t in scene["turn_results"])
                save()
    finally:
        report["wall_seconds"] = round(time.monotonic()-start, 3)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["intent_cache_hits"] = recognizer.cache_hits
        save()
        await recognizer.client.close()
        await judge_client.close()
    print(json.dumps({"intent_metrics":report["intent_metrics"], "dialog_metrics":report["dialog_metrics"], "wall_seconds":report["wall_seconds"]}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=BACKEND / "evaluation/datasets/support_v1.json")
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--api-base", default="http://localhost:18081")
    parser.add_argument("--report", type=Path, default=ROOT / "data/eval/support_v1" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rescore", type=Path, help="Recheck an existing report without model calls")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
