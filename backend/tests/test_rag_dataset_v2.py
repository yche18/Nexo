"""v2 只修订两题的评判标注，不改原问题、证据语料或 v1 文件。"""
import copy
import hashlib
import json
from pathlib import Path


DATASETS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets"
V1_SHA256 = "03566df0ed799f9ccb8a4d76bf6bea550edacb39df6cad56634e358cbf8ee886"


def load_dataset(name):
    return json.loads((DATASETS / name).read_text(encoding="utf-8"))


def test_v1_bytes_remain_frozen():
    assert hashlib.sha256((DATASETS / "rag_support_v1.json").read_bytes()).hexdigest() == V1_SHA256


def test_v2_preserves_all_questions_ids_and_corpus():
    v1 = load_dataset("rag_support_v1.json")
    v2 = load_dataset("rag_support_v2.json")
    assert v2["version"] == "rag-support-v2"
    assert v2["parent_version"] == v1["version"] == "rag-support-v1"
    assert "no independent human annotation" in v2["annotation_note"]
    assert "fixed before v2 model runs" in v2["annotation_note"]
    assert len(v1["cases"]) == len(v2["cases"]) == 24
    assert v1["corpus"] == v2["corpus"]
    for original, revised in zip(v1["cases"], v2["cases"]):
        assert original["id"] == revised["id"]
        assert original["question"] == revised["question"]
        assert original["relevant_chunk_ids"] == revised["relevant_chunk_ids"]
        if original["id"] not in ("rag_05", "rag_19"):
            assert original == revised
        else:
            old_fields = copy.deepcopy(original)
            new_fields = copy.deepcopy(revised)
            for field in ("required_points", "forbidden_claims"):
                del old_fields[field]
                del new_fields[field]
            assert old_fields == new_fields
    old_top = {key: value for key, value in v1.items()
               if key not in ("version", "annotation_note", "cases")}
    new_top = {key: value for key, value in v2.items()
               if key not in ("version", "annotation_note", "parent_version", "cases")}
    assert old_top == new_top


def test_rag05_only_clarifies_the_first_required_point():
    original = {case["id"]: case for case in load_dataset("rag_support_v1.json")["cases"]}["rag_05"]
    revised = {case["id"]: case for case in load_dataset("rag_support_v2.json")["cases"]}["rag_05"]
    assert original["required_points"][0] != revised["required_points"][0]
    assert original["required_points"][1:] == revised["required_points"][1:]
    assert original["forbidden_claims"] == revised["forbidden_claims"]
    assert "提示用户补充用途即可" in revised["required_points"][0]
    assert "不要求替用户填写或猜测具体用途" in revised["required_points"][0]


def test_rag19_preserves_original_points_and_appends_three_objective_checks():
    original = {case["id"]: case for case in load_dataset("rag_support_v1.json")["cases"]}["rag_19"]
    revised = {case["id"]: case for case in load_dataset("rag_support_v2.json")["cases"]}["rag_19"]
    old_points = original["required_points"]
    old_forbidden = original["forbidden_claims"]
    assert len(old_points) == 3
    assert revised["required_points"][:len(old_points)] == old_points
    assert revised["forbidden_claims"][:len(old_forbidden)] == old_forbidden
    assert len(revised["required_points"]) == len(old_points) + 3
    assert len(revised["forbidden_claims"]) == len(old_forbidden) + 3
    assert "没有提出重装建议时，此项视为满足" in revised["required_points"][-1]
    assert "明确禁止、风险提醒" in revised["forbidden_claims"][-2]
    assert "不算" in revised["forbidden_claims"][-2]
