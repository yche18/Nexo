"""选择完整证据片段，再有限扩窗；预算单位明确为 JSON 字符数。"""
import asyncio
import hashlib
import json
from typing import Any, Dict, List


def chunk_identity(item: Dict[str, Any]) -> str:
    if item.get("chunk_id"):
        return str(item["chunk_id"])
    # 兼容旧工具返回值；新版知识库均提供稳定 chunk_id。
    raw = json.dumps([item.get("title"), item.get("chunk"), item.get("content")], ensure_ascii=False)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def usable_chunk(item: Any) -> bool:
    return (isinstance(item, dict) and not item.get("fallback")
            and isinstance(item.get("content"), str) and bool(item["content"].strip())
            and item.get("status", "ready") in ("ready", "legacy"))


def evidence_chars(items: List[Dict[str, Any]]) -> int:
    # 计算实际证据列表的 JSON 长度，含标题、来源和 ID，不冒充模型 token 数。
    return len(json.dumps(items, ensure_ascii=False))


def select_chunks(items: List[Dict[str, Any]], top_k: int, max_chars: int) -> List[Dict[str, Any]]:
    selected, seen = [], set()
    for item in items:
        if len(selected) >= top_k:
            break
        if not usable_chunk(item):
            continue
        key = chunk_identity(item)
        if key in seen:
            continue
        seen.add(key)
        candidate = dict(item)
        candidate["evidence_role"] = "core"
        if evidence_chars(selected + [candidate]) <= max_chars:
            selected.append(candidate)
    return selected


def same_section(core: Dict[str, Any], neighbor: Dict[str, Any]) -> bool:
    if core.get("status") != "ready" or neighbor.get("status") != "ready":
        return False
    if not core.get("doc_id") or not core.get("doc_version"):
        return False
    return all(core.get(key) == neighbor.get(key) for key in ("doc_id", "doc_version", "section"))


async def expand_neighbors(
    cores: List[Dict[str, Any]], loader, window: int = 1, max_seeds: int = 3,
) -> Dict[str, Any]:
    if loader is None or window <= 0 or max_seeds <= 0:
        return {"chunks": [], "errors": []}
    seeds = [item for item in cores if item.get("doc_id") and item.get("doc_version")
             and item.get("status") == "ready"][:max_seeds]
    results = await asyncio.gather(
        *[asyncio.wait_for(loader(item, window=window), timeout=5.0) for item in seeds],
        return_exceptions=True,
    )
    neighbors, errors = [], []
    for core, result in zip(seeds, results):
        if isinstance(result, BaseException):
            errors.append(type(result).__name__)
            continue
        if not isinstance(result, list):
            errors.append("InvalidNeighborResult")
            continue
        for item in result:
            if not usable_chunk(item) or not same_section(core, item):
                continue
            index, core_index = item.get("chunk_index"), core.get("chunk_index")
            if not isinstance(index, int) or not isinstance(core_index, int):
                continue
            if abs(index - core_index) > window:
                continue
            neighbor = dict(item)
            neighbor["evidence_role"] = "neighbor"
            neighbors.append(neighbor)
    return {"chunks": neighbors, "errors": errors}


def merge_windows(
    cores: List[Dict[str, Any]], neighbors: List[Dict[str, Any]], max_chars: int,
) -> List[Dict[str, Any]]:
    # 核心片段优先占预算，重叠窗口按 chunk_id 合并；不拼掉原始片段 ID。
    selected = list(cores)
    seen = {chunk_identity(item) for item in selected}
    for item in neighbors:
        key = chunk_identity(item)
        if key not in seen and evidence_chars(selected + [item]) <= max_chars:
            seen.add(key)
            selected.append(item)
    return selected


def build_context(items: List[Dict[str, Any]]) -> str:
    # 工具和 trace 共用同一份完整证据，避免评测与真实回答的上下文不一致。
    return json.dumps(items, ensure_ascii=False)
