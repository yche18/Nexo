"""
亮点：MCP 工具调用框架

核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 查询改写（Query Rewriting）—— 用 LLM 把用户原始问题扩写成多个角度的子查询，
     再合并去重，解决"召回不全"问题。
  2. 结果重排（Reranking）—— 对融合候选用独立重排服务打分，按相关性重新排序，
     解决"召回不好/排序差"问题。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。
"""
import asyncio
import hashlib
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content, thinking_request_kwargs
from mcp.evidence import (chunk_identity, usable_chunk, select_chunks, expand_neighbors,
                          merge_windows, evidence_chars)

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排
    retrieval:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN：放行一次探测

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    supports_rerank: bool = False            # 是否支持结果重排
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    核心优化链路（针对检索类工具）：
      用户查询 → 查询改写（多角度子查询）→ 并行召回 → 结果重排 → 返回 Top-K
    """

    def __init__(self, api_key: str, base_url: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at, reranked)
        self._keyword_tool_name: Optional[str] = None
        self._reranker = None
        self._neighbor_loader = None
        self._candidate_k = 20
        self._rrf_k = 60
        self._neighbor_window = 1
        self._max_expansion_seeds = 3
        self._evidence_max_chars = 8000
        self._rag_generation = 0

    def set_rag_components(
        self, keyword_tool_name: Optional[str] = "keyword_search", reranker=None,
        neighbor_loader=None, candidate_k: int = 20, rrf_k: int = 60,
        neighbor_window: int = 1, max_expansion_seeds: int = 3,
        evidence_max_chars: int = 8000,
    ) -> None:
        self._keyword_tool_name = keyword_tool_name
        self._reranker = reranker
        self._neighbor_loader = neighbor_loader
        self._candidate_k = max(1, candidate_k)
        self._rrf_k = max(1, rrf_k)
        self._neighbor_window = max(0, neighbor_window)
        self._max_expansion_seeds = max(0, max_expansion_seeds)
        self._evidence_max_chars = max(2, evidence_max_chars)
        self.invalidate_rag_cache()

    def invalidate_rag_cache(self) -> None:
        # 配置或知识版本变化后使旧召回失效；不清理其他工具的缓存。
        self._rag_generation += 1
        for key in list(self._cache):
            if key.startswith("knowledge_search:") or (
                self._keyword_tool_name and key.startswith(self._keyword_tool_name + ":")
            ):
                del self._cache[key]


    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,          # >0 时对结果重排，取 Top-K
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 可选重排 → 缓存写入
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        cache_generation = self._rag_generation
        cache_rerank_top_k = rerank_top_k if rerank_top_k > 0 and tool.supports_rerank else 0

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params, cache_rerank_top_k)
            if cached is not None:
                cached_data, cached_reranked = cached
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(
                    success=True,
                    data=cached_data,
                    tool_name=name,
                    cached=True,
                    reranked=cached_reranked,
                )

        # 熔断检查
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(self._run_handler(tool, params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

            # 重排（针对返回列表的检索工具）
            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = params.get("query", "")
                ranked = await self._rank_candidates(query, data)
                data = ranked["items"][:rerank_top_k]
                reranked = ranked["status"] == "success"

            # 写缓存：缓存最终返回结果，避免下次命中未重排的原始结果。
            if tool.cache_ttl > 0 and cache_generation == self._rag_generation:
                self._set_cache(name, params, data, tool.cache_ttl, cache_rerank_top_k, reranked)

            return ToolResult(success=True, data=data, tool_name=name,
                              latency_ms=latency, reranked=reranked)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    async def _run_handler(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Any:
        """
        执行工具 handler。

        优先支持 async handler；如果历史工具仍是同步函数，则放入线程池执行，
        避免阻塞事件循环。
        """
        if inspect.iscoroutinefunction(tool.handler):
            return await tool.handler(params, context)
        result = await asyncio.to_thread(tool.handler, params, context)
        if inspect.isawaitable(result):
            return await result
        return result

    # ── 查询改写（解决召回不全）────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "退款流程"
          改写: ["如何申请退款", "退款需要多少天", "退款政策是什么"]
        """
        prompt = f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
                **thinking_request_kwargs(),
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            # 原始查询也保留，去重
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """改写向量召回 + 原问题 BM25 → 两级 RRF → 重排 → 完整证据与有限扩窗。"""
        t0 = time.monotonic()
        generation = self._rag_generation
        if top_k <= 0:
            return ToolResult(success=False, data=[], tool_name=tool_name, error="top_k 必须大于 0")
        sub_queries = await self.rewrite_query(query, n=3)
        # 限制改写数量，且始终保留原始问题。改写模型异常格式不影响主问题检索。
        sub_queries = list(dict.fromkeys(
            [query] + [q for q in sub_queries if isinstance(q, str) and q.strip()]
        ))[:4]
        tasks = [self.call(tool_name, {"query": q, "top_k": self._candidate_k}, context)
                 for q in sub_queries]
        keyword_enabled = bool(self._keyword_tool_name)
        if keyword_enabled:
            tasks.append(self.call(self._keyword_tool_name,
                                   {"query": query, "top_k": self._candidate_k}, context))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dense_lists, dense_status = [], []
        for result in results[:len(sub_queries)]:
            chunks, state = self._retrieval_channel(result)
            dense_lists.append(chunks)
            dense_status.append(state)
        keyword_items, keyword_state = [], {"status": "disabled", "count": 0}
        if keyword_enabled:
            keyword_items, keyword_state = self._retrieval_channel(results[-1])
        # 先将多条 dense 查询变成一票，再和 BM25 等权融合，避免 dense 四票压过一票。
        dense_items = self._rrf(dense_lists, self._candidate_k)
        candidates = self._rrf([dense_items, keyword_items], self._candidate_k)
        ranked = await self._rank_candidates(query, candidates)
        cores = select_chunks(ranked["items"], top_k, self._evidence_max_chars)
        expanded = await expand_neighbors(cores, self._neighbor_loader,
                                          self._neighbor_window, self._max_expansion_seeds)
        evidence = merge_windows(cores, expanded["chunks"], self._evidence_max_chars)
        channel_failed = any(state["status"] == "failed" for state in dense_status)
        channel_failed = channel_failed or keyword_state["status"] == "failed"
        retrieval = {
            "queries": sub_queries,
            "channels": {"dense": dense_status, "keyword": keyword_state},
            "candidate_count": len(candidates),
            "rerank": {"status": ranked["status"], "error": ranked.get("error")},
            "neighbor_errors": expanded["errors"],
            "degraded": channel_failed or ranked["status"] == "failed" or bool(expanded["errors"]),
            "evidence": evidence,
            "evidence_chars": evidence_chars(evidence),
            "budget_chars": self._evidence_max_chars,
            "generation": self._rag_generation,
            "elapsed_ms": round((time.monotonic() - t0) * 1000, 1),
        }
        if generation != self._rag_generation:
            # 该请求可能同时读到导入前后的片段，不向 Agent 交付混合版本证据。
            retrieval.update(evidence=[], evidence_chars=2, degraded=True,
                             version_changed=True, started_generation=generation)
            return ToolResult(
                success=False, data=[], tool_name=tool_name,
                error="知识库在检索期间已更新，请重试",
                retrieval=retrieval, latency_ms=retrieval["elapsed_ms"],
            )
        return ToolResult(
            success=bool(evidence), data=evidence, tool_name=tool_name,
            error=None if evidence else "未检索到可用知识证据",
            reranked=ranked["status"] == "success", retrieval=retrieval,
            latency_ms=retrieval["elapsed_ms"],
        )

    @staticmethod
    def _retrieval_channel(result) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not isinstance(result, ToolResult):
            return [], {"status": "failed", "count": 0, "error": type(result).__name__}
        # fallback 即使 success=True，也只是降级说明，不能当作知识库证据。
        if not result.success or result.error or not isinstance(result.data, list):
            return [], {"status": "failed", "count": 0, "error": result.error or "检索返回格式错误"}
        chunks = [dict(item) for item in result.data if usable_chunk(item)]
        state = "success" if chunks else "empty"
        if any(isinstance(item, dict) and item.get("fallback") for item in result.data):
            state = "failed"
        return chunks, {"status": state, "count": len(chunks), "cached": result.cached}

    def _rrf(self, lists: List[List[Dict[str, Any]]], top_k: int) -> List[Dict[str, Any]]:
        scores, items = {}, {}
        for rows in lists:
            seen = set()
            rank = 0
            for item in rows:
                if not usable_chunk(item):
                    continue
                key = chunk_identity(item)
                if key in seen:
                    continue
                seen.add(key)
                rank += 1
                scores[key] = scores.get(key, 0.0) + 1.0 / (self._rrf_k + rank)
                if key not in items:
                    items[key] = dict(item)
        order = sorted(scores, key=lambda key: scores[key], reverse=True)
        result = []
        for key in order[:top_k]:
            item = dict(items[key])
            item["rrf_score"] = scores[key]
            result.append(item)
        return result

    async def _rank_candidates(self, query: str, items: List[Any]) -> Dict[str, Any]:
        if self._reranker is None:
            return {"items": items, "status": "disabled", "error": None}
        try:
            return await self._reranker.rerank(query, items)
        except Exception as ex:
            return {"items": items, "status": "failed", "error": type(ex).__name__}

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """保留原内部入口；重排失败保留候选顺序，状态由调用方单独记录。"""
        ranked = await self._rank_candidates(query, items)
        return ranked["items"][:top_k]

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict, rerank_top_k: int = 0) -> str:
        payload = {"params": params, "rerank_top_k": rerank_top_k}
        if name == "knowledge_search" or name == self._keyword_tool_name:
            payload["rag_generation"] = self._rag_generation
        return f"{name}:{hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict, rerank_top_k: int = 0) -> Optional[Tuple[Any, bool]]:
        key = self._cache_key(name, params, rerank_top_k)
        if key in self._cache:
            data, expire_at, reranked = self._cache[key]
            if time.monotonic() < expire_at:
                return data, reranked
            del self._cache[key]
        return None

    def _set_cache(
        self,
        name: str,
        params: Dict,
        data: Any,
        ttl: float,
        rerank_top_k: int = 0,
        reranked: bool = False,
    ) -> None:
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[self._cache_key(name, params, rerank_top_k)] = (data, time.monotonic() + ttl, reranked)

    # ── 参数校验 ──────────────────────────────────────────────────────────────

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，不合法时抛出 ValueError。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    if not isinstance(value, self._TYPE_MAP[expected_type]):
                        raise ValueError(
                            f"工具 {tool.name} 参数 {key} 类型错误: 期望 {expected_type}，实际 {type(value).__name__}"
                        )

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
