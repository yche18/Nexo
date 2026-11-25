"""独立重排服务；失败时保留融合顺序，不再追加一次聊天模型调用。"""
import asyncio
import logging
import math
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class HTTPReranker:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "BAAI/bge-reranker-v2-m3",
        timeout_s: float = 8.0,
        client=None,
    ):
        self.url = (url or "").strip()
        self.model = model
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=timeout_s)
        self._owns_client = client is None

    async def rerank(self, query: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.url:
            return {"items": items, "status": "disabled", "error": None}
        if not items:
            return {"items": [], "status": "empty", "error": None}
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        documents = [f"{item.get('title', '')}\n{item.get('content', '')}" for item in items]
        try:
            response = await asyncio.wait_for(
                self._client.post(
                    self.url,
                    headers=headers,
                    json={"model": self.model, "query": query,
                          "documents": documents, "top_n": len(documents),
                          "return_documents": False},
                    timeout=self.timeout_s,
                ),
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            rows = response.json().get("results")
            if not isinstance(rows, list) or len(rows) != len(items):
                raise ValueError("重排服务未返回完整候选列表")
            seen, ranked = set(), []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("重排结果格式错误")
                index = row.get("index")
                score = row.get("relevance_score")
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError("重排索引不是整数")
                if index < 0 or index >= len(items) or index in seen:
                    raise ValueError("重排索引越界或重复")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError("重排分数无效")
                seen.add(index)
                item = dict(items[index])
                item["rerank_score"] = float(score)
                ranked.append(item)
            ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
            return {"items": ranked, "status": "success", "error": None}
        except Exception as ex:
            # 不输出请求头或密钥；保留 RRF 结果供上层降级。
            error = type(ex).__name__
            logger.warning("重排服务不可用，保留融合顺序: %s", error)
            return {"items": items, "status": "failed", "error": error}

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
