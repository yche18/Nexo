"""RAG 工具返回、Agent 消息和评测 trace 的接口一致性。"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agents.agent_orchestrator import Request, TechnicalAgent
from agents.tools import build_shared_rag_tools
from core.intent_recognizer import IntentCategory, UrgencyLevel
from evaluation.run_rag_eval import read_chat_evidence


class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return SimpleNamespace(content=[{
                "type": "tool_use", "id": "rag-one", "name": "search_knowledge_base",
                "input": {"query": "VPN 401", "top_k": 2},
            }])
        return SimpleNamespace(content=[{"type": "text", "text": "请检查登录状态。"}])


class FakeManager:
    def __init__(self, evidence, success=True, degraded=False):
        self.calls = []
        self.result = SimpleNamespace(
            success=success, data=evidence, reranked=False,
            error=None if success else "未检索到可用知识证据",
            retrieval={"evidence": evidence, "degraded": degraded,
                       "rerank": {"status": "disabled", "error": None}},
        )

    async def search_with_rewrite(self, tool_name, query, top_k=5):
        self.calls.append((tool_name, query, top_k))
        return self.result


def run_agent(manager):
    client = FakeClient()
    agent = TechnicalAgent(client, "test-model")
    agent.set_shared_tools(build_shared_rag_tools(manager))
    request = Request(message="VPN 401", user_id="test", conv_id="test",
                      intent=IntentCategory.TECHNICAL_LOGIN, urgency=UrgencyLevel.MEDIUM)
    response = asyncio.run(agent.handle(request))
    results = [message["content"] for message in client.calls[-1]["messages"]
               if message["role"] == "user" and isinstance(message["content"], list)]
    sent = json.loads(results[-1][0]["content"])
    evidence, calls = read_chat_evidence({"found": True, "trace": {"tool_calls": response.tool_traces}})
    return response, sent, evidence, calls


@pytest.mark.parametrize("degraded", [False, True])
def test_actual_chunks_are_sent_once_and_retained_in_trace(degraded):
    chunks = [{"chunk_id": "vpn-0", "title": "VPN", "chunk": 0,
               "content": "401 通常表示认证失败", "status": "ready"}]
    manager = FakeManager(chunks, degraded=degraded)
    response, sent, evidence, retrievals = run_agent(manager)
    assert response.success is True
    assert sent["results"] == evidence == chunks
    assert "evidence" not in sent["retrieval"]
    assert manager.result.retrieval["evidence"] == chunks
    assert retrievals[0]["degraded"] == degraded
    assert manager.calls == [("knowledge_search", "VPN 401", 2)]
    assert response.tool_traces[0]["result_success"] is True


def test_failed_retrieval_returns_no_fake_evidence_but_preserves_diagnostics():
    manager = FakeManager([], success=False, degraded=True)
    response, sent, evidence, retrievals = run_agent(manager)
    assert sent["success"] is False
    assert sent["results"] == evidence == []
    assert "未检索到可用知识证据" in sent["error"]
    assert retrievals[0]["degraded"] is True
    assert response.tool_traces[0]["result_success"] is False
    assert response.tool_traces[0]["error"] == sent["error"]
