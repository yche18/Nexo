"""Completed-request statistics drive Monitor; pending requests are not failures."""
import asyncio
from types import SimpleNamespace

import pytest

from agents import agent_orchestrator as module
from agents.agent_orchestrator import (
    AgentCallResult, AgentOrchestrator, AgentType, DeviceAgent,
    GeneralAgent, NetworkAgent, Request, TechnicalAgent,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel
from monitor.performance_monitor import PerformanceMonitor


def setup(agent_class, intent, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.delenv('NEXO_MONITOR_FALLBACK_PENALTY', raising=False)
    agent = agent_class(None, 'offline')
    router = AgentOrchestrator.__new__(AgentOrchestrator)
    router._pool = {AgentType.GENERAL: [GeneralAgent(None, 'offline')], agent.agent_type: [agent]}
    monitor = PerformanceMonitor(router, SimpleNamespace(get_stats=lambda: {}))
    req = Request(message='请继续处理', user_id='test', conv_id='test', intent=intent,
                  urgency=UrgencyLevel.LOW, intent_confidence=.9)
    return agent, router, monitor, req, clock


@pytest.mark.parametrize('agent_class,intent', [
    (TechnicalAgent, IntentCategory.TECHNICAL),
    (NetworkAgent, IntentCategory.NETWORK_ISSUE),
    (DeviceAgent, IntentCategory.DEVICE_FAILURE),
])
def test_pending_first_request_does_not_trigger_monitor_fallback(monkeypatch, agent_class, intent):
    async def scenario():
        agent, router, monitor, req, clock = setup(agent_class, intent, monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()
        async def call(request):
            started.set()
            await release.wait()
            return AgentCallResult('处理建议')
        agent._call_llm = call
        task = asyncio.create_task(agent.handle(req))
        await started.wait()
        clock[0] = 10.0
        await monitor._collect()
        assert agent.stats.total == agent.stats.success == 0
        assert agent.stats.success_rate == 1.0
        assert agent.stats.monitor_penalty == 0.0
        assert router._route_decision(req).primary_agent == agent.agent_type
        clock[0] = 15.0
        release.set()
        response = await task
        assert response.success
        assert agent.stats.total == agent.stats.success == 1
        assert agent.stats.avg_ms == 15000.0
        # Before the next sample, no stale false-failure penalty remains.
        assert router._route_decision(req).primary_agent == agent.agent_type
        await monitor._collect()
        assert agent.stats.monitor_penalty == .4  # Existing latency formula still applies.
        assert router._route_decision(req).primary_agent == agent.agent_type
    asyncio.run(scenario())


def test_completed_failure_still_counts_and_triggers_existing_fallback(monkeypatch):
    async def scenario():
        agent, router, monitor, req, clock = setup(NetworkAgent, IntentCategory.NETWORK_ISSUE, monkeypatch)
        started, release = asyncio.Event(), asyncio.Event()
        async def call(request):
            started.set()
            await release.wait()
            raise RuntimeError('failed model call')
        agent._call_llm = call
        task = asyncio.create_task(agent.handle(req))
        await started.wait()
        await monitor._collect()
        assert agent.stats.total == 0 and agent.stats.monitor_penalty == 0.0
        clock[0] = 1.0
        release.set()
        response = await task
        assert not response.success
        assert (agent.stats.total, agent.stats.success, agent.stats.total_ms) == (1, 0, 1000.0)
        await monitor._collect()
        assert agent.stats.monitor_penalty == .5
        assert router._route_decision(req).primary_agent == AgentType.GENERAL
    asyncio.run(scenario())


def test_concurrent_pending_request_does_not_dilute_completed_success(monkeypatch):
    async def scenario():
        agent, router, monitor, req, clock = setup(DeviceAgent, IntentCategory.DEVICE_FAILURE, monkeypatch)
        gates = [asyncio.Event(), asyncio.Event()]
        started = asyncio.Queue()
        async def call(request):
            index = int(request.message)
            await started.put(index)
            await gates[index].wait()
            return AgentCallResult('处理建议')
        agent._call_llm = call
        tasks = [asyncio.create_task(agent.handle(Request(message=str(i), user_id='u', conv_id=str(i))))
                 for i in range(2)]
        await started.get()
        await started.get()
        clock[0] = 1.0
        gates[0].set()
        await tasks[0]
        await monitor._collect()
        assert (agent.stats.total, agent.stats.success, agent.stats.avg_ms) == (1, 1, 1000.0)
        assert agent.stats.monitor_penalty == 0.0
        assert router._route_decision(req).primary_agent == AgentType.DEVICE
        clock[0] = 2.0
        gates[1].set()
        await tasks[1]
        assert (agent.stats.total, agent.stats.success, agent.stats.avg_ms) == (2, 2, 1500.0)
    asyncio.run(scenario())


def test_cancelled_pending_request_does_not_count_as_completed_failure(monkeypatch):
    async def scenario():
        agent, router, monitor, req, clock = setup(NetworkAgent, IntentCategory.NETWORK_ISSUE, monkeypatch)
        started = asyncio.Event()
        async def call(request):
            started.set()
            await asyncio.Event().wait()
        agent._call_llm = call
        task = asyncio.create_task(agent.handle(req))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await monitor._collect()
        assert (agent.stats.total, agent.stats.success, agent.stats.total_ms) == (0, 0, 0.0)
        assert agent.stats.monitor_penalty == 0.0
    asyncio.run(scenario())
