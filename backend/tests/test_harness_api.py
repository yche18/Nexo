"""API integration without model/network calls (run in backend image with FastAPI)."""
import asyncio
from enum import Enum
import sys
from types import SimpleNamespace

import pytest
pytest.importorskip("fastapi")
from api import main
from harness.state import CaseConflictError, new_case


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"


@pytest.mark.parametrize("mode", ["off", "observe", "enforce"])
def test_chat_case_persistence_and_routing_inputs(monkeypatch, mode):
    events = []
    captured = []
    entities = {"target_system": ["SAP"]}
    old = new_case("u", "c")
    old["facts"] = {"target_system": {"value": "SAP", "source": "user_reported"}}
    class Store:
        async def load(self, user, conv):
            events.append("load")
            assert (user, conv) == ("u", "c")
            return old
        async def save(self, case, expected_version):
            events.append("save")
            assert expected_version == 0
            return {**case, "version": 1}
    class Memory:
        async def get_context(self, *args, **kwargs):
            return SimpleNamespace(recent_messages=[], to_prompt_text=lambda: "旧摘要")
        async def add_message(self, *args, **kwargs): events.append("memory")
        async def update_profile(self, *args): pass
    class Orchestrator:
        async def recognize_intent(self, message, history):
            events.append("recognize")
            assert message == "继续" and history is None
            return SimpleNamespace(entities=entities, intent=SimpleNamespace(value="query"),
                intent_group="general", urgency=None, confidence=0.9, source_scores={})
        async def run(self, req):
            captured.append(req)
            general = SimpleNamespace(value="general")
            return SimpleNamespace(case_state=req.case_state, request_id=req.request_id, response="回答",
                intent=req.intent, agent_type=general, agent_types=[general], primary_agent=general,
                supporting_agents=[], tools_used=[], routing_reason="unchanged", routing_confidence=.9,
                escalated=False, latency_ms=1, harness_events=[])
    monkeypatch.setenv("NEXO_HARNESS_MODE", mode)
    monkeypatch.setattr(main, "_orchestrator", Orchestrator())
    monkeypatch.setattr(main, "_memory", Memory())
    monkeypatch.setattr(main, "_case_store", Store())
    monkeypatch.setitem(sys.modules, "memory.conversation_memory", SimpleNamespace(MsgRole=Role))
    answer = asyncio.run(main.chat(main.ChatRequest(message="继续", user_id="u", conv_id="c")))
    assert captured[0].entities is entities
    assert answer.response == "回答"
    if mode == "enforce":
        assert events.index("recognize") < events.index("load")
        assert events.index("save") < events.index("memory")
        assert answer.case_state["version"] == 1
        assert "user_id" not in answer.case_state
    else:
        assert "load" not in events and "save" not in events
        assert answer.case_state == {}


def test_case_conflict_does_not_append_stale_chat_memory(monkeypatch):
    events = []
    class Store:
        async def load(self, *args): return new_case("u", "c")
        async def save(self, *args, **kwargs): raise CaseConflictError("stale")
    class Memory:
        async def get_context(self, *args, **kwargs):
            return SimpleNamespace(recent_messages=[], to_prompt_text=lambda: "")
        async def add_message(self, *args, **kwargs): events.append("memory")
    class Orchestrator:
        async def recognize_intent(self, *args, **kwargs):
            return SimpleNamespace(entities={}, intent=None, intent_group="general", urgency=None, confidence=1)
        async def run(self, req): return SimpleNamespace(case_state=req.case_state)
    monkeypatch.setenv("NEXO_HARNESS_MODE", "enforce")
    monkeypatch.setattr(main, "_case_store", Store())
    monkeypatch.setattr(main, "_memory", Memory())
    monkeypatch.setattr(main, "_orchestrator", Orchestrator())
    monkeypatch.setitem(sys.modules, "memory.conversation_memory", SimpleNamespace(MsgRole=Role))
    with pytest.raises(main.HTTPException) as error:
        asyncio.run(main.chat(main.ChatRequest(message="继续", user_id="u", conv_id="c")))
    assert error.value.status_code == 409
    assert events == []


@pytest.fixture
def persistent_api(monkeypatch):
    """Real API, CaseStore and MemoryManager; only external IO/model is fake."""
    import copy
    from harness.state import CaseStore
    from harness.context import build_messages
    from memory.conversation_memory import MemoryManager
    from test_harness_state import FakeRedis

    class Redis(FakeRedis):
        async def lpush(self, key, value): self.data.setdefault(key, []).insert(0, value)
        async def expire(self, key, ttl): self.expiries[key] = ttl
        async def llen(self, key): return len(self.data.get(key, []))
        async def lrange(self, key, start, stop): return self.data.get(key, [])[start:stop+1]
        async def delete(self, key): self.data.pop(key, None)
        async def setex(self, key, ttl, value):
            self.data[key] = value
            self.expiries[key] = ttl
    redis = Redis()
    memory = MemoryManager.__new__(MemoryManager)
    memory._redis = redis
    memory._model = 'offline-summary'
    summaries = []
    episodic = []
    async def summarize(**kwargs):
        summaries.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='用户正在处理办公问题。')])
    memory._client = SimpleNamespace(messages=SimpleNamespace(create=summarize))
    async def empty_history(*args): return []
    async def empty_profile(*args): return {}
    async def store_episode(*args): episodic.append(args)
    async def no_profile_update(*args): pass
    monkeypatch.setattr(memory, '_search_episodic', empty_history)
    monkeypatch.setattr(memory, '_get_profile', empty_profile)
    monkeypatch.setattr(memory, '_store_episodic', store_episode)
    monkeypatch.setattr(memory, 'update_profile', no_profile_update)
    captured = []
    class Orchestrator:
        async def recognize_intent(self, message, history):
            return SimpleNamespace(entities={}, intent=SimpleNamespace(value='query'), intent_group='general',
                                   urgency=None, confidence=.9, source_scores={})
        async def run(self, req):
            captured.append({'user': req.user_id, 'conv': req.conv_id,
                             'state': copy.deepcopy(req.case_state), 'messages': build_messages(req, '')})
            general = SimpleNamespace(value='general')
            return SimpleNamespace(case_state=req.case_state, request_id=req.request_id, response='已记录当前反馈。',
                intent=req.intent, agent_type=general, agent_types=[general], primary_agent=general,
                supporting_agents=[], tools_used=[], routing_reason='offline', routing_confidence=.9,
                escalated=False, latency_ms=1, harness_events=[])
    store = CaseStore(redis)
    monkeypatch.setenv('NEXO_HARNESS_MODE', 'enforce')
    monkeypatch.setattr(main, '_orchestrator', Orchestrator())
    monkeypatch.setattr(main, '_memory', memory)
    monkeypatch.setattr(main, '_case_store', store)
    return SimpleNamespace(store=store, memory=memory, captured=captured, summaries=summaries, episodic=episodic)


def test_api_isolates_same_conversation_across_users_and_same_user_across_conversations(persistent_api):
    import httpx
    from harness.state import apply_progress
    env = persistent_api
    async def run():
        for user, conv, system in [('u1', 'c1', 'SAP'), ('u2', 'c1', 'GitLab'), ('u1', 'c2', 'Jira')]:
            quote = '系统是' + system
            case = apply_progress(new_case(user, conv), {'facts': [
                {'field': 'target_system', 'value': system, 'source_quote': quote}]}, quote, 'seed')
            await env.store.save(case, 0)
            await env.memory.add_message(user, conv, Role.USER, '私有上下文' + system)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as http:
            responses = []
            for user, conv in [('u1', 'c1'), ('u2', 'c1'), ('u1', 'c2')]:
                response = await http.post('/chat', json={'message': '继续处理', 'user_id': user, 'conv_id': conv})
                assert response.status_code == 200
                responses.append(response.json())
        assert len({r['case_state']['case_id'] for r in responses}) == 3
        for response, captured, expected in zip(responses, env.captured, ['SAP', 'GitLab', 'Jira']):
            assert response['case_state']['facts']['target_system']['value'] == expected
            assert response['case_state']['version'] == 2
            assert 'user_id' not in response['case_state'] and 'conv_id' not in response['case_state']
            prompt = str(captured['messages'])
            assert '私有上下文' + expected in prompt
            assert all(other not in prompt for other in ['SAP', 'GitLab', 'Jira'] if other != expected)
    asyncio.run(run())


def test_api_keeps_case_facts_and_done_steps_after_actual_memory_compression(persistent_api):
    import httpx
    from harness.state import apply_progress
    env = persistent_api
    async def run():
        quote = 'VPN连接超时，我已经换过网络，还是超时'
        case = apply_progress(new_case('u', 'c'), {
            'facts': [{'field': 'target_system', 'value': 'VPN', 'source_quote': quote}],
            'steps': [{'step_id': 'network', 'description': '换过网络', 'status': 'user_reported_done',
                       'source_quote': quote, 'result': '还是超时'}]}, quote, 'seed')
        saved = await env.store.save(case, 0)
        # Real add_message threshold triggers _compress, summary write and 5-message retention.
        for index in range(15):
            await env.memory.add_message('u', 'c', Role.USER, '第%d条历史反馈' % index,
                                         metadata={'case_id': saved['case_id']})
        recent = await env.memory._get_working_memory('u', 'c')
        assert len(recent) == 5 and len(env.summaries) == 1 and len(env.episodic) == 1
        assert await env.memory._redis.get(env.memory._summary_key('u', 'c')) == '用户正在处理办公问题。'
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url='http://test') as http:
            response = await http.post('/chat', json={'message': '接下来怎么办？', 'user_id': 'u', 'conv_id': 'c'})
        assert response.status_code == 200
        actual = response.json()['case_state']
        assert actual['case_id'] == saved['case_id']
        assert actual['facts'] == saved['facts'] and actual['steps'] == saved['steps']
        prompt = str(env.captured[-1]['messages'])
        assert 'VPN' in prompt and 'user_reported_done' in prompt and '还是超时' in prompt
        assert '用户正在处理办公问题。' in prompt
    asyncio.run(run())
