import hashlib
import json
import pytest
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock
from agent_sdk.sub_agents.base import SubAgentInput, SubAgentOutput, SubAgent
from agent_sdk.workspace.store import WorkspaceStore


def test_sub_agent_input_defaults():
    inp = SubAgentInput(query="What does TCS do?", entities=["TCS.NS"])
    assert inp.workspace_context == ""
    assert inp.user_profile == {}


def test_sub_agent_output_defaults():
    out = SubAgentOutput(agent_name="macro", findings="Market is bullish")
    assert out.structured == {}
    assert out.confidence == 1.0
    assert out.cached is False
    assert isinstance(out.computed_at, datetime)


def test_sub_agent_output_round_trip():
    out = SubAgentOutput(
        agent_name="fundamental",
        findings="Revenue grew 15%",
        structured={"revenue_growth": 15.0},
        confidence=0.9,
    )
    d = out.model_dump()
    restored = SubAgentOutput.model_validate(d)
    assert restored.agent_name == "fundamental"
    assert restored.structured["revenue_growth"] == 15.0


def _make_agent(name="macro", tools=None, cache_ttl=None):
    return SubAgent(
        name=name,
        model_id="azure/gpt-5-nano",
        tools=tools or [],
        system_prompt="You are a macro analyst.",
        output_schema=SubAgentOutput,
        reads_from=[],
        writes_to=name,
        cache_ttl=cache_ttl,
    )


def test_cache_key_no_entities_daily_bucket():
    agent = _make_agent(cache_ttl=21600)  # 6h → daily bucket
    inp = SubAgentInput(query="market outlook", entities=[])
    key = agent.cache_key(inp)
    assert key.startswith("sub_agent:macro:")
    parts = key.split(":")
    assert len(parts) == 4
    assert len(parts[3]) == 10  # YYYY-MM-DD


def test_cache_key_hourly_for_short_ttl():
    agent = _make_agent(cache_ttl=900)  # 15min → hourly bucket
    inp = SubAgentInput(query="price check", entities=["TCS.NS"])
    key = agent.cache_key(inp)
    parts = key.split(":")
    assert len(parts[3]) == 13  # YYYY-MM-DD-HH


def test_cache_key_entity_hash_is_order_independent():
    agent = _make_agent(cache_ttl=21600)
    inp1 = SubAgentInput(query="compare", entities=["TCS.NS", "INFY.NS"])
    inp2 = SubAgentInput(query="compare", entities=["INFY.NS", "TCS.NS"])
    assert agent.cache_key(inp1) == agent.cache_key(inp2)


async def test_run_cache_hit_skips_graph():
    agent = _make_agent(cache_ttl=3600)
    cached_output = SubAgentOutput(agent_name="macro", findings="cached", cached=False)

    store = WorkspaceStore(redis_url=None)
    await store.init()

    sub_agent_cache = AsyncMock()
    sub_agent_cache.get = AsyncMock(return_value=cached_output.model_dump())

    agent._graph = AsyncMock()  # should NOT be called

    result = await agent.run(
        inp=SubAgentInput(query="q", entities=[]),
        workspace_store=store,
        workspace_id="ws-test",
        sub_agent_cache=sub_agent_cache,
    )
    assert result.cached is True
    agent._graph.ainvoke.assert_not_called()


async def test_run_cache_miss_invokes_graph_and_writes():
    agent = _make_agent(cache_ttl=3600)

    store = WorkspaceStore(redis_url=None)
    await store.init()

    sub_agent_cache = AsyncMock()
    sub_agent_cache.get = AsyncMock(return_value=None)
    sub_agent_cache.set = AsyncMock()

    expected_output = SubAgentOutput(agent_name="macro", findings="fresh result")
    mock_graph = AsyncMock()
    mock_graph.ainvoke = AsyncMock(return_value={"findings": "fresh result", "messages": []})
    agent._graph = mock_graph
    agent._parse_output = MagicMock(return_value=expected_output)

    result = await agent.run(
        inp=SubAgentInput(query="q", entities=[]),
        workspace_store=store,
        workspace_id="ws-test",
        sub_agent_cache=sub_agent_cache,
    )
    assert result.findings == "fresh result"
    sub_agent_cache.set.assert_called_once()
    written = await store.read("ws-test", "macro")
    assert written is not None
