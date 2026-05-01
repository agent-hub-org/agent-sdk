import pytest
from langchain_core.messages import HumanMessage


def _make_state(template="company_snapshot", agents_done=None, populated=None):
    from agent_sdk.agents.state import FinancialAnalysisState
    return FinancialAnalysisState(
        messages=[HumanMessage(content="test")],
        current_template=template,
        entities=["TCS.NS"],
        workspace_id="ws-test",
        agents_completed=agents_done or set(),
        workspace_populated=populated or set(),
    )


def test_dispatcher_routes_to_company_profiling_first():
    from agent_sdk.agents.graph import _route_from_dispatcher
    from langgraph.types import Send
    state = _make_state("company_snapshot")
    result = _route_from_dispatcher(state)
    assert isinstance(result, list)
    assert any(isinstance(r, Send) and r.node == "company_profiling" for r in result)


def test_dispatcher_routes_to_synthesis_when_all_done():
    from agent_sdk.agents.graph import _route_from_dispatcher
    state = _make_state(
        "company_snapshot",
        agents_done={"company_profiling"},
        populated={"company_profiling"},
    )
    result = _route_from_dispatcher(state)
    assert result == "synthesis_node"


def test_dispatcher_macro_query_dispatches_only_macro():
    from agent_sdk.agents.graph import _route_from_dispatcher
    from langgraph.types import Send
    state = _make_state("macro_query")
    result = _route_from_dispatcher(state)
    assert len(result) == 1
    assert result[0].node == "macro"


def test_dispatcher_buy_decision_round1_parallel():
    from agent_sdk.agents.graph import _route_from_dispatcher
    from langgraph.types import Send
    state = _make_state("buy_decision")
    result = _route_from_dispatcher(state)
    dispatched = {r.node for r in result if isinstance(r, Send)}
    # Round 1: agents with no deps or only user_profile dep (always available)
    assert "macro" in dispatched
    assert "company_profiling" in dispatched
    assert "news_sentiment" not in dispatched  # reads_from=["company_profiling"]
    # Round 2 agents should NOT be dispatched yet
    assert "fundamental" not in dispatched
    assert "technical" not in dispatched
