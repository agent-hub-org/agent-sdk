from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


def test_phase_should_continue_executes_pending_tools_at_budget():
    from agent_sdk.agents.state import PhaseSubgraphState
    from agent_sdk.agents.subgraphs.phase_subgraph import phase_should_continue

    state = PhaseSubgraphState(
        current_phase="company_analysis",
        phase_iteration=4,
        phase_budget=4,
        phase_messages=[
            AIMessage(
                content="Need one last tool call",
                tool_calls=[{"name": "lookup", "args": {"ticker": "AAPL"}, "id": "call_1"}],
            )
        ],
    )

    assert phase_should_continue(state) == "phase_tool_node"


def test_phase_after_tool_router_stops_at_budget():
    from agent_sdk.agents.state import PhaseSubgraphState
    from agent_sdk.agents.subgraphs.phase_subgraph import phase_after_tool_router

    state = PhaseSubgraphState(
        current_phase="company_analysis",
        phase_iteration=4,
        phase_budget=4,
    )

    assert phase_after_tool_router(state) == "phase_finalize"


@pytest.mark.asyncio
async def test_phase_finalize_preserves_final_prose():
    from agent_sdk.agents.state import PhaseSubgraphState
    from agent_sdk.agents.subgraphs.phase_subgraph import phase_finalize

    state = PhaseSubgraphState(
        current_phase="risk_assessment",
        phase_findings="[calculate_risk_metrics] -> beta 1.2",
        phase_messages=[AIMessage(content="Overall risk is elevated but manageable.")],
    )

    result = await phase_finalize(None, state)

    assert "[calculate_risk_metrics] -> beta 1.2" in result["running_context"]
    assert "Overall risk is elevated but manageable." in result["running_context"]


def test_phase_input_from_parent_preserves_message_objects():
    from agent_sdk.agents.state import FinancialAnalysisState
    from agent_sdk.agents.subgraphs.phase_subgraph import _phase_input_from_parent

    system = SystemMessage(content="system")
    human = HumanMessage(content="compare AAPL and MSFT")
    state = FinancialAnalysisState(
        messages=[system, human],
        scratchpad="comparative_analysis: compare both entities",
        running_context="prior results",
        entity_focus="AAPL",
    )

    payload = _phase_input_from_parent(state, phase_name="entity_analysis", entity_focus="MSFT")

    assert payload["messages"][0] is system
    assert payload["messages"][1] is human
    assert payload["current_phase"] == "entity_analysis"
    assert payload["entity_focus"] == "MSFT"


@pytest.mark.asyncio
async def test_run_phase_subgraph_maps_back_stable_outputs():
    from agent_sdk.agents.state import FinancialAnalysisState
    from agent_sdk.agents.subgraphs.phase_subgraph import run_phase_subgraph

    phase_graph = AsyncMock()
    phase_graph.ainvoke = AsyncMock(return_value={
        "running_context": "=== COMPANY ANALYSIS ===\ncontent\n=== END COMPANY ANALYSIS ===",
        "tool_calls_log": [{"action": "tool_call", "tool": "lookup"}],
    })
    state = FinancialAnalysisState(
        messages=[HumanMessage(content="Analyse AAPL")],
        current_phase="company_analysis",
        iteration=2,
    )

    result = await run_phase_subgraph(
        agent=None,
        phase_graph=phase_graph,
        state=state,
        phase_name="company_analysis",
    )

    assert result["running_context"].startswith("=== COMPANY ANALYSIS ===")
    assert result["tool_calls_log"] == [{"action": "tool_call", "tool": "lookup"}]
    assert result["iteration"] == 3
