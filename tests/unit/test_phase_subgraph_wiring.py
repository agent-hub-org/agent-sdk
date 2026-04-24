"""
Integration test for the compiled phase subgraph wiring.

Exercises the full phase_init → phase_llm_call → phase_tool_node →
phase_after_tool_router → phase_llm_call → phase_finalize path using a mock
agent and a controlled sequence of LLM responses. Catches wiring bugs (wrong
edge targets, wrong node signatures) that unit tests of individual nodes cannot.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _make_mock_agent(tool_results: list[str] | None = None):
    """Build a minimal mock agent compatible with phase subgraph node signatures."""
    tool = MagicMock()
    tool.name = "mock_tool"
    tool.ainvoke = AsyncMock(return_value="tool output")

    agent = MagicMock()
    agent.tools_by_name = {"mock_tool": tool}
    agent.get_bound_llm = lambda llm, tools: llm
    agent.get_available_tools = lambda phase_tools=None: phase_tools or []
    agent._get_breaker = MagicMock(return_value=MagicMock(is_open=False, record_success=MagicMock(), record_failure=MagicMock()))
    agent._mcp_manager = None
    agent._phase_tools_cache = {}
    return agent, tool


@pytest.mark.asyncio
async def test_phase_subgraph_tool_call_then_prose():
    """
    Full wiring test: LLM calls one tool, then produces final prose.

    Expected path:
        phase_init → phase_llm_call (tool_calls) → phase_tool_node →
        phase_after_tool_router → phase_llm_call (prose) → phase_finalize
    """
    from agent_sdk.agents.subgraphs.phase_subgraph import create_phase_subgraph
    from agent_sdk.agents.state import PhaseSubgraphState

    agent, tool = _make_mock_agent()

    # First LLM response: one tool call
    ai_with_tool = AIMessage(
        content="",
        tool_calls=[{"name": "mock_tool", "args": {"q": "AAPL"}, "id": "call_1"}],
    )
    # Second LLM response: final prose
    ai_final = AIMessage(content="Company analysis complete. Revenue grew 12% YoY.")

    llm_mock = AsyncMock()
    llm_mock.ainvoke = AsyncMock(side_effect=[ai_with_tool, ai_final])
    # get_bound_llm returns the llm directly, so mock _invoke_with_retry at call site
    agent.llm = llm_mock
    agent.get_bound_llm = lambda llm, tools: llm

    with patch("agent_sdk.agents.subgraphs.phase_subgraph._get_phase_llm", return_value=llm_mock), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._get_phase_tools", return_value=[tool]), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._invoke_with_retry", side_effect=[ai_with_tool, ai_final]), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._execute_tool_calls",
               new_callable=AsyncMock,
               return_value=[ToolMessage(content="Revenue: $394B", tool_call_id="call_1")]):

        subgraph = create_phase_subgraph(agent).compile()

        initial_state = PhaseSubgraphState(
            parent_system_text="You are a financial analyst.",
            parent_user_message=HumanMessage(content="Analyze AAPL"),
            scratchpad="company_analysis: retrieve AAPL fundamentals",
            running_context="=== REGIME ASSESSMENT ===\nBull market\n=== END ===",
            current_phase="company_analysis",
            phase_budget=4,
        )

        result = await subgraph.ainvoke(initial_state)

    # phase_findings accumulates tool results during execution (via phase_tool_node)
    assert "Revenue: $394B" in result["phase_findings"]
    # running_context is the labeled phase block built by phase_finalize,
    # combining tool findings AND final prose
    assert "=== COMPANY ANALYSIS ===" in result["running_context"]
    assert "=== END COMPANY ANALYSIS ===" in result["running_context"]
    assert "Revenue: $394B" in result["running_context"]
    assert "Company analysis complete" in result["running_context"]
    # Phase iteration advanced past 0 (one tool call turn happened)
    assert result["phase_iteration"] >= 1
    # Tool call log populated
    assert len(result["tool_calls_log"]) >= 2  # one tool_call + one tool_result entry


@pytest.mark.asyncio
async def test_phase_subgraph_no_tool_calls_goes_direct_to_finalize():
    """
    LLM produces final prose immediately with no tool calls.

    Expected path: phase_init → phase_llm_call (prose) → phase_finalize
    """
    from agent_sdk.agents.subgraphs.phase_subgraph import create_phase_subgraph
    from agent_sdk.agents.state import PhaseSubgraphState

    agent, _ = _make_mock_agent()
    ai_final = AIMessage(content="Market regime is bullish based on macro indicators.")

    with patch("agent_sdk.agents.subgraphs.phase_subgraph._get_phase_llm", return_value=MagicMock()), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._get_phase_tools", return_value=[]), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._invoke_with_retry", return_value=ai_final):

        subgraph = create_phase_subgraph(agent).compile()

        result = await subgraph.ainvoke(PhaseSubgraphState(
            parent_system_text="You are a financial analyst.",
            parent_user_message=HumanMessage(content="Assess the market regime"),
            current_phase="regime_assessment",
        ))

    assert "Market regime is bullish" in result["running_context"]
    assert result["phase_iteration"] == 1


@pytest.mark.asyncio
async def test_phase_subgraph_budget_exhaustion_routes_to_finalize():
    """
    When phase_iteration reaches phase_budget after tool execution,
    phase_after_tool_router must route to phase_finalize (not phase_llm_call).
    """
    from agent_sdk.agents.subgraphs.phase_subgraph import create_phase_subgraph
    from agent_sdk.agents.state import PhaseSubgraphState

    agent, tool = _make_mock_agent()

    call_count = 0

    async def controlled_invoke(llm_bound, prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        return AIMessage(
            content="",
            tool_calls=[{"name": "mock_tool", "args": {}, "id": f"call_{call_count}"}],
        )

    # Patch the budget dict to enforce a budget of 2 for this test
    with patch.dict("agent_sdk.agents.subgraphs.phase_subgraph._PHASE_BUDGETS", {"company_analysis": 2}), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._get_phase_llm", return_value=MagicMock()), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._get_phase_tools", return_value=[tool]), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._invoke_with_retry", side_effect=controlled_invoke), \
         patch("agent_sdk.agents.subgraphs.phase_subgraph._execute_tool_calls",
               new_callable=AsyncMock,
               return_value=[ToolMessage(content="data", tool_call_id="call_x")]):

        result = await create_phase_subgraph(agent).compile().ainvoke(
            PhaseSubgraphState(
                parent_system_text="sys",
                parent_user_message=HumanMessage(content="q"),
                current_phase="company_analysis",
            )
        )

    # Should have stopped at budget=2, not looped indefinitely
    assert result["phase_iteration"] == 2
    # running_context should have the phase block (may be "(no data retrieved)" since no prose)
    assert "=== COMPANY ANALYSIS ===" in result["running_context"]
