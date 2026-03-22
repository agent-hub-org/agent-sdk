from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END

from agent_sdk.agents.state import AgentState, FinancialAnalysisState

logger = logging.getLogger("agent_sdk.graph")
from agent_sdk.agents.nodes import (
    initialize,
    llm_call,
    tool_node,
    summarize_conversation,
    should_continue,
    post_tool_router,
    pre_llm_router,
)


def create_graph(agent, checkpointer: Optional[Any] = None):
    """
    Build a LangGraph-based autonomous agent graph with optional persistence.

    Dependencies (`llm`, `tools_by_name`, `summarizer`) live on the `agent`
    instance and are bound to node functions via `functools.partial`.
    This keeps AgentState free of non-serializable objects.

    Persistence & short-term memory
    --------------------------------
    - Short-term conversational memory is handled by:
      - `AgentState.messages` being annotated with `add_messages` so messages
        accumulate across steps and invocations.
      - A LangGraph checkpointer (e.g. `InMemorySaver`) passed as `checkpointer`
        and a `thread_id` in the `config` when invoking the graph.
    - `summary` is preserved across invocations because only
      `{"messages": [...]}` is passed to `ainvoke`, so the checkpointed
      summary is never overwritten.

    Graph flow
    ----------
    START → initialize → llm_call → should_continue → tool_node → post_tool_router → llm_call
                                                     │                              → summarize_conversation → llm_call
                                                     → summarize_conversation → llm_call
                                                     → END
    """

    graph = StateGraph(AgentState)

    # initialize and should_continue don't need agent — register directly
    graph.add_node("initialize", initialize)
    graph.add_node("llm_call", partial(llm_call, agent))
    graph.add_node("tool_node", partial(tool_node, agent))
    graph.add_node("summarize_conversation", partial(summarize_conversation, agent))

    graph.add_edge(START, "initialize")
    graph.add_conditional_edges("initialize", pre_llm_router)
    graph.add_conditional_edges("llm_call", should_continue)
    graph.add_conditional_edges("tool_node", post_tool_router)
    graph.add_edge("summarize_conversation", "llm_call")

    return graph.compile(checkpointer=checkpointer)


def create_financial_reasoning_graph(agent, checkpointer: Optional[Any] = None):
    """
    Build the financial reasoning cognitive pipeline.

    This is a hierarchical reasoning graph that replaces the flat
    llm_call → tools → llm_call loop with structured analytical phases:

        query_classifier → regime_assessment → causal_analysis →
        sector_analysis → company_analysis → risk_assessment → synthesis

    Each phase:
    - Has a specific analytical lens (system prompt)
    - Has access to only the tools relevant to that phase
    - Writes structured findings to typed state fields
    - Can read structured findings from prior phases

    The query classifier at the front determines which phases to activate,
    so simple queries skip unnecessary phases.

    Graph flow
    ----------
    START → initialize → classify_query → phase_router
        → regime_assessment_phase → phase_advance → phase_router
        → causal_analysis_phase   → phase_advance → phase_router
        → sector_analysis_phase   → phase_advance → phase_router
        → company_analysis_phase  → phase_advance → phase_router
        → risk_assessment_phase   → phase_advance → phase_router
        → synthesis_phase         → END

    Each *_phase node is itself a sub-loop:
        phase_llm_call → phase_should_continue → phase_tool_node → phase_llm_call
                                               → phase_complete (write structured output)
    """
    from agent_sdk.agents.nodes import (
        financial_initialize,
        classify_query_node,
        phase_router,
        regime_assessment_node,
        causal_analysis_node,
        sector_analysis_node,
        company_analysis_node,
        risk_assessment_node,
        synthesis_node,
        phase_advance,
        financial_tool_node,
        financial_should_continue,
        summarize_conversation,
    )

    graph = StateGraph(FinancialAnalysisState)

    # --- Core nodes ---
    graph.add_node("initialize", financial_initialize)
    graph.add_node("classify_query", partial(classify_query_node, agent))
    graph.add_node("phase_router", phase_router)

    # --- Phase nodes (each runs the LLM with phase-specific prompt + tools) ---
    graph.add_node("regime_assessment", partial(regime_assessment_node, agent))
    graph.add_node("causal_analysis", partial(causal_analysis_node, agent))
    graph.add_node("sector_analysis", partial(sector_analysis_node, agent))
    graph.add_node("company_analysis", partial(company_analysis_node, agent))
    graph.add_node("risk_assessment", partial(risk_assessment_node, agent))
    graph.add_node("synthesis", partial(synthesis_node, agent))

    # --- Shared utility nodes ---
    graph.add_node("phase_advance", phase_advance)
    graph.add_node("financial_tool_node", partial(financial_tool_node, agent))
    graph.add_node("summarize_conversation", partial(summarize_conversation, agent))

    # --- Edges ---
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "classify_query")
    graph.add_edge("classify_query", "phase_router")

    # Phase router dispatches to the next phase or END
    _phase_route_map = {
        "regime_assessment": "regime_assessment",
        "causal_analysis": "causal_analysis",
        "sector_analysis": "sector_analysis",
        "company_analysis": "company_analysis",
        "risk_assessment": "risk_assessment",
        "synthesis": "synthesis",
        END: END,
    }
    graph.add_conditional_edges("phase_router", _route_phase, _phase_route_map)

    # Each phase node → check if tools needed or phase complete
    _phase_continue_map = {
        "financial_tool_node": "financial_tool_node",
        "phase_advance": "phase_advance",
    }
    for phase_name in ["regime_assessment", "causal_analysis", "sector_analysis",
                       "company_analysis", "risk_assessment", "synthesis"]:
        graph.add_conditional_edges(
            phase_name, partial(financial_should_continue, phase_name), _phase_continue_map
        )

    # Tool node → back to current phase
    _back_to_phase_map = {
        "regime_assessment": "regime_assessment",
        "causal_analysis": "causal_analysis",
        "sector_analysis": "sector_analysis",
        "company_analysis": "company_analysis",
        "risk_assessment": "risk_assessment",
        "synthesis": "synthesis",
    }
    graph.add_conditional_edges("financial_tool_node", _route_back_to_phase, _back_to_phase_map)

    # Phase advance → back to router for next phase
    graph.add_edge("phase_advance", "phase_router")

    # Summarize → back to current phase
    graph.add_conditional_edges("summarize_conversation", _route_back_to_phase, _back_to_phase_map)

    return graph.compile(checkpointer=checkpointer)


def _route_phase(state: FinancialAnalysisState) -> str:
    """Route to the next phase in the pipeline, or END if all phases done."""
    logger.info("_route_phase — phases_to_run=%s, current_phase=%s",
                state.phases_to_run, state.current_phase)
    if not state.phases_to_run:
        logger.info("No phases remaining, routing to END")
        return END

    next_phase = state.phases_to_run[0]
    phase_to_node = {
        "regime_assessment": "regime_assessment",
        "causal_analysis": "causal_analysis",
        "sector_analysis": "sector_analysis",
        "company_analysis": "company_analysis",
        "risk_assessment": "risk_assessment",
        "synthesis": "synthesis",
    }
    target = phase_to_node.get(next_phase, END)
    logger.info("Routing to phase: %s (node: %s)", next_phase, target)
    return target


def _route_back_to_phase(state: FinancialAnalysisState) -> str:
    """Route back to the current active phase after tool execution."""
    logger.info("_route_back_to_phase — current_phase=%s", state.current_phase)
    return state.current_phase