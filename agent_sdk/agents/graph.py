from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

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
    orchestrate,
    load_user_context,
    memory_writer,
)

def merge_context(state) -> dict:
    """Dummy node to fan-in parallel branches."""
    return {}


def create_graph(agent, checkpointer: Optional[Any] = None):
    """
    Build a LangGraph-based autonomous agent graph with optional persistence.

    Standard mode: pure ReAct loop. The `orchestrate` node writes a tool-specific
    plan to scratchpad once; every subsequent `llm_call` iteration sees the plan
    (via scratchpad) and the completed work (via running_context), so the LLM
    self-regulates without re-planning or making extra tool calls.

    Graph flow
    ----------
    START → initialize → load_user_context → orchestrate → pre_llm_router
        → llm_call → should_continue → tool_node → post_tool_router → llm_call
                                     → summarize_conversation → llm_call
                                     → memory_writer → END
    """

    graph = StateGraph(AgentState)

    graph.add_node("initialize", initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("orchestrate", partial(orchestrate, agent))
    graph.add_node("llm_call", partial(llm_call, agent))
    graph.add_node("tool_node", partial(tool_node, agent))
    graph.add_node("summarize_conversation", partial(summarize_conversation, agent))
    graph.add_node("memory_writer", partial(memory_writer, agent))
    graph.add_node("merge_context", merge_context)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_edge("initialize", "orchestrate")
    graph.add_edge(["load_user_context", "orchestrate"], "merge_context")
    graph.add_conditional_edges("merge_context", pre_llm_router, {
        "llm_call": "llm_call",
        "summarize_conversation": "summarize_conversation",
    })

    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "summarize_conversation": "summarize_conversation",
        END: "memory_writer",
    })
    graph.add_conditional_edges("tool_node", post_tool_router)
    graph.add_edge("summarize_conversation", "llm_call")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer)


def create_financial_reasoning_graph(agent, checkpointer: Optional[Any] = None):
    """
    Build the financial reasoning cognitive pipeline.

    All phases use `financial_phase_executor` — a mini ReAct loop with native
    tool binding. Every LLM call within a phase sees the full plan (scratchpad)
    and completed work (running_context), so the LLM self-regulates without
    extra planning calls or redundant tool calls.

    Parallelism: sector_analysis and company_analysis run concurrently (Send)
    when both are in phases_to_run after causal_analysis completes.

    Graph flow
    ----------
    START → initialize → load_user_context → financial_orchestrate → phase_router
        → regime_assessment     → phase_advance
        → causal_analysis       → [Send sector+company in parallel | phase_advance]
        → sector_analysis       → phase_advance  ┐ (parallel, merge via fan-in)
        → company_analysis      → phase_advance  ┘
        → comparative_analysis  → phase_advance
        → risk_assessment       → phase_advance
        → synthesis (no tools)  → memory_writer → END
    phase_advance → phase_router (loop)
    """
    from agent_sdk.agents.nodes import (
        financial_initialize,
        financial_orchestrate,
        financial_phase_executor,
        phase_router,
        phase_advance,
        parallel_fan_in,
        comparative_analysis_node,
        synthesis_node,
    )

    graph = StateGraph(FinancialAnalysisState)

    # --- Core nodes ---
    graph.add_node("initialize", financial_initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("financial_orchestrate", partial(financial_orchestrate, agent))
    graph.add_node("phase_router", phase_router)
    graph.add_node("phase_advance", phase_advance)
    graph.add_node("memory_writer", partial(memory_writer, agent))

    # --- Phase executor nodes (one per phase, each bound to its phase name) ---
    _tool_phases = [
        "regime_assessment",
        "causal_analysis",
        "sector_analysis",
        "company_analysis",
        "risk_assessment",
    ]
    for ph in _tool_phases:
        graph.add_node(ph, partial(financial_phase_executor, ph, agent))

    # --- Non-tool phases ---
    graph.add_node("comparative_analysis", partial(comparative_analysis_node, agent))
    graph.add_node("synthesis", partial(synthesis_node, agent))

    # --- Edges ---
    graph.add_node("merge_context", merge_context)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_edge("initialize", "financial_orchestrate")
    graph.add_edge(["load_user_context", "financial_orchestrate"], "merge_context")
    graph.add_edge("merge_context", "phase_router")

    # Phase router → entry node of each phase
    _phase_route_map = {
        "regime_assessment":    "regime_assessment",
        "causal_analysis":      "causal_analysis",
        "sector_analysis":      "sector_analysis",
        "company_analysis":     "company_analysis",
        "comparative_analysis": "comparative_analysis",
        "risk_assessment":      "risk_assessment",
        "synthesis":            "synthesis",
        END:                    END,
    }
    graph.add_conditional_edges("phase_router", _route_phase, _phase_route_map)

    # regime_assessment → phase_advance (sequential)
    graph.add_edge("regime_assessment", "phase_advance")

    # causal_analysis → parallel fan-out (sector + company) or sequential phase_advance
    graph.add_conditional_edges("causal_analysis", _causal_analysis_router)

    # sector_analysis and company_analysis → parallel_fan_in (handles both parallel and sequential cases)
    graph.add_node("parallel_fan_in", parallel_fan_in)
    graph.add_edge("sector_analysis", "parallel_fan_in")
    graph.add_edge("company_analysis", "parallel_fan_in")
    graph.add_edge("parallel_fan_in", "phase_router")

    # risk_assessment + comparative_analysis → phase_advance
    graph.add_edge("risk_assessment", "phase_advance")
    graph.add_edge("comparative_analysis", "phase_advance")

    # phase_advance → phase_router (loop)
    graph.add_edge("phase_advance", "phase_router")

    # synthesis → memory_writer → END
    graph.add_edge("synthesis", "memory_writer")
    graph.add_edge("memory_writer", END)

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
        "regime_assessment":    "regime_assessment",
        "causal_analysis":      "causal_analysis",
        "sector_analysis":      "sector_analysis",
        "company_analysis":     "company_analysis",
        "comparative_analysis": "comparative_analysis",
        "risk_assessment":      "risk_assessment",
        "synthesis":            "synthesis",
    }
    target = phase_to_node.get(next_phase, END)
    logger.info("Routing to phase: %s (node: %s)", next_phase, target)
    return target


def _causal_analysis_router(state: FinancialAnalysisState):
    """
    After causal_analysis: if both sector_analysis and company_analysis are pending,
    fan them out in parallel via Send(). Otherwise advance sequentially.
    """
    phases = state.phases_to_run
    has_sector = "sector_analysis" in phases
    has_company = "company_analysis" in phases

    if has_sector and has_company:
        logger.info("Fanning out sector_analysis + company_analysis in parallel")
        return [Send("sector_analysis", state), Send("company_analysis", state)]

    logger.info("causal_analysis → phase_advance (sequential, only one of sector/company in plan)")
    return "phase_advance"
