from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_sdk.agents.state import AgentState, FinancialAnalysisState
from agent_sdk.agents.subgraphs import create_react_subgraph, create_phase_subgraph
from agent_sdk.agents.subgraphs.phase_subgraph import run_phase_subgraph

logger = logging.getLogger("agent_sdk.graph")
from agent_sdk.agents.nodes import (
    initialize,
    orchestrate,
    load_user_context,
    memory_writer,
)


def merge_context(state) -> dict:
    """Dummy node to fan-in parallel branches."""
    return {}


# ============================================================================
# STANDARD MODE
# ============================================================================

def create_graph(agent, checkpointer: Optional[Any] = None):
    """
    Standard mode: orchestrate once, then delegate to reusable ReAct subgraph.

    Graph flow
    ----------
    START -> initialize -> load_user_context -> orchestrate -> merge_context
        -> react_loop (subgraph: llm_call <-> tool_node with checkpointing)
        -> memory_writer -> END
    """
    graph = StateGraph(AgentState)
    react = create_react_subgraph(agent).compile()

    graph.add_node("initialize", initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("orchestrate", partial(orchestrate, agent))
    graph.add_node("react_loop", react)
    graph.add_node("memory_writer", partial(memory_writer, agent))
    graph.add_node("merge_context", merge_context)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_edge("initialize", "orchestrate")
    graph.add_edge(["load_user_context", "orchestrate"], "merge_context")
    graph.add_edge("merge_context", "react_loop")
    graph.add_edge("react_loop", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# FINANCIAL REASONING PIPELINE
# ============================================================================

def create_financial_reasoning_graph(agent, checkpointer: Optional[Any] = None):
    """
    Financial pipeline with phase-level subgraphs.

    Each tool phase is a compiled PhaseSubgraph with per-iteration checkpointing.
    Comparative analysis fans out to parallel entity_analysis subgraphs via Send().
    """
    from agent_sdk.agents.nodes import (
        financial_initialize,
        financial_orchestrate,
        phase_router,
        phase_advance,
        parallel_fan_in,
        synthesis_node,
    )

    graph = StateGraph(FinancialAnalysisState)

    # Compile the phase subgraph once and wrap it per phase so branch-local
    # phase buffers never write directly into the parent graph state.
    phase_subgraph = create_phase_subgraph(agent).compile()

    # --- Core nodes ---
    graph.add_node("initialize", financial_initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("financial_orchestrate", partial(financial_orchestrate, agent))
    graph.add_node("phase_router", phase_router)
    graph.add_node("phase_advance", phase_advance)
    graph.add_node("memory_writer", partial(memory_writer, agent))

    # --- Phase subgraph nodes (one compiled subgraph, many node names) ---
    _tool_phases = [
        "regime_assessment",
        "causal_analysis",
        "sector_analysis",
        "company_analysis",
        "risk_assessment",
    ]
    for ph in _tool_phases:
        graph.add_node(ph, partial(run_phase_subgraph, agent, phase_subgraph, phase_name=ph))

    # Comparative analysis uses the same phase subgraph with entity_focus
    graph.add_node(
        "entity_analysis",
        partial(run_phase_subgraph, agent, phase_subgraph, phase_name="entity_analysis"),
    )

    # --- Non-tool phases ---
    graph.add_node("synthesis", partial(synthesis_node, agent))

    # --- Edges ---
    graph.add_node("merge_context", merge_context)
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_edge("initialize", "financial_orchestrate")
    graph.add_edge(["load_user_context", "financial_orchestrate"], "merge_context")
    graph.add_edge("merge_context", "phase_router")

    # Phase router -> entry node of each phase
    _phase_route_map = {
        "regime_assessment":    "regime_assessment",
        "causal_analysis":      "causal_analysis",
        "sector_analysis":      "sector_analysis",
        "company_analysis":     "company_analysis",
        "risk_assessment":      "risk_assessment",
        "synthesis":            "synthesis",
        "phase_advance":        "phase_advance",
        END:                    END,
    }
    graph.add_conditional_edges("phase_router", _route_phase, _phase_route_map)

    # Sequential phases -> phase_advance
    graph.add_edge("regime_assessment", "phase_advance")
    graph.add_edge("risk_assessment", "phase_advance")

    # Causal analysis -> parallel fan-out or sequential
    graph.add_conditional_edges("causal_analysis", _causal_analysis_router)

    # Sector + company -> parallel_fan_in
    graph.add_node("parallel_fan_in", parallel_fan_in)
    graph.add_edge("sector_analysis", "parallel_fan_in")
    graph.add_edge("company_analysis", "parallel_fan_in")
    graph.add_edge("parallel_fan_in", "phase_router")

    # Entity analysis (comparative) -> parallel_fan_in -> phase_router
    graph.add_edge("entity_analysis", "parallel_fan_in")

    # phase_advance -> phase_router loop
    graph.add_edge("phase_advance", "phase_router")

    # synthesis -> memory_writer -> END
    graph.add_edge("synthesis", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer)


def _route_phase(state: FinancialAnalysisState):
    """Route to the next phase in the pipeline, or END if all phases done."""
    logger.info("_route_phase — phases_to_run=%s, current_phase=%s",
                state.phases_to_run, state.current_phase)
    if not state.phases_to_run:
        logger.info("No phases remaining, routing to END")
        return END

    next_phase = state.phases_to_run[0]

    # Comparative analysis: fan out to parallel entity_analysis subgraphs
    if next_phase == "comparative_analysis":
        entities = getattr(state, "entities", []) or []
        if len(entities) > 1:
            logger.info("Fanning out comparative analysis for %d entities via Send()", len(entities))
            return [
                Send("entity_analysis", _build_entity_analysis_state(state, e))
                for e in entities
            ]
        elif entities:
            logger.info("Single entity comparative analysis — sending one branch")
            return Send("entity_analysis", _build_entity_analysis_state(state, entities[0]))
        else:
            logger.warning("comparative_analysis requested but no entities found, skipping")
            return "phase_advance"

    phase_to_node = {
        "regime_assessment":    "regime_assessment",
        "causal_analysis":      "causal_analysis",
        "sector_analysis":      "sector_analysis",
        "company_analysis":     "company_analysis",
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

    logger.info("causal_analysis -> phase_advance (sequential, only one of sector/company in plan)")
    return "phase_advance"


def _build_entity_analysis_state(state: FinancialAnalysisState, entity: str) -> FinancialAnalysisState:
    """Clone the parent state for one comparative-analysis branch without serializing messages."""
    return state.model_copy(update={
        "entity_focus": entity,
    })
