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
    Financial pipeline with dependency-aware phase scheduling.

    Graph flow
    ----------
    START -> initialize -> [load_user_context ∥ financial_orchestrate] -> merge_context
          -> phase_scheduler -> (conditional: phase nodes, synthesis, END)
             Phase nodes all route back to phase_scheduler (fan-in automatic).
          -> synthesis -> memory_writer -> END

    phase_scheduler replaces the old phase_router + phase_advance + parallel_fan_in
    + _causal_analysis_router.  It reads phase_outputs (keyed by completed phase name)
    and PHASE_REGISTRY.depends_on to compute which phases are ready to run — including
    concurrent fan-out for phases whose deps are satisfied simultaneously.
    """
    from agent_sdk.agents.nodes import (
        financial_initialize,
        financial_orchestrate,
        synthesis_node,
    )
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY

    graph = StateGraph(FinancialAnalysisState)

    phase_subgraph = create_phase_subgraph(agent).compile()

    # --- Core nodes ---
    graph.add_node("initialize", financial_initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("financial_orchestrate", partial(financial_orchestrate, agent))
    graph.add_node("merge_context", merge_context)
    graph.add_node("phase_scheduler", _phase_scheduler_node)
    graph.add_node("synthesis", partial(synthesis_node, agent))
    graph.add_node("memory_writer", partial(memory_writer, agent))

    # --- Phase subgraph nodes (one compiled subgraph, many node names) ---
    # Derived from PHASE_REGISTRY so adding a phase only requires a registry entry.
    _tool_phases = [
        name for name in PHASE_REGISTRY
        if name not in ("synthesis", "entity_analysis")
    ]
    for ph in _tool_phases:
        graph.add_node(ph, partial(run_phase_subgraph, agent, phase_subgraph, phase_name=ph))

    # Comparative analysis fans out to entity_analysis (same subgraph, entity_focus set per branch)
    graph.add_node(
        "entity_analysis",
        partial(run_phase_subgraph, agent, phase_subgraph, phase_name="entity_analysis"),
    )

    # --- Edges ---
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_edge("initialize", "financial_orchestrate")
    graph.add_edge(["load_user_context", "financial_orchestrate"], "merge_context")
    graph.add_edge("merge_context", "phase_scheduler")

    # All phase nodes route back to phase_scheduler.
    # When multiple parallel branches converge, LangGraph fans them all in before
    # calling phase_scheduler (automatic fork/join semantics).
    for ph in _tool_phases:
        graph.add_edge(ph, "phase_scheduler")
    graph.add_edge("entity_analysis", "phase_scheduler")

    # phase_scheduler conditional edge decides what runs next
    graph.add_conditional_edges("phase_scheduler", _route_from_scheduler)

    # Terminal edge
    graph.add_edge("synthesis", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer)


# ============================================================================
# Scheduler logic
# ============================================================================

def _completed_phases(state: FinancialAnalysisState) -> set[str]:
    """Return the set of phase names known to be complete."""
    done = set((state.phase_outputs or {}).keys())
    # entity_analysis completing implicitly marks comparative_analysis done too
    if "entity_analysis" in done:
        done.add("comparative_analysis")
    return done


def _phase_scheduler_node(state: FinancialAnalysisState) -> dict:
    """
    Advance phases_to_run by pruning any phases whose outputs have arrived.

    Called after every phase completion (including parallel fan-in: LangGraph
    automatically waits for all concurrent branches before calling this node).
    """
    completed = _completed_phases(state)
    remaining = [p for p in state.phases_to_run if p not in completed]
    next_phase = remaining[0] if remaining else "done"
    logger.info(
        "phase_scheduler: completed=%s remaining=%s → next=%s",
        sorted(completed), remaining, next_phase,
    )
    return {
        "phases_to_run": remaining,
        "current_phase": next_phase,
    }


def _route_from_scheduler(state: FinancialAnalysisState):
    """
    Conditional edge: decide which phase(s) to run next.

    Parallel fan-out: when multiple phases have all their deps satisfied,
    return [Send(phase, state), ...] so they run concurrently.
    """
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY

    remaining = state.phases_to_run
    if not remaining:
        logger.info("phase_scheduler: all phases complete → END")
        return END

    completed = _completed_phases(state)

    next_p = remaining[0]

    if next_p == "synthesis":
        logger.info("phase_scheduler → synthesis")
        return "synthesis"

    # Comparative analysis: fan out to parallel entity_analysis subgraphs
    if next_p == "comparative_analysis":
        entities = state.entities or []
        if len(entities) > 1:
            logger.info("phase_scheduler: fanning out %d entity_analysis branches", len(entities))
            return [
                Send("entity_analysis", _build_entity_analysis_state(state, e))
                for e in entities
            ]
        elif entities:
            logger.info("phase_scheduler: single entity_analysis branch for '%s'", entities[0])
            return Send("entity_analysis", _build_entity_analysis_state(state, entities[0]))
        else:
            logger.warning("phase_scheduler: comparative_analysis but no entities — skipping to synthesis")
            return "synthesis"

    # Find all phases ready to run simultaneously (depends_on all satisfied).
    # A dependency is satisfied if it appears in completed or was never in the plan.
    plan_set = set(state.phases_to_run) | completed
    ready: list[str] = []
    for p in remaining:
        if p in ("synthesis", "comparative_analysis"):
            continue
        phase_def = PHASE_REGISTRY.get(p)
        deps = phase_def.depends_on if phase_def else []
        if all(d in completed or d not in plan_set for d in deps):
            ready.append(p)

    if not ready:
        # No phase has its deps met yet — this indicates a DAG issue or a phase that
        # should have been removed from the plan.  Route to synthesis as a safety valve.
        logger.warning(
            "phase_scheduler: no phases ready (deps not satisfied) — phases=%s completed=%s",
            remaining, sorted(completed),
        )
        return "synthesis"

    if len(ready) > 1:
        logger.info("phase_scheduler: parallel fan-out → %s", ready)
        return [Send(p, state) for p in ready]

    logger.info("phase_scheduler → %s", ready[0])
    return ready[0]


def _build_entity_analysis_state(state: FinancialAnalysisState, entity: str) -> FinancialAnalysisState:
    """Clone the parent state for one comparative-analysis branch."""
    return state.model_copy(update={"entity_focus": entity})
