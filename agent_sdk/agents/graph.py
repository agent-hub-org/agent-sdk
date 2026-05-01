from __future__ import annotations

import logging
from functools import partial
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from agent_sdk.agents.state import AgentState, FinancialAnalysisState, state_field
from agent_sdk.agents.subgraphs import create_react_subgraph, create_phase_subgraph
from agent_sdk.agents.subgraphs.phase_subgraph import run_phase_subgraph
from agent_sdk.sub_agents.routing_templates import ROUTING_TEMPLATES
from agent_sdk.sub_agents.registry import SUB_AGENT_REGISTRY

logger = logging.getLogger("agent_sdk.graph")
from agent_sdk.agents.nodes import (
    initialize,
    orchestrate,
    load_user_context,
    memory_writer,
)



def _pydantic_state_as_mapping(state) -> dict:
    """Coerce a Pydantic FinancialAnalysisState into a plain dict for routing-condition callbacks.

    Routing template conditions like `_has_holdings` call `.get` on the state, so we
    surface the model as a dict (without copying messages — they aren't read by conditions).
    """
    try:
        return state.model_dump(exclude={"messages"})
    except Exception:
        return {f: getattr(state, f, None) for f in ("has_holdings", "knowledge_level", "current_template")}


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
    Financial pipeline using the sub-agent dispatcher topology.

    Graph flow
    ----------
    START -> initialize -> load_user_context -> financial_orchestrate
          -> sub_agent_dispatcher
                |  (conditional: Send list of ready sub-agents OR synthesis_node)
                v
          [sub-agent nodes] -> sub_agent_dispatcher (fan-in automatic)
          -> synthesis_node -> compliance_node
          -> (conditional: jargon_simplifier_node | workspace_flush)
          -> workspace_flush -> memory_writer -> END

    Each sub-agent node calls `SubAgent.run` against the workspace store, then
    returns a delta merging its name into `agents_completed` and `workspace_populated`
    via the union reducers on FinancialAnalysisState.
    """
    from agent_sdk.agents.nodes import (
        financial_initialize,
        synthesis_node,
        compliance_node,
        jargon_simplifier_node,
        workspace_flush_node,
    )
    from agent_sdk.financial.orchestrator import financial_orchestrate as _financial_orchestrate

    graph = StateGraph(FinancialAnalysisState)

    # --- Core nodes ---
    graph.add_node("initialize", financial_initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("financial_orchestrate", partial(_financial_orchestrate, agent))
    graph.add_node("sub_agent_dispatcher", _sub_agent_dispatcher_node)

    # --- Sub-agent nodes (analysis + strategy agents only; post-process nodes added below) ---
    _post_process_names = {"synthesis", "compliance", "jargon_simplifier", "user_profiling"}
    dispatched_names = [n for n in SUB_AGENT_REGISTRY if n not in _post_process_names]
    for name in dispatched_names:
        graph.add_node(name, _make_sub_agent_node(agent, SUB_AGENT_REGISTRY[name]))

    # --- Terminal sequential nodes ---
    graph.add_node("synthesis_node", partial(synthesis_node, agent))
    graph.add_node("compliance_node", partial(compliance_node, agent))
    graph.add_node("jargon_simplifier_node", partial(jargon_simplifier_node, agent))
    graph.add_node("workspace_flush", partial(workspace_flush_node, agent))
    graph.add_node("memory_writer", partial(memory_writer, agent))

    # --- Edges ---
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_edge("load_user_context", "financial_orchestrate")
    graph.add_edge("financial_orchestrate", "sub_agent_dispatcher")

    # All sub-agent nodes route back to dispatcher (LangGraph fan-in joins parallel branches).
    for name in dispatched_names:
        graph.add_edge(name, "sub_agent_dispatcher")

    # Dispatcher conditional: returns a Send list (parallel fan-out) or the next node name.
    _route_map = {name: name for name in dispatched_names}
    _route_map["synthesis_node"] = "synthesis_node"
    graph.add_conditional_edges(
        "sub_agent_dispatcher",
        _route_from_dispatcher,
        _route_map,
    )

    # Sequential post-process pipeline.
    graph.add_edge("synthesis_node", "compliance_node")
    graph.add_conditional_edges(
        "compliance_node",
        lambda s: "jargon_simplifier_node" if _should_simplify(s) else "workspace_flush",
        {
            "jargon_simplifier_node": "jargon_simplifier_node",
            "workspace_flush": "workspace_flush",
        },
    )
    graph.add_edge("jargon_simplifier_node", "workspace_flush")
    graph.add_edge("workspace_flush", "memory_writer")
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


# ============================================================================
# Sub-agent dispatcher (sub-agent architecture)
# ============================================================================

# Sub-agent dependencies that are pre-populated outside the sub-agent fan-out.
# `user_profile` is loaded by `load_user_context` before the dispatcher runs.
_ALWAYS_AVAILABLE: frozenset[str] = frozenset({"user_profile"})


def _sub_agent_dispatcher_node(state) -> dict:
    """Pure pass-through node — routing decisions live in `_route_from_dispatcher`."""
    return {}


def _route_from_dispatcher(state):
    """Resolve which sub-agents are ready to run.

    Returns either:
      * a list of `Send(node_name, state)` for parallel fan-out, or
      * the literal string ``"synthesis_node"`` when no further sub-agents remain.
    """
    template_name = state_field(state, "current_template", "general_analysis")
    template = ROUTING_TEMPLATES.get(template_name)
    if template is None or not template.required_agents:
        return "synthesis_node"

    completed = state_field(state, "agents_completed", set()) or set()
    populated = state_field(state, "workspace_populated", set()) or set()

    # Routing template conditions expect a mapping (they call `.get`); adapt Pydantic models.
    cond_state = state if isinstance(state, dict) else _pydantic_state_as_mapping(state)

    ready: list[str] = []
    for spec in template.required_agents:
        if spec.name in completed:
            continue
        if spec.condition is not None:
            try:
                if not spec.condition(cond_state):
                    continue
            except Exception:
                logger.warning(
                    "_route_from_dispatcher: condition for '%s' raised — skipping",
                    spec.name,
                    exc_info=True,
                )
                continue
        sub_agent = SUB_AGENT_REGISTRY.get(spec.name)
        if sub_agent is None:
            continue
        deps_met = all(
            dep in populated or dep in _ALWAYS_AVAILABLE
            for dep in sub_agent.reads_from
        )
        if deps_met:
            ready.append(spec.name)

    if not ready:
        return "synthesis_node"

    return [Send(name, state) for name in ready]


def _should_simplify(state) -> bool:
    """True when the active template's post-process includes a satisfied jargon_simplifier spec."""
    template_name = state_field(state, "current_template", "")
    template = ROUTING_TEMPLATES.get(template_name)
    if template is None:
        return False
    cond_state = state if isinstance(state, dict) else _pydantic_state_as_mapping(state)
    for spec in template.post_process:
        if spec.name != "jargon_simplifier":
            continue
        if spec.condition is None:
            return True
        try:
            return bool(spec.condition(cond_state))
        except Exception:
            return False
    return False


async def _build_sub_agent_input_async(sub_agent, state, workspace_store):
    """Assemble a `SubAgentInput` by reading the sub-agent's declared `reads_from` keys."""
    from agent_sdk.sub_agents.base import SubAgentInput

    workspace_id = state_field(state, "workspace_id", "")
    context_parts: list[str] = []
    for key in sub_agent.reads_from:
        if key in _ALWAYS_AVAILABLE:
            continue
        val = await workspace_store.read(workspace_id, key)
        if val:
            context_parts.append(f"[{key.upper()}]\n{val.get('findings', '')}")

    messages = state_field(state, "messages", []) or []
    query = ""
    if messages:
        query = getattr(messages[-1], "content", "") or ""
    entities = state_field(state, "entities", []) or []

    return SubAgentInput(
        query=query,
        entities=list(entities),
        workspace_context="\n\n".join(context_parts),
        user_profile={},
    )


def _make_sub_agent_node(agent, sub_agent):
    """Build a LangGraph async node that runs a single sub-agent against the workspace."""
    async def _node(state) -> dict:
        workspace_store = getattr(agent, "workspace_store", None)
        sub_agent_cache = getattr(agent, "sub_agent_cache", None)
        workspace_id = state_field(state, "workspace_id", "")

        if workspace_store is None:
            logger.warning(
                "sub_agent[%s]: agent has no workspace_store — marking complete without execution",
                sub_agent.name,
            )
            return {
                "workspace_populated": {sub_agent.writes_to},
                "agents_completed": {sub_agent.name},
            }

        inp = await _build_sub_agent_input_async(sub_agent, state, workspace_store)
        ran_ok = False
        try:
            await sub_agent.run(inp, workspace_store, workspace_id, sub_agent_cache)
            ran_ok = True
        except Exception:
            logger.exception("sub_agent[%s]: run failed", sub_agent.name)

        return {
            "workspace_populated": {sub_agent.writes_to} if ran_ok else set(),
            "agents_completed": {sub_agent.name},
        }

    _node.__name__ = f"sub_agent_node__{sub_agent.name}"
    return _node
