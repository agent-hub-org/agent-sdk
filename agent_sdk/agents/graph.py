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
    triager,
    triager_router,
    parallel_planner,
    after_planner_router,
    stateless_executor,
    batch_check,
    synthesizer,
    load_user_context,
    memory_writer,
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
    START → initialize → triager
        → (opaque)   llm_call → should_continue → tool_node → post_tool_router → llm_call
                                                → summarize_conversation → llm_call
                                                → END
        → (analytical) parallel_planner → stateless_executor → batch_check
                                              ↑ (next batch)        ↓ (all done)
                                                              synthesizer → END
    """

    graph = StateGraph(AgentState)

    graph.add_node("initialize", initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("triager", partial(triager, agent))
    graph.add_node("memory_writer", partial(memory_writer, agent))
    # Opaque path: existing ReAct loop
    graph.add_node("llm_call", partial(llm_call, agent))
    graph.add_node("tool_node", partial(tool_node, agent))
    graph.add_node("summarize_conversation", partial(summarize_conversation, agent))
    # Analytical path: plan → parallel execute → synthesize
    graph.add_node("parallel_planner", partial(parallel_planner, agent))
    graph.add_node("stateless_executor", partial(stateless_executor, agent))
    graph.add_node("synthesizer", partial(synthesizer, agent))

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_conditional_edges("load_user_context", pre_llm_router, {
        "llm_call": "triager",
        "summarize_conversation": "summarize_conversation",
    })
    graph.add_conditional_edges("triager", triager_router, {
        "llm_call": "llm_call",
        "parallel_planner": "parallel_planner",
    })

    # Opaque path — redirect END through memory_writer
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "summarize_conversation": "summarize_conversation",
        END: "memory_writer",
    })
    graph.add_conditional_edges("tool_node", post_tool_router)
    graph.add_edge("summarize_conversation", "llm_call")

    # Analytical path — conditional in case planner fails (falls back to llm_call)
    graph.add_conditional_edges("parallel_planner", after_planner_router, {
        "stateless_executor": "stateless_executor",
        "llm_call": "llm_call",
    })
    graph.add_conditional_edges("stateless_executor", batch_check, {
        "stateless_executor": "stateless_executor",
        "synthesizer": "synthesizer",
    })
    graph.add_edge("synthesizer", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer)


def create_financial_reasoning_graph(agent, checkpointer: Optional[Any] = None):
    """
    Build the financial reasoning cognitive pipeline.

    Each analytical phase now uses a 3-sub-node structure:
        {phase}      — planner: 1 LLM call → produces phase_tool_plan
        {phase}_exec — stateless executor: asyncio.gather all tools (no LLM)
        {phase}_synth — synthesizer: 1 LLM call → structured phase output

    Phases without tools (synthesis) and the parallel comparative_analysis
    are unchanged. The query classifier at the front determines which phases
    to activate, so simple queries skip unnecessary phases.

    Graph flow
    ----------
    START → initialize → classify_query → phase_router
        → regime_assessment (plan) → regime_assessment_exec → regime_assessment_synth → phase_advance
        → causal_analysis (plan)   → causal_analysis_exec   → causal_analysis_synth   → phase_advance
        → sector_analysis (plan)   → sector_analysis_exec   → sector_analysis_synth   → phase_advance
        → company_analysis (plan)  → company_analysis_exec  → company_analysis_synth  → phase_advance
        → comparative_analysis (unchanged — parallel entity analysis)                 → phase_advance
        → risk_assessment (plan)   → risk_assessment_exec   → risk_assessment_synth   → phase_advance
        → synthesis (unchanged — no tools, pure LLM synthesis)                        → END
    """
    from agent_sdk.agents.nodes import (
        financial_initialize,
        classify_query_node,
        phase_router,
        comparative_analysis_node,
        synthesis_node,
        phase_advance,
        financial_phase_planner,
        financial_stateless_executor_node,
        financial_phase_synthesizer,
        _financial_after_plan,
    )

    graph = StateGraph(FinancialAnalysisState)

    # --- Core nodes ---
    graph.add_node("initialize", financial_initialize)
    graph.add_node("load_user_context", partial(load_user_context, agent))
    graph.add_node("classify_query", partial(classify_query_node, agent))
    graph.add_node("phase_router", phase_router)
    graph.add_node("phase_advance", phase_advance)
    graph.add_node("memory_writer", partial(memory_writer, agent))

    # --- Phases with the new plan → exec → synth sub-structure ---
    _tool_phases = [
        "regime_assessment",
        "causal_analysis",
        "sector_analysis",
        "company_analysis",
        "risk_assessment",
    ]
    for ph in _tool_phases:
        graph.add_node(ph,            partial(financial_phase_planner,           ph, agent))
        graph.add_node(f"{ph}_exec",  partial(financial_stateless_executor_node, ph, agent))
        graph.add_node(f"{ph}_synth", partial(financial_phase_synthesizer,       ph, agent))

    # Parallel Fan-out: Sector and Company analysis can run concurrently
    # We define a combined node that triggers both
    def parallel_analysis_router(state: FinancialAnalysisState) -> list[str]:
        if state.current_phase == "causal_analysis" and "sector_analysis" in state.phases_to_run and "company_analysis" in state.phases_to_run:
            return ["sector_analysis", "company_analysis"]
        return [state.current_phase]

    # Note: To implement true parallelization in LangGraph, we use the 'Send' pattern or
    # simply add edges from one node to multiple nodes.
    # Here we modify the edges instead of adding a router node for simplicity.

    # --- Unchanged phases ---
    graph.add_node("comparative_analysis", partial(comparative_analysis_node, agent))
    graph.add_node("synthesis", partial(synthesis_node, agent))

    # --- Edges ---
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "load_user_context")
    graph.add_edge("load_user_context", "classify_query")
    graph.add_edge("classify_query", "phase_router")

    # Phase router → entry node of each phase (the planner, or unchanged node)
    _phase_route_map = {
        "regime_assessment":   "regime_assessment",
        "causal_analysis":     "causal_analysis",
        "sector_analysis":     "sector_analysis",
        "company_analysis":    "company_analysis",
        "comparative_analysis":"comparative_analysis",
        "risk_assessment":     "risk_assessment",
        "synthesis":           "synthesis",
        END: END,
    }
    graph.add_conditional_edges("phase_router", _route_phase, _phase_route_map)

    # For each tool-using phase: planner → (exec or synth) → synth → phase_advance
    for ph in _tool_phases:
        graph.add_conditional_edges(
            ph,
            partial(_financial_after_plan, ph),
            {f"{ph}_exec": f"{ph}_exec", f"{ph}_synth": f"{ph}_synth"},
        )
        graph.add_edge(f"{ph}_exec",  f"{ph}_synth")

        # Modified phase_advance for parallelization
        if ph == "causal_analysis":
            # After causal analysis synthesis, if both sector and company are next, run them in parallel
            def _causal_synth_router(state):
                if "sector_analysis" in state.phases_to_run and "company_analysis" in state.phases_to_run:
                    return [Send("sector_analysis", state), Send("company_analysis", state)]
                return "phase_advance"

            graph.add_conditional_edges(f"{ph}_synth", _causal_synth_router)
        elif ph in ("sector_analysis", "company_analysis"):
            # Parallel nodes wait for each other by routing to a common sync point (phase_advance)
            graph.add_edge(f"{ph}_synth", "phase_advance")
        else:
            graph.add_edge(f"{ph}_synth", "phase_advance")

    # Comparative analysis → phase_advance (unchanged)
    graph.add_edge("comparative_analysis", "phase_advance")

    # Phase advance → router for next phase
    graph.add_edge("phase_advance", "phase_router")

    # Synthesis → memory_writer → END
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
        "regime_assessment": "regime_assessment",
        "causal_analysis": "causal_analysis",
        "sector_analysis": "sector_analysis",
        "company_analysis": "company_analysis",
        "comparative_analysis": "comparative_analysis",
        "risk_assessment": "risk_assessment",
        "synthesis": "synthesis",
    }
    target = phase_to_node.get(next_phase, END)
    logger.info("Routing to phase: %s (node: %s)", next_phase, target)
    return target

