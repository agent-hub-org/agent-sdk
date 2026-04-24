"""
Financial phase executor as a compiled LangGraph subgraph.

The phase subgraph runs on an internal state schema so phase-local buffers,
budgets, and tool bindings stay isolated from the parent financial graph.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import partial
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent_sdk.agents.nodes import (
    _execute_tool_calls,
    _get_phase_llm,
    _get_phase_tools,
    _invoke_with_retry,
)
from agent_sdk.agents.state import FinancialAnalysisState, PhaseSubgraphState
from agent_sdk.metrics import llm_call_duration

logger = logging.getLogger("agent_sdk.subgraphs.phase")

_PHASE_BUDGETS = {
    "regime_assessment": 4,
    "causal_analysis": 4,
    "sector_analysis": 5,
    "company_analysis": 10,
    "risk_assessment": 6,
    "entity_analysis": 5,
}


def _extract_phase_plan(full_plan: Optional[str], phase: str) -> Optional[str]:
    """Return the single plan line for this phase; fall back to full plan."""
    if not full_plan:
        return None
    phase_key = phase.replace("_", " ")
    for line in full_plan.splitlines():
        stripped = line.strip().lower()
        if stripped.startswith(phase + ":") or stripped.startswith(phase_key + ":"):
            return line.strip()
    return full_plan


def _build_phase_system_text(
    *,
    base_system: str,
    phase_name: str,
    running_context: Optional[str],
    as_of_date: Optional[str],
    phase_plan: Optional[str],
    entity_focus: Optional[str],
    iteration: int,
) -> str:
    """Build the dynamic system prompt for one phase iteration."""
    phase_header = (
        f"\n\n=== CURRENT PHASE: {phase_name.upper().replace('_', ' ')} ===\n"
        f"Focus on the {phase_name.replace('_', ' ')} section of your EXECUTION PLAN. "
        f"Call the tools needed for this phase. Stop when this phase section is complete."
    )

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if as_of_date:
        year = as_of_date[:4]
        date_context = (
            f"\n\nTODAY'S DATE: {today}. HISTORICAL REFERENCE DATE: {as_of_date}\n"
            f"Always include the historical year ({year}) in search queries."
        )
    else:
        year = datetime.now(timezone.utc).year
        date_context = (
            f"\n\nTODAY'S DATE: {today}\n"
            f"Always include the current year ({year}) in search queries for up-to-date results."
        )

    extra_sections: list[str] = []
    if phase_plan:
        extra_sections.append("PHASE PLAN (follow this):\n" + phase_plan)
    if running_context:
        extra_sections.append(
            "PRIOR PHASE RESULTS (already done — do not re-fetch):\n" + running_context
        )

    system_text = base_system + phase_header + date_context
    if iteration == 0:
        system_text += (
            "\n\nBefore your first tool call: identify the single most critical "
            "data gap for this phase and the one tool that fills it. Then act immediately."
        )
    if extra_sections:
        system_text += "\n\n" + "\n\n".join(extra_sections)

    if entity_focus:
        system_text += (
            f"\n\n=== FOCUS ENTITY: {entity_focus} ===\n"
            f"Perform a thorough company analysis for {entity_focus}. "
            f"Retrieve fundamentals, valuation, recent performance, and key risks."
        )

    return system_text


def _append_findings(existing: Optional[str], addition: Optional[str]) -> Optional[str]:
    if not addition:
        return existing
    if existing:
        return f"{existing}\n\n{addition}"
    return addition


def _phase_input_from_parent(
    state: FinancialAnalysisState,
    *,
    phase_name: str,
    entity_focus: Optional[str] = None,
) -> dict:
    """Map parent financial state into the isolated phase subgraph state.

    Extracts only the system prompt text and last HumanMessage rather than
    copying the full parent message list, which would be O(n) per phase.
    """
    parent_system_text = ""
    parent_user_message = None
    for msg in state.messages:
        if isinstance(msg, SystemMessage):
            parent_system_text = msg.content
            break
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            parent_user_message = msg
            break

    return {
        "parent_system_text": parent_system_text,
        "parent_user_message": parent_user_message,
        "scratchpad": state.scratchpad,
        "running_context": state.running_context,
        "as_of_date": state.as_of_date,
        "model_id": state.model_id,
        "tool_timeout": state.tool_timeout,
        "current_phase": phase_name,
        "entity_focus": entity_focus,
    }


async def phase_init(agent, state: PhaseSubgraphState) -> dict:
    """Prepare the isolated phase thread and its dynamic system prompt."""
    phase_name = state.current_phase or "unknown"
    budget = _PHASE_BUDGETS.get(phase_name, 4)

    base_system = state.parent_system_text
    phase_plan = _extract_phase_plan(state.scratchpad, phase_name)
    system_text = _build_phase_system_text(
        base_system=base_system,
        phase_name=phase_name,
        running_context=state.running_context,
        as_of_date=state.as_of_date,
        phase_plan=phase_plan,
        entity_focus=state.entity_focus,
        iteration=0,
    )

    tools = _get_phase_tools(agent, "company_analysis" if state.entity_focus else phase_name)

    return {
        "phase_messages": [] if state.parent_user_message is None else [state.parent_user_message],
        "phase_system_text": system_text,
        "phase_base_system_text": base_system,
        "phase_iteration": 0,
        "phase_budget": budget,
        "phase_tools": [t.name for t in tools],
        "phase_plan": phase_plan,
        "phase_findings": None,
    }


async def phase_llm_call(agent, state: PhaseSubgraphState) -> dict:
    """Run one LLM turn within the phase using the current dynamic system prompt."""
    llm = _get_phase_llm(agent, state)
    tools = [agent.tools_by_name[n] for n in state.phase_tools if n in agent.tools_by_name]
    llm_with_tools = agent.get_bound_llm(llm, tools) if tools else llm

    prompt = [SystemMessage(content=state.phase_system_text), *list(state.phase_messages)]
    _t0 = time.monotonic()
    response = await _invoke_with_retry(llm_with_tools, prompt)
    _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    llm_call_duration.labels(agent="sdk", model=_model_name, phase=state.current_phase).observe(
        time.monotonic() - _t0
    )
    return {
        "phase_messages": [response],
        "phase_iteration": state.phase_iteration + 1,
    }


async def phase_tool_node(agent, state: PhaseSubgraphState) -> dict:
    """Execute tool calls from the last phase LLM response and refresh context."""
    if not state.phase_messages:
        return {}

    last_msg = state.phase_messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", None) or []
    if not tool_calls:
        return {}

    tools = [agent.tools_by_name[n] for n in state.phase_tools if n in agent.tools_by_name]
    results = await _execute_tool_calls(agent, tool_calls, state.tool_timeout, phase_tools=tools)

    log_entries = []
    findings_parts = []
    for tc in tool_calls:
        log_entries.append({
            "action": "tool_call",
            "phase": state.current_phase,
            "tool": tc["name"],
            "args": tc.get("args", {}),
        })
    for tc, tr in zip(tool_calls, results):
        log_entries.append({
            "action": "tool_result",
            "phase": state.current_phase,
            "tool": tc["name"],
            "result_length": len(tr.content),
            "result_preview": tr.content[:500] if len(tr.content) > 500 else tr.content,
        })
        findings_parts.append(f"[{tc['name']}] -> {tr.content}")

    findings = "\n".join(findings_parts)
    updated_findings = _append_findings(state.phase_findings, findings)
    updated_running_context = _append_findings(state.running_context, findings)
    updated_system_text = _build_phase_system_text(
        base_system=state.phase_base_system_text,
        phase_name=state.current_phase,
        running_context=updated_running_context,
        as_of_date=state.as_of_date,
        phase_plan=state.phase_plan,
        entity_focus=state.entity_focus,
        iteration=state.phase_iteration,
    )

    return {
        "phase_messages": list(results),
        "phase_findings": updated_findings,
        "phase_system_text": updated_system_text,
        "tool_calls_log": log_entries,
        "running_context": updated_running_context,
    }


def phase_should_continue(state: PhaseSubgraphState) -> str:
    """Route to tool execution when needed, otherwise finalize."""
    if not state.phase_messages:
        return "phase_finalize"

    last_msg = state.phase_messages[-1]
    if getattr(last_msg, "tool_calls", None):
        return "phase_tool_node"

    return "phase_finalize"


def phase_after_tool_router(state: PhaseSubgraphState) -> str:
    """After tool execution, either continue the loop or stop at the budget."""
    if state.phase_iteration >= state.phase_budget:
        logger.warning("Phase %s hit budget (%d)", state.current_phase, state.phase_budget)
        return "phase_finalize"
    return "phase_llm_call"


async def phase_finalize(agent, state: PhaseSubgraphState) -> dict:
    """Persist final prose and compile the phase findings block."""
    final_message = state.phase_messages[-1] if state.phase_messages else None
    final_prose = None
    if (
        isinstance(final_message, AIMessage)
        and not getattr(final_message, "tool_calls", None)
        and final_message.content
    ):
        final_prose = str(final_message.content)

    findings = _append_findings(state.phase_findings, final_prose) or "(no data retrieved)"
    phase_block = (
        f"=== {state.current_phase.upper().replace('_', ' ')} ===\n"
        f"{findings}\n"
        f"=== END {state.current_phase.upper().replace('_', ' ')} ==="
    )

    logger.info(
        "phase_subgraph: phase '%s' complete - %d chars added to running_context",
        state.current_phase,
        len(phase_block),
    )

    return {
        "running_context": phase_block,
    }


def create_phase_subgraph(agent):
    """Build the compiled phase executor subgraph using isolated internal state."""
    graph = StateGraph(PhaseSubgraphState)

    graph.add_node("phase_init", partial(phase_init, agent))
    graph.add_node("phase_llm_call", partial(phase_llm_call, agent))
    graph.add_node("phase_tool_node", partial(phase_tool_node, agent))
    graph.add_node("phase_finalize", partial(phase_finalize, agent))

    graph.add_edge(START, "phase_init")
    graph.add_edge("phase_init", "phase_llm_call")
    graph.add_conditional_edges("phase_llm_call", phase_should_continue, {
        "phase_tool_node": "phase_tool_node",
        "phase_finalize": "phase_finalize",
    })
    graph.add_conditional_edges("phase_tool_node", phase_after_tool_router, {
        "phase_llm_call": "phase_llm_call",
        "phase_finalize": "phase_finalize",
    })
    graph.add_edge("phase_finalize", END)

    return graph


async def run_phase_subgraph(
    agent,
    phase_graph,
    state: FinancialAnalysisState,
    *,
    phase_name: str,
    config=None,
) -> dict:
    """
    Run one financial phase through the isolated subgraph and map the results back.

    The parent graph only receives stable shared outputs; phase-local state never
    writes back into FinancialAnalysisState.
    """
    subgraph_input = _phase_input_from_parent(
        state,
        phase_name=phase_name,
        entity_focus=state.entity_focus,
    )
    result = await phase_graph.ainvoke(subgraph_input, config=config)
    return {
        "running_context": result["running_context"],
        "tool_calls_log": result.get("tool_calls_log", []),
        "iteration": state.iteration + 1,
    }
