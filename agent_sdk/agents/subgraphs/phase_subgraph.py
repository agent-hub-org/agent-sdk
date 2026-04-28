"""
Financial phase executor as a compiled LangGraph subgraph.

The phase subgraph runs on an internal state schema so phase-local buffers,
budgets, and tool bindings stay isolated from the parent financial graph.
"""

from __future__ import annotations

import logging
import re
import time
from functools import partial
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent_sdk.agents.nodes import _execute_tool_calls
from agent_sdk.agents.llm_utils import invoke_with_retry as _invoke_with_retry
from agent_sdk.financial.phase_helpers import get_phase_llm as _get_phase_llm, get_phase_tools as _get_phase_tools
from agent_sdk.financial.utils import format_date_context as _format_date_context
from agent_sdk.agents.state import FinancialAnalysisState, PhaseSubgraphState
from agent_sdk.metrics import llm_call_duration

logger = logging.getLogger("agent_sdk.subgraphs.phase")

def _get_phase_budget(phase_name: str) -> int:
    """Read budget from PHASE_REGISTRY; fall back to 4 for unknown phases."""
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY
    phase_def = PHASE_REGISTRY.get(phase_name)
    return phase_def.budget if phase_def else 4


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


_LOCAL_FINDINGS_MAX = 5     # sliding window size
_LOCAL_FINDINGS_CAP = 1600  # max chars per entry


def _build_prior_phases_summary(parent_phase_outputs: dict | None, phase_name: str) -> str:
    """Condense prior phase key_facts into a compact background string (pure string op, no LLM)."""
    if not parent_phase_outputs:
        return ""
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY
    parts: list[str] = []
    for p_name in PHASE_REGISTRY:
        if p_name == phase_name:
            break
        po_raw = parent_phase_outputs.get(p_name)
        if po_raw is None:
            continue
        key_facts = po_raw.get("key_facts", []) if isinstance(po_raw, dict) else []
        label = p_name.upper().replace("_", " ")
        if key_facts:
            facts_str = "\n".join(f"  • {f}" for f in key_facts[:8])
            parts.append(f"[{label}]\n{facts_str}")
        elif isinstance(po_raw, dict) and po_raw.get("findings"):
            # Fall back to a short snippet of the findings text
            snippet = po_raw["findings"][:400].rstrip()
            parts.append(f"[{label}]\n  {snippet}…")
    return "\n\n".join(parts)


def _build_local_context_section(window: list[str]) -> str:
    """Format the rolling tool-result window for LLM injection."""
    if not window:
        return ""
    return "CURRENT PHASE TOOL RESULTS (most recent first):\n" + "\n---\n".join(
        reversed(window)  # most recent last = closer to the actual LLM turn
    )


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

    date_context = _format_date_context(as_of_date)

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


_KEY_FACT_RE = re.compile(r"^[\s]*(?:[-•*]|\d+[.)]) (.+)", re.MULTILINE)


def _extract_key_facts(text: str) -> list[str]:
    """Pull bullet/numbered items from phase findings text (cheap regex, no LLM)."""
    matches = _KEY_FACT_RE.findall(text)
    return [m.strip() for m in matches if len(m.strip()) > 10][:15]


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

    parent_phase_outputs = getattr(state, "phase_outputs", None) or {}

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
        "parent_phase_outputs": {k: v.model_dump() for k, v in parent_phase_outputs.items()},
    }


async def phase_init(agent, state: PhaseSubgraphState) -> dict:
    """Prepare the isolated phase thread and set the stable system prompt (set once, never mutated)."""
    phase_name = state.current_phase or "unknown"
    budget = _get_phase_budget(phase_name)

    base_system = state.parent_system_text
    phase_plan = _extract_phase_plan(state.scratchpad, phase_name)

    # Compute a compact summary of prior phases from typed outputs (pure string op, no LLM)
    prior_summary = _build_prior_phases_summary(state.parent_phase_outputs, phase_name)

    # Build the STABLE system text once: no per-iteration running_context dump.
    # Only prior_phases_summary (a small condensed version) is injected here.
    system_text = _build_phase_system_text(
        base_system=base_system,
        phase_name=phase_name,
        running_context=prior_summary or None,  # compact summary, not the full blob
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
        "prior_phases_summary": prior_summary,
        "local_findings_window": [],
        "local_context_section": "",
        "phase_system_stable": True,
    }


async def phase_llm_call(agent, state: PhaseSubgraphState) -> dict:
    """Run one LLM turn within the phase.

    system_text is the stable background set by phase_init (never rebuilt here).
    local_context_section is injected as a HumanMessage prefix so the system
    prompt stays at a fixed size regardless of how many tool calls have occurred.
    """
    llm = _get_phase_llm(agent, state)
    tools = [agent.tools_by_name[n] for n in state.phase_tools if n in agent.tools_by_name]
    llm_with_tools = agent.get_bound_llm(llm, tools) if tools else llm

    messages: list = [SystemMessage(content=state.phase_system_text)]
    # Inject rolling tool results as a HumanMessage prefix (not part of the system prompt)
    if state.local_context_section:
        messages.append(HumanMessage(content=state.local_context_section))
    messages.extend(list(state.phase_messages))

    _t0 = time.monotonic()
    response = await _invoke_with_retry(llm_with_tools, messages)
    _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
    llm_call_duration.labels(agent="sdk", model=_model_name, phase=state.current_phase).observe(
        time.monotonic() - _t0
    )
    return {
        "phase_messages": [response],
        "phase_iteration": state.phase_iteration + 1,
    }


async def phase_tool_node(agent, state: PhaseSubgraphState) -> dict:
    """Execute tool calls from the last phase LLM response and update the local context window.

    No longer rebuilds the (stable) system prompt. Instead updates local_findings_window
    (a sliding window of the last _LOCAL_FINDINGS_MAX tool-result strings, each capped to
    _LOCAL_FINDINGS_CAP chars) and recomputes local_context_section for the next LLM turn.
    """
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

    # Update sliding window: cap each entry, keep last _LOCAL_FINDINGS_MAX
    new_entry = findings[:_LOCAL_FINDINGS_CAP] if len(findings) > _LOCAL_FINDINGS_CAP else findings
    updated_window = list(state.local_findings_window) + [new_entry]
    updated_window = updated_window[-_LOCAL_FINDINGS_MAX:]
    updated_ctx_section = _build_local_context_section(updated_window)

    return {
        "phase_messages": list(results),
        "phase_findings": updated_findings,
        "local_findings_window": updated_window,
        "local_context_section": updated_ctx_section,
        "tool_calls_log": log_entries,
        # running_context still accumulated for backward compat with synthesis fallback
        "running_context": _append_findings(state.running_context, findings),
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
    from agent_sdk.agents.state import PhaseOutput

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
        "phase_subgraph: phase '%s' complete - %d chars",
        state.current_phase,
        len(phase_block),
    )

    phase_output = PhaseOutput(
        findings=findings,
        key_facts=_extract_key_facts(findings),
        confidence=1.0,
    )

    return {
        "running_context": phase_block,
        "phase_outputs": {state.current_phase: phase_output},
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
    from agent_sdk.agents.state import PhaseOutput

    subgraph_input = _phase_input_from_parent(
        state,
        phase_name=phase_name,
        entity_focus=state.entity_focus,
    )

    try:
        result = await phase_graph.ainvoke(subgraph_input, config=config)
        return {
            "running_context": result["running_context"],
            "tool_calls_log": result.get("tool_calls_log", []),
            "phase_outputs": result.get("phase_outputs", {}),
            "iteration": state.iteration + 1,
        }
    except Exception as exc:
        # Degrade gracefully: one failing phase must not crash the whole pipeline.
        # Write a low-confidence PhaseOutput so the scheduler and synthesis know it failed.
        logger.error(
            "run_phase_subgraph: phase '%s' raised an exception — degrading gracefully: %s",
            phase_name, exc, exc_info=True,
        )
        degraded_output = PhaseOutput(
            findings=f"Phase {phase_name} failed: {exc}",
            key_facts=[],
            confidence=0.0,
        )
        error_block = (
            f"=== {phase_name.upper().replace('_', ' ')} ===\n"
            f"(Phase failed: {exc})\n"
            f"=== END {phase_name.upper().replace('_', ' ')} ==="
        )
        return {
            "running_context": error_block,
            "tool_calls_log": [],
            "phase_outputs": {phase_name: degraded_output},
            "iteration": state.iteration + 1,
        }
