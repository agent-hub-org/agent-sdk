"""Financial pipeline phase helpers: tools, prompts, LLM selection."""
from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_sdk.financial.utils import format_date_context

logger = logging.getLogger("agent_sdk.financial.phase_helpers")

_RESEARCH_ONLY_TOOLS = frozenset({
    "check_papers_in_db",
    "retrieve_papers",
    "hybrid_retrieve_papers",
    "download_and_store_arxiv_papers",
})


def get_phase_llm(agent, state):
    """Return the LLM for the current phase, supporting model_id override."""
    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        return get_llm(state.model_id)
    return agent.llm


def get_phase_tools(agent, phase: str) -> list:
    """Get tools appropriate for a specific reasoning phase (cached per fingerprint)."""
    tool_fingerprint = frozenset(agent.tools_by_name.keys())
    cache_key = (phase, tool_fingerprint)
    cached = agent._phase_tools_cache.get(cache_key)
    if cached is not None:
        return cached

    from agent_sdk.financial.causal_graph import get_causal_graph_tools
    from agent_sdk.financial.ontology import get_ontology_tools
    from agent_sdk.financial.quant_tools import get_quant_tools
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY

    all_financial: dict[str, object] = {}
    for t in get_quant_tools() + get_causal_graph_tools() + get_ontology_tools():
        all_financial[t.name] = t

    phase_def = PHASE_REGISTRY.get(phase)
    wanted_names = set(phase_def.financial_tool_names) if phase_def else set()
    financial_tools = [t for name, t in all_financial.items() if name in wanted_names]
    agent_tools = [t for t in agent.tools_by_name.values() if t.name not in _RESEARCH_ONLY_TOOLS]

    seen: set[str] = set()
    combined: list = []
    for t in financial_tools + agent_tools:
        if t.name not in seen:
            combined.append(t)
            seen.add(t.name)

    result = agent.get_available_tools(phase_tools=combined)
    agent._phase_tools_cache[cache_key] = result
    return result


def build_phase_prompt(state, phase_system_prompt: str) -> list:
    """Build the message list for a phase LLM call."""
    date_context = format_date_context(getattr(state, "as_of_date", None))
    messages = [SystemMessage(content=phase_system_prompt + date_context)]

    responded_ids = {
        msg.tool_call_id
        for msg in state.messages
        if isinstance(msg, ToolMessage) and msg.tool_call_id
    }
    last_human_idx = max(
        (i for i, m in enumerate(state.messages) if isinstance(m, HumanMessage)),
        default=-1,
    )

    for i, msg in enumerate(state.messages):
        if isinstance(msg, HumanMessage):
            messages.append(msg)
        elif isinstance(msg, ToolMessage):
            messages.append(msg)
        elif isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                pending = [tc["id"] for tc in msg.tool_calls if tc["id"] not in responded_ids]
                if pending:
                    logger.warning(
                        "build_phase_prompt: dropping AIMessage with %d unmatched tool_call_id(s) %s",
                        len(pending), pending,
                    )
                else:
                    messages.append(msg)
            elif msg.content and i < last_human_idx:
                messages.append(msg)

    return messages


def format_context(data: dict | None) -> str:
    """Format a dict as readable context for injection into prompts."""
    if not data:
        return "(not yet assessed)"
    try:
        return json.dumps(data, indent=2, default=str)
    except (TypeError, ValueError):
        return str(data)


def format_tool_catalog(tools: list) -> str:
    """Format a list of tools into a compact catalog string for planner prompts."""
    lines = ["TOOLS AVAILABLE:"]
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = (getattr(t, "description", "") or "")[:120]
        try:
            schema = t.get_input_schema().model_json_schema() if hasattr(t, "get_input_schema") else {}
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            arg_parts = [
                f"{k}: {v.get('type', 'any')}{'*' if k in required else ''}"
                for k, v in props.items()
            ]
            args_str = ", ".join(arg_parts) if arg_parts else "no args"
        except Exception:
            args_str = "see description"
        lines.append(f"- {name}: {desc}\n  args: {args_str}")
    return "\n".join(lines)
