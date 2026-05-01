from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import time
import uuid
from typing import Any, Literal, Sequence

# ContextVar so notepad tool closures know which session they're running in
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_sdk_session_id", default="default"
)

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from agent_sdk.agents.state import AgentState, FinancialAnalysisState, state_field
from agent_sdk.agents.llm_utils import invoke_with_retry, compress_running_context
from agent_sdk.agents.tool_executor import execute_tool_calls
from agent_sdk.financial.orchestrator import financial_orchestrate, extract_json as _extract_json, normalize_classification as _normalize_classification
from agent_sdk.agents.memory_nodes import load_user_context, memory_writer
from agent_sdk.config import settings
from agent_sdk.metrics import llm_call_duration

logger = logging.getLogger("agent_sdk.nodes")

# Re-export for backward compatibility (graph.py, subgraphs import these by name)
_invoke_with_retry = invoke_with_retry
_compress_running_context = compress_running_context
_execute_tool_calls = execute_tool_calls

# Pre-compiled regexes
_MALFORMED_TOOL_PATTERN = re.compile(r'<function=(\w+)(.*?)</function>', re.DOTALL)


def _parse_malformed_tool_call(failed_generation: str) -> list[dict] | None:
    matches = _MALFORMED_TOOL_PATTERN.findall(failed_generation)
    if not matches:
        return None
    import json
    tool_calls = []
    for func_name, raw_args in matches:
        raw_args = raw_args.strip()
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            logger.warning("Could not parse args for tool '%s': %s", func_name, raw_args)
            return None
        tool_calls.append({
            "name": func_name,
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:24]}",
        })
    return tool_calls if tool_calls else None


def _strip_context_block(text: str) -> str:
    marker = "[/CONTEXT]"
    if marker in text:
        return text[text.find(marker) + len(marker):].strip()
    return text.strip()


def _estimate_token_count(messages: Sequence, extra_text: str = "") -> int:
    msg_chars = sum(
        len(getattr(m, "content", "") or "") // (3 if isinstance(m, ToolMessage) else 4)
        for m in messages
    )
    return msg_chars + len(extra_text) // 4


def _strip_dangling_tool_calls(messages: list) -> list:
    responded = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    result = []
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            unanswered = [tc for tc in msg.tool_calls if tc["id"] not in responded]
            if unanswered:
                if msg.content:
                    result.append(AIMessage(content=msg.content))
                continue
        result.append(msg)
    return result


async def initialize(state: AgentState) -> dict:
    _clear: dict = {"running_context": None, "scratchpad": None}
    if state.messages and isinstance(state.messages[0], SystemMessage):
        if state.system_prompt and state.messages[0].content != state.system_prompt:
            logger.info("Updating system prompt (%d chars)", len(state.system_prompt))
            return {**_clear, "messages": [RemoveMessage(id=state.messages[0].id),
                                           SystemMessage(content=state.system_prompt)]}
        logger.debug("System prompt already present, skipping initialization")
        return _clear
    content = state.system_prompt or (
        "You are an autonomous assistant. "
        "You may call tools to achieve the user's goal, "
        "or respond directly when tools are not needed."
    )
    logger.info("Initialized agent with system prompt (%d chars)", len(content))
    return {**_clear, "messages": [SystemMessage(content=content)]}


async def llm_call(agent, state: AgentState) -> dict:
    logger.info("LLM call — iteration %d/%d, message count: %d, model_id: %s",
                state.iteration + 1, state.max_iterations, len(state.messages),
                state.model_id or "default")

    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        llm = get_llm(state.model_id)
    else:
        llm = agent.llm

    tools = list(agent.tools_by_name.values())
    llm_with_tools = agent.get_bound_llm(llm, tools)

    extra_sections: list[str] = []
    if state.summary:
        extra_sections.append(
            "INTERNAL CONTEXT (for your reference only — do NOT include, repeat, "
            "or paraphrase any of this in your response to the user):\n"
            f"Previous conversation summary: {state.summary}"
        )

    plan = getattr(state, "scratchpad", None)
    if plan:
        extra_sections.append(
            "EXECUTION PLAN (your task checklist — follow this, don't re-plan):\n" + plan
        )

    running_ctx = getattr(state, "running_context", None)
    ctx_reset_update: dict = {}
    session_id_for_ctx = getattr(state, "session_id", "default")
    if session_id_for_ctx in agent._pending_ctx_compressions:
        task = agent._pending_ctx_compressions.pop(session_id_for_ctx)
        compressed = await task
        running_ctx = compressed
        ctx_reset_update = {"running_context": f"__RESET__:{compressed}"}
        logger.debug("llm_call: applied compressed running_context (%d chars)", len(compressed))
    if running_ctx:
        extra_sections.append(
            "WORK DONE SO FAR (tool results / completed phases — use this, don't re-fetch):\n"
            + running_ctx
        )

    perspective = getattr(state, "perspective_context", None)
    if perspective:
        extra_sections.append(perspective)

    session_id = getattr(state, "session_id", "default")
    if session_id not in agent._session_notepads:
        notepad_from_state = getattr(state, "session_notepad", None)
        if notepad_from_state:
            agent._session_notepads[session_id] = dict(notepad_from_state)
    notepad = agent._session_notepads.get(session_id) or getattr(state, "session_notepad", None)
    if notepad:
        extra_sections.append(
            "SESSION NOTEPAD (important findings from earlier in this conversation — "
            "use these, don't re-derive them):\n"
            + "\n".join(f"• {k}: {v}" for k, v in notepad.items())
        )

    validation_hint = getattr(state, "validation_hint", None)
    if validation_hint:
        extra_sections.insert(0, validation_hint)

    if state.iteration == 0 and extra_sections:
        has_prior_tool_results = any(isinstance(m, ToolMessage) for m in state.messages[-10:])
        if not has_prior_tool_results:
            extra_sections.append(
                "Before your first tool call: identify the single most specific "
                "piece of information needed, then call exactly the right tool."
            )

    if extra_sections:
        messages = list(state.messages)
        appended = "\n\n".join(extra_sections)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=f"{messages[0].content}\n\n{appended}")
        else:
            messages.insert(0, SystemMessage(content=appended))
        prompt = messages
    else:
        prompt = list(state.messages)

    try:
        _t0 = time.monotonic()
        response = await invoke_with_retry(llm_with_tools, prompt)
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        _phase = getattr(state, "current_phase", None) or "standard"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase=_phase).observe(time.monotonic() - _t0)
    except Exception as e:
        error_body = getattr(e, "body", None) or {}
        if isinstance(error_body, dict):
            failed_gen = error_body.get("failed_generation") or error_body.get("error", {}).get("failed_generation")
        else:
            failed_gen = None

        if failed_gen:
            logger.warning("LLM produced malformed tool call, attempting to parse: %s", failed_gen.strip())
            parsed_calls = _parse_malformed_tool_call(failed_gen)
            if parsed_calls:
                valid_calls = [tc for tc in parsed_calls if tc["name"] in agent.tools_by_name]
                if valid_calls:
                    logger.info("Recovered %d tool call(s) from malformed generation: %s",
                                len(valid_calls), [tc["name"] for tc in valid_calls])
                    return {
                        "messages": [AIMessage(content="", tool_calls=valid_calls)],
                        "iteration": state.iteration + 1,
                    }
            logger.warning("No tool calls in failed generation — recovering as plain text response (%d chars)",
                           len(failed_gen))
            return {
                "messages": [AIMessage(content=failed_gen)],
                "iteration": state.iteration + 1,
            }
        raise

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        logger.info("LLM requested %d tool call(s): %s",
                    len(tool_calls), [tc["name"] for tc in tool_calls])
    else:
        logger.info("LLM returned final response (%d chars)", len(response.content))

    result: dict = {
        "messages": [response],
        "iteration": state.iteration + 1,
        **ctx_reset_update,
    }
    if getattr(state, "validation_hint", None):
        result["validation_hint"] = None
    return result


async def summarize_conversation(agent, state: AgentState) -> dict:
    logger.info("Summarizing conversation — %d messages, existing summary: %s",
                len(state.messages), "yes" if state.summary else "no")

    keep_n = max(state.keep_last_n_messages, 1)
    all_messages = list(state.messages)
    has_system = all_messages and isinstance(all_messages[0], SystemMessage)
    system_msg = all_messages[0] if has_system else None
    conversation = all_messages[1:] if has_system else all_messages

    messages_to_prune = conversation[:-keep_n] if len(conversation) > keep_n else []
    if not messages_to_prune:
        logger.info("Not enough messages to prune, skipping summarization")
        return {}

    messages_to_prune = _strip_dangling_tool_calls(messages_to_prune)
    if not messages_to_prune:
        logger.info("Nothing to prune after stripping dangling tool_calls, skipping summarization")
        return {}

    _PRESERVE_INSTRUCTION = (
        "Preserve ALL ticker symbols, asset names, prices, financial metrics, percentages, "
        "and specific dates EXACTLY as they appear — do not paraphrase, round, or omit numbers. "
        "IMPORTANT: Just state the facts. Do NOT mention that you were asked to summarize."
    )
    existing_summary = state.summary or ""
    if existing_summary:
        summary_message = (
            f"This is a summary of the conversation to date: {existing_summary}\n\n"
            f"Extend the summary by taking into account the new messages below. {_PRESERVE_INSTRUCTION}"
        )
    else:
        summary_message = (
            f"Create a summary of the facts and context from the conversation below. {_PRESERVE_INSTRUCTION}"
        )

    summarizer_input = (
        [SystemMessage(content=summary_message)]
        + messages_to_prune
        + [SystemMessage(content=(
            "Provide a summary capturing the key facts, decisions, results, and all specific "
            f"data points (numbers, tickers, dates). {_PRESERVE_INSTRUCTION}"
        ))]
    )

    summarizer = agent.summarizer or agent.llm
    if hasattr(summarizer, "ainvoke"):
        response = await summarizer.ainvoke(summarizer_input)
    else:
        response = summarizer.invoke(summarizer_input)

    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_prune]
    logger.info("Summarization complete — pruned %d messages, keeping SystemMessage + last %d",
                len(delete_messages), keep_n)
    return {"summary": response.content, "messages": delete_messages}


async def tool_node(agent, state: AgentState) -> dict:
    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    timeout = state.tool_timeout
    session_id = getattr(state, "session_id", "default")

    old_ctx = state.running_context or ""
    if len(old_ctx) > 6000 and session_id not in agent._pending_ctx_compressions:
        agent._pending_ctx_compressions[session_id] = asyncio.create_task(
            compress_running_context(agent, old_ctx)
        )
        logger.debug("tool_node: launched background context compression (%d chars)", len(old_ctx))

    _ctx_token = _current_session_id.set(session_id)
    try:
        results = await execute_tool_calls(agent, tool_calls, timeout)
    finally:
        _current_session_id.reset(_ctx_token)

    if not results:
        return {"messages": []}

    section = "\n".join(
        f"[{msg.name}] → {msg.content}"
        for msg in results
        if hasattr(msg, "name") and msg.name
    )
    logger.debug("tool_node: running_context updated (%d chars): %.500s", len(section), section)

    updated_notepad = agent._session_notepads.get(session_id)
    notepad_update = {"session_notepad": updated_notepad} if updated_notepad else {}
    return {"messages": list(results), "running_context": section, **notepad_update}


def post_tool_router(state: AgentState) -> Literal["llm_call"]:
    return "llm_call"


def pre_llm_router(state: AgentState) -> Literal["summarize_conversation", "llm_call"]:
    summary_text = state.summary or ""
    threshold = int(state.max_context_tokens * 0.8)
    est = _estimate_token_count(state.messages, summary_text)
    needs_summarization = (
        state.enable_summarization
        and (
            len(state.messages) > state.keep_last_n_messages
            or est > threshold
        )
    )
    if needs_summarization:
        logger.debug("Pre-LLM routing → summarize_conversation (messages=%d, est_tokens=%d, threshold=%d)",
                     len(state.messages), est, threshold)
        return "summarize_conversation"
    logger.debug("Pre-LLM routing → llm_call")
    return "llm_call"


def should_continue(state: AgentState) -> "Literal['tool_node', 'llm_call', '__end__'] | Command":
    if state.iteration >= state.max_iterations:
        logger.warning("Iteration limit reached (%d), stopping agent", state.max_iterations)
        return END

    last_message = state.messages[-1]
    has_tool_calls = bool(getattr(last_message, "tool_calls", None))
    if has_tool_calls:
        logger.debug("Routing → tool_node")
        return "tool_node"

    from agent_sdk.agents.response_validator import validate_response, build_correction_hint
    tool_calls_made = sum(1 for m in state.messages if getattr(m, "tool_calls", None))
    response_text = last_message.content if hasattr(last_message, "content") else ""
    issues = validate_response(response_text, tool_calls_made=tool_calls_made, require_citations=False)

    already_retried = getattr(state, "validation_retried", False)
    if issues and not already_retried and state.iteration < state.max_iterations - 1:
        hint = build_correction_hint(issues)
        logger.info("Response quality issues — routing back to llm_call with correction hint")
        return Command(
            goto="llm_call",
            update={"validation_hint": hint, "validation_retried": True},
        )

    if issues:
        logger.warning("Response quality issues detected (no budget for retry): %s", issues)
    logger.debug("Routing → END")
    return END


# ─── Standard Mode Orchestration ───────────────────────────────────────────

_STANDARD_ORCHESTRATOR_PROMPT = """\
You are a task planner for a financial and research AI assistant.

{tool_catalog}

Your job: write a clear, numbered execution plan for answering the user's query.

For each step:
- State what information you need
- Specify which tool(s) to call and with what exact arguments
- Group steps that are independent (can run at the same time) on the same line

Be specific: use exact ticker symbols (e.g. RELIANCE.NS), exact search queries (e.g. "RBI repo rate April 2026"), exact function names.

If the query can be answered without tools, write: "Answer directly — no tools needed."

Keep the plan concise — the executor will see it at every step and use it as a checklist.

CRITICAL: Do NOT include any step that asks the user for clarification, confirmation, or \
disambiguation. If the query is ambiguous, make the most reasonable interpretation and \
proceed directly with tool calls.
"""


async def orchestrate(agent, state: AgentState) -> dict:
    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = _strip_context_block(msg.content)
            break

    if not user_query:
        return {}

    words = user_query.split()
    if len(words) <= 8 and "\n" not in user_query:
        logger.debug("orchestrate: trivial query (%d words) — skipping plan", len(words))
        return {}

    tools = list(agent.tools_by_name.values())
    if not tools:
        return {}

    tool_names = "\n".join(f"- {getattr(t, 'name', str(t))}" for t in tools)
    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        llm = get_llm(state.model_id)
    else:
        llm = agent.llm

    try:
        _t0 = time.monotonic()
        response = await invoke_with_retry(llm, [
            SystemMessage(content=_STANDARD_ORCHESTRATOR_PROMPT.format(tool_catalog=tool_names)),
            HumanMessage(content=f"Query: {user_query}"),
        ])
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase="orchestrate").observe(
            time.monotonic() - _t0
        )
        plan_text = (response.content or "").strip()
        if plan_text:
            logger.info("orchestrate: plan written (%d chars)", len(plan_text))
            return {"scratchpad": plan_text}
    except Exception:
        logger.exception("orchestrate: planning failed — proceeding without plan")
    return {}


# ─── Financial Pipeline ─────────────────────────────────────────────────────

async def financial_initialize(state) -> dict:
    return await initialize(state)



async def synthesis_node(agent, state) -> dict:
    """Synthesise the final response from every populated workspace key.

    Replaces the legacy phase-output synthesis: instead of reading
    `state.phase_outputs`, this reads each entry referenced by
    `workspace_populated` from the request-scoped WorkspaceStore and hands the
    rendered context to the registry's `synthesis` sub-agent.
    """
    from agent_sdk.sub_agents.registry import SUB_AGENT_REGISTRY
    from agent_sdk.sub_agents.base import SubAgentInput

    logger.info("Running synthesis sub-agent")
    synthesis_agent = SUB_AGENT_REGISTRY["synthesis"]
    workspace_store = getattr(agent, "workspace_store", None)
    sub_agent_cache = getattr(agent, "sub_agent_cache", None)
    workspace_id = state_field(state, "workspace_id", "")
    populated = state_field(state, "workspace_populated", set()) or set()

    context_parts: list[str] = []
    if workspace_store is not None:
        for key in populated:
            val = await workspace_store.read(workspace_id, key)
            if val:
                context_parts.append(f"[{key.upper()}]\n{val.get('findings', '')}")

    messages = state_field(state, "messages", []) or []
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content or ""
            break
    entities = state_field(state, "entities", []) or []

    inp = SubAgentInput(
        query=query,
        entities=list(entities),
        workspace_context="\n\n".join(context_parts),
    )

    if workspace_store is None or sub_agent_cache is None:
        logger.warning("synthesis_node: missing workspace_store or sub_agent_cache — emitting fallback")
        return {"messages": [AIMessage(content="(synthesis unavailable: workspace not initialised)")]}

    try:
        out = await synthesis_agent.run(inp, workspace_store, workspace_id, sub_agent_cache)
    except Exception:
        logger.exception("synthesis_node: synthesis sub-agent failed")
        return {"messages": [AIMessage(content="(synthesis failed — see logs)")]}

    return {"messages": [AIMessage(content=out.findings)]}


async def compliance_node(agent, state) -> dict:
    """Run the compliance sub-agent against the synthesis output."""
    from agent_sdk.sub_agents.registry import SUB_AGENT_REGISTRY
    from agent_sdk.sub_agents.base import SubAgentInput

    compliance_agent = SUB_AGENT_REGISTRY["compliance"]
    workspace_store = getattr(agent, "workspace_store", None)
    sub_agent_cache = getattr(agent, "sub_agent_cache", None)
    workspace_id = state_field(state, "workspace_id", "")

    if workspace_store is None or sub_agent_cache is None:
        logger.warning("compliance_node: missing workspace_store or sub_agent_cache — passing through")
        return {}

    synthesis_out = await workspace_store.read(workspace_id, "synthesis") or {}
    messages = state_field(state, "messages", []) or []
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content or ""
            break

    inp = SubAgentInput(
        query=query,
        entities=[],
        workspace_context=synthesis_out.get("findings", ""),
    )
    try:
        out = await compliance_agent.run(inp, workspace_store, workspace_id, sub_agent_cache)
    except Exception:
        logger.exception("compliance_node: compliance sub-agent failed")
        return {}

    return {"messages": [AIMessage(content=out.findings)]}


async def jargon_simplifier_node(agent, state) -> dict:
    """Rewrite the compliance-cleared response for beginner/intermediate users."""
    from agent_sdk.sub_agents.registry import SUB_AGENT_REGISTRY
    from agent_sdk.sub_agents.base import SubAgentInput

    simplifier = SUB_AGENT_REGISTRY["jargon_simplifier"]
    workspace_store = getattr(agent, "workspace_store", None)
    sub_agent_cache = getattr(agent, "sub_agent_cache", None)
    workspace_id = state_field(state, "workspace_id", "")

    if workspace_store is None or sub_agent_cache is None:
        logger.warning("jargon_simplifier_node: missing workspace_store or sub_agent_cache — passing through")
        return {}

    compliance_out = await workspace_store.read(workspace_id, "compliance") or {}
    inp = SubAgentInput(
        query="",
        entities=[],
        workspace_context=compliance_out.get("findings", ""),
    )
    try:
        out = await simplifier.run(inp, workspace_store, workspace_id, sub_agent_cache)
    except Exception:
        logger.exception("jargon_simplifier_node: simplifier sub-agent failed")
        return {}

    return {"messages": [AIMessage(content=out.findings)]}


async def workspace_flush_node(agent, state) -> dict:
    """Drop all workspace keys after the response has been built."""
    workspace_store = getattr(agent, "workspace_store", None)
    workspace_id = state_field(state, "workspace_id", "")
    if workspace_store is None or not workspace_id:
        return {}
    try:
        await workspace_store.flush(workspace_id)
    except Exception:
        logger.exception("workspace_flush_node: flush failed for workspace_id=%s", workspace_id)
    return {}
