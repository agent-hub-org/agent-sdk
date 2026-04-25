from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from typing import Any, Literal, Sequence

# ContextVar so notepad tool closures know which session they're running in
# without needing session_id passed as an explicit argument.
_current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "agent_sdk_session_id", default="default"
)

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from agent_sdk.agents.state import AgentState, FinancialAnalysisState
from agent_sdk.config import settings
from agent_sdk.mcp.exceptions import MCPSessionError
from agent_sdk.metrics import llm_call_duration, tool_call_duration

logger = logging.getLogger("agent_sdk.nodes")

# Pre-compiled regexes — defined once at import time, not per-call
_MALFORMED_TOOL_PATTERN = re.compile(r'<function=(\w+)(.*?)</function>', re.DOTALL)
_JSON_FENCE_PATTERN = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)

# Tools that belong to the research agent and must never appear in financial analyst phases
_RESEARCH_ONLY_TOOLS = frozenset({
    "check_papers_in_db",
    "retrieve_papers",
    "hybrid_retrieve_papers",
    "download_and_store_arxiv_papers",
})



def _parse_malformed_tool_call(failed_generation: str) -> list[dict] | None:
    """
    Parse a malformed tool call string like:
      <function=get_ticker_data{"ticker": "RELIANCE.NS"}</function>
    into a list of structured tool call dicts.
    Returns None if parsing fails.
    """
    matches = _MALFORMED_TOOL_PATTERN.findall(failed_generation)

    if not matches:
        return None

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
    """Strip the [CONTEXT]...[/CONTEXT] wrapper prepended by _build_dynamic_context."""
    marker = "[/CONTEXT]"
    if marker in text:
        return text[text.find(marker) + len(marker):].strip()
    return text.strip()


_RETRYABLE_HTTP_CODES = frozenset({429, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    """Return True for transient LLM errors that are safe to retry."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    status = (
        getattr(exc, "status_code", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
    )
    return status in _RETRYABLE_HTTP_CODES


async def _invoke_with_retry(llm_bound, prompt, max_retries: int | None = None, base_delay: float | None = None):
    """Invoke an LLM with exponential backoff on transient errors."""
    import random as _random
    max_retries = max_retries if max_retries is not None else settings.llm_retry_max
    base_delay = base_delay if base_delay is not None else settings.llm_retry_base_delay
    for attempt in range(max_retries):
        try:
            return await llm_bound.ainvoke(prompt)
        except Exception as e:
            if attempt == max_retries - 1 or not _is_retryable(e):
                raise
            delay = min(30.0, base_delay * (2 ** attempt) + _random.uniform(0, 0.5))
            logger.warning(
                "LLM call attempt %d/%d failed (%s) — retrying in %.2fs",
                attempt + 1, max_retries, type(e).__name__, delay,
            )
            await asyncio.sleep(delay)


async def _compress_running_context(agent, text: str) -> str:
    """Compress running_context to ~2000 chars, preserving all key findings.
    Runs as a background task during tool execution so it adds zero user-facing latency."""
    summarizer = getattr(agent, "summarizer", None) or getattr(agent, "llm", None)
    if summarizer is None:
        return text
    try:
        resp = await summarizer.ainvoke([
            SystemMessage(content=(
                "Compress the following agent work log to ≤2000 chars. "
                "Preserve ALL entity names, numbers, dates, and key findings exactly. "
                "Discard verbose raw tool output prose. Output the compressed log only."
            )),
            HumanMessage(content=text[:16000]),
        ])
        logger.debug("running_context compressed: %d → %d chars", len(text), len(resp.content))
        return resp.content
    except Exception:
        logger.warning("running_context compression failed — using original")
        return text


async def initialize(state: AgentState) -> dict:
    """
    Runs once at START before the agent loop.
    Inserts the system prompt as the first message in state so it is
    persisted by the checkpointer and never re-sent on subsequent LLM calls.
    If no system_prompt is provided, a default is used.

    Does not require agent dependencies — stays a plain function.
    """
    # Clear per-request transient fields at the start of every turn
    _clear: dict = {"running_context": None, "scratchpad": None}

    # If a SystemMessage already exists, check whether it needs updating
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
    """
    Core autonomous agent step:
    - Merge existing conversation summary (if any) into the system prompt.
    - Call the LLM (bound with tools) with the persisted messages.
    - Return a new assistant message (which may include tool calls).
    - Increment the iteration counter.

    `agent` is bound via functools.partial at graph build time.
    """

    logger.info("LLM call — iteration %d/%d, message count: %d, model_id: %s",
                state.iteration + 1, state.max_iterations, len(state.messages),
                state.model_id or "default")

    # Use dynamic model if model_id is set, otherwise fall back to agent.llm
    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        llm = get_llm(state.model_id)
    else:
        llm = agent.llm

    tools = list(agent.tools_by_name.values())
    llm_with_tools = agent.get_bound_llm(llm, tools)

    # Build appended context sections (summary + plan + running_context + perspective)
    extra_sections: list[str] = []

    if state.summary:
        extra_sections.append(
            "INTERNAL CONTEXT (for your reference only — do NOT include, repeat, "
            "or paraphrase any of this in your response to the user):\n"
            f"Previous conversation summary: {state.summary}"
        )

    # Inject the execution plan (scratchpad) so the agent knows what it set out to do
    plan = getattr(state, "scratchpad", None)
    if plan:
        extra_sections.append(
            "EXECUTION PLAN (your task checklist — follow this, don't re-plan):\n" + plan
        )

    # Inject accumulated work done so far this request.
    # If a background compression task is ready, use the compressed version and update state.
    running_ctx = getattr(state, "running_context", None)
    ctx_reset_update: dict = {}
    session_id_for_ctx = getattr(state, "session_id", "default")
    if session_id_for_ctx in agent._pending_ctx_compressions:
        task = agent._pending_ctx_compressions.pop(session_id_for_ctx)
        compressed = await task  # instant if tool calls already finished
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

    # Session notepad: inject cross-request discoveries so agent doesn't re-derive known facts.
    # Lazy-restore from state if in-memory dict was cleared (e.g. server restart).
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

    # Inject correction hint from response validator when routing back for quality fix.
    validation_hint = getattr(state, "validation_hint", None)
    if validation_hint:
        extra_sections.insert(0, validation_hint)
        # Clear the hint so it only appears once — update happens via returned state dict below

    # CoT nudge on the very first iteration when no tool results exist yet.
    # Encourages the agent to plan once before acting, reducing wasted tool calls.
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
        response = await _invoke_with_retry(llm_with_tools, prompt)
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        _phase = getattr(state, "current_phase", None) or "standard"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase=_phase).observe(time.monotonic() - _t0)
    except Exception as e:
        # Handle malformed tool calls from Groq (and similar providers)
        # The model generates text like <function=name{...}</function> instead of
        # structured tool calls, causing a 400 error server-side. We parse the
        # failed generation and construct a proper AIMessage with tool_calls.
        error_body = getattr(e, "body", None) or {}
        if isinstance(error_body, dict):
            # failed_generation can be top-level or nested under "error"
            failed_gen = error_body.get("failed_generation") or error_body.get("error", {}).get("failed_generation")
        else:
            failed_gen = None

        if failed_gen:
            logger.warning("LLM produced malformed tool call, attempting to parse: %s", failed_gen.strip())
            parsed_calls = _parse_malformed_tool_call(failed_gen)

            if parsed_calls:
                # Validate that all parsed tool names actually exist
                valid_calls = [tc for tc in parsed_calls if tc["name"] in agent.tools_by_name]
                if valid_calls:
                    logger.info("Recovered %d tool call(s) from malformed generation: %s",
                                len(valid_calls), [tc["name"] for tc in valid_calls])
                    response = AIMessage(
                        content="",
                        tool_calls=valid_calls,
                    )
                    return {
                        "messages": [response],
                        "iteration": state.iteration + 1,
                    }

            # No tool calls found — the model produced a plain-text response but
            # triggered a 400 (common with some Groq models when tools are bound).
            # Treat the failed_generation as the model's intended text response.
            logger.warning("No tool calls in failed generation — recovering as plain text response (%d chars)",
                           len(failed_gen))
            response = AIMessage(content=failed_gen)
            return {
                "messages": [response],
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
    # Clear validation_hint after consuming it so it doesn't repeat on next iteration.
    if getattr(state, "validation_hint", None):
        result["validation_hint"] = None
    return result


def _strip_dangling_tool_calls(messages: list) -> list:
    """Remove AIMessages with unanswered tool_calls from a message list.

    The OpenAI API rejects any prompt where an assistant message with tool_calls
    is not immediately followed by ToolMessages for every tool_call_id. This happens
    when the iteration limit cuts off execution before tool responses arrive.
    """
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


async def summarize_conversation(agent, state: AgentState) -> dict:
    """
    Summarize older messages and prune them from state.
    - Extends any existing summary with new conversation turns.
    - Deletes all but the most recent `keep_last_n_messages` messages via RemoveMessage.

    `agent` is bound via functools.partial at graph build time.
    """

    logger.info("Summarizing conversation — %d messages, existing summary: %s",
                len(state.messages), "yes" if state.summary else "no")

    keep_n = max(state.keep_last_n_messages, 1)
    all_messages = list(state.messages)

    # Separate the SystemMessage (always at index 0) — it must never be pruned
    has_system = all_messages and isinstance(all_messages[0], SystemMessage)
    system_msg = all_messages[0] if has_system else None
    conversation = all_messages[1:] if has_system else all_messages

    # Split into messages to prune vs. messages to keep
    messages_to_prune = conversation[:-keep_n] if len(conversation) > keep_n else []
    if not messages_to_prune:
        logger.info("Not enough messages to prune, skipping summarization")
        return {}

    # Strip AIMessages with unanswered tool_calls — OpenAI rejects them in the summarizer prompt
    messages_to_prune = _strip_dangling_tool_calls(messages_to_prune)
    if not messages_to_prune:
        logger.info("Nothing to prune after stripping dangling tool_calls, skipping summarization")
        return {}

    # Build summarization prompt from ONLY the messages being pruned
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

    # Delete only the pruned messages — never the SystemMessage or kept messages
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_prune]

    logger.info("Summarization complete — pruned %d messages, keeping SystemMessage + last %d",
                len(delete_messages), keep_n)

    return {"summary": response.content, "messages": delete_messages}


async def _execute_tool_calls(agent, tool_calls: list[dict], timeout: float, phase_tools: list | None = None) -> list[ToolMessage]:
    _lookup = {**agent.tools_by_name, **({t.name: t for t in phase_tools} if phase_tools else {})}

    async def _execute(tool_call: dict) -> tuple[ToolMessage, bool, str]:
        """Execute one tool call. Returns (ToolMessage, needs_summary, raw_content)."""
        name = tool_call["name"]
        args = tool_call.get("args", {})
        logger.info("Executing tool '%s' with args: %s", name, args)

        if name not in _lookup:
            return (
                ToolMessage(
                    content=f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(_lookup.keys()))}",
                    tool_call_id=tool_call["id"],
                ),
                False,
                "",
            )

        tool = _lookup[name]
        breaker = agent._get_breaker(name)

        if breaker.is_open:
            logger.warning("Circuit breaker OPEN for '%s' — returning error message", name)
            return (
                ToolMessage(
                    content=(
                        f"Tool '{name}' is temporarily unavailable (circuit breaker open). "
                        "Please try again later or rephrase your query."
                    ),
                    tool_call_id=tool_call["id"],
                ),
                False,
                "",
            )

        try:
            _tool_t0 = time.monotonic()
            if hasattr(tool, "ainvoke"):
                observation = await tool.ainvoke(args)
            elif hasattr(tool, "arun"):
                observation = await tool.arun(args)
            else:
                import asyncio
                observation = await asyncio.to_thread(
                    tool.invoke if hasattr(tool, "invoke") else tool.run, args
                )
            breaker.record_success()
            tool_call_duration.labels(agent="sdk", tool_name=name).observe(time.monotonic() - _tool_t0)
            obs_str = str(observation)
            raw_len = len(obs_str)
            logger.info("Tool '%s' completed — result length: %d chars", name, raw_len)

            # Flag large, unstructured results for deferred parallel summarization.
            threshold = settings.large_result_threshold
            needs_summary = (
                raw_len > threshold
                and not obs_str.lstrip().startswith(('{', '['))
                and bool(getattr(agent, "summarizer", None) or getattr(agent, "llm", None))
            )
        except Exception as exc:
            breaker.record_failure(name)
            logger.exception("Tool '%s' failed", name)
            return (
                ToolMessage(
                    content=f"Tool '{name}' failed: {exc}. Check the tool schema and retry with correct arguments.",
                    tool_call_id=tool_call["id"],
                ),
                False,
                "",
            )

        return ToolMessage(content=obs_str, tool_call_id=tool_call["id"]), needs_summary, obs_str

    async def _summarize(raw: str, tool_name: str) -> str:
        """Summarize a single large tool result via LLM."""
        summarizer = getattr(agent, "summarizer", None) or getattr(agent, "llm", None)
        try:
            resp = await summarizer.ainvoke([
                SystemMessage(content=(
                    "Summarize the following tool output into a concise but complete summary. "
                    "Preserve ALL specific facts, numbers, names, dates, and URLs. "
                    "Do not omit key information — compress prose, not data."
                )),
                HumanMessage(content=raw[:32000]),
            ])
            logger.info("Tool '%s' result summarized: %d → %d chars", tool_name, len(raw), len(resp.content))
            return resp.content
        except Exception:
            logger.warning("Tool result summarization failed for '%s' — using raw result", tool_name)
            return raw

    async def _execute_with_per_tool_timeout(tool_call: dict) -> tuple[ToolMessage, bool, str]:
        """Wrap _execute with a per-tool timeout so one slow tool cannot block others."""
        name = tool_call.get("name", "unknown")
        per_timeout = settings.per_tool_timeout_map.get(name, settings.per_tool_timeout)
        try:
            return await asyncio.wait_for(_execute(tool_call), timeout=per_timeout)
        except asyncio.TimeoutError:
            logger.warning("Tool '%s' timed out after %.0fs (per-tool limit)", name, per_timeout)
            return (
                ToolMessage(
                    content=f"Tool '{name}' timed out after {per_timeout:.0f}s.",
                    tool_call_id=tool_call["id"],
                ),
                False,
                "",
            )

    async def _gather_with_timeout():
        return await asyncio.wait_for(
            asyncio.gather(*[_execute_with_per_tool_timeout(tc) for tc in tool_calls]),
            timeout=timeout,
        )

    try:
        raw_results = await _gather_with_timeout()
    except Exception as e:
        if isinstance(e, asyncio.TimeoutError):
            logger.error("Tool execution timed out after %.0fs — emitting error ToolMessages", timeout)
            raw_results = [
                (
                    ToolMessage(
                        content=f"Tool execution timed out after {timeout:.0f} seconds.",
                        tool_call_id=tc["id"],
                    ),
                    False,
                    "",
                )
                for tc in tool_calls
            ]
        else:
            # Check if this is an MCP session termination error (typed exception from mcp/client.py)
            if isinstance(e, MCPSessionError):
                logger.warning("MCP session dropped — attempting reconnect and retry")
                if agent._mcp_manager is not None:
                    new_tools = await agent._mcp_manager.reconnect()
                    # Update the agent's tool registry with fresh tool instances
                    agent.tools = list(agent.tools)  # keep non-MCP tools
                    for t in new_tools:
                        agent.tools_by_name[t.name] = t
                    # Invalidate caches that hold stale tool object references
                    agent._phase_tools_cache.clear()
                    agent._bound_llm_cache.clear()
                    # Rebuild the lookup dict so the retry uses the new tool objects
                    _lookup.clear()
                    _lookup.update(agent.tools_by_name)
                    if phase_tools:
                        _lookup.update({t.name: t for t in phase_tools})
                    logger.info("Reconnected — retrying %d tool call(s)", len(tool_calls))
                    raw_results = await _gather_with_timeout()
                else:
                    raise
            else:
                logger.error("Unhandled exception in tool execution — returning error messages", exc_info=True)
                raw_results = [
                    (
                        ToolMessage(
                            content=f"Tool execution failed: {e}",
                            tool_call_id=tc["id"],
                        ),
                        False,
                        "",
                    )
                    for tc in tool_calls
                ]

    # Parallel summarization: collect all large results and summarize concurrently.
    messages = [msg for msg, _, _ in raw_results]
    needs_summary_idx = [
        (i, raw, tc["name"])
        for i, ((_, needs, raw), tc) in enumerate(zip(raw_results, tool_calls))
        if needs
    ]
    if needs_summary_idx:
        summaries = await asyncio.gather(
            *[_summarize(raw, name) for _, raw, name in needs_summary_idx]
        )
        for (i, _, _), summary in zip(needs_summary_idx, summaries):
            messages[i] = ToolMessage(
                content=summary,
                tool_call_id=messages[i].tool_call_id,
            )

    return messages


async def tool_node(agent, state: AgentState) -> dict:
    """
    Execute any tool calls from the last assistant message and
    return the resulting tool messages.

    If an MCP session has dropped (McpError: Session terminated),
    reconnects and retries the failed tool calls once.

    Tool results are also accumulated into state.scratchpad so the memory
    pipeline captures tool findings for all query types (opaque and analytical).

    `agent` is bound via functools.partial at graph build time.
    """

    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    timeout = state.tool_timeout
    session_id = getattr(state, "session_id", "default")

    # Launch running_context compression BEFORE tool execution so it runs concurrently.
    # Tool calls take 5–30s; compression (~1s) finishes during that window at zero cost.
    old_ctx = state.running_context or ""
    if len(old_ctx) > 6000 and session_id not in agent._pending_ctx_compressions:
        agent._pending_ctx_compressions[session_id] = asyncio.create_task(
            _compress_running_context(agent, old_ctx)
        )
        logger.debug("tool_node: launched background context compression (%d chars)", len(old_ctx))

    # Set ContextVar so notepad tool closures know which session they're in
    _ctx_token = _current_session_id.set(session_id)
    try:
        results = await _execute_tool_calls(agent, tool_calls, timeout)
    finally:
        _current_session_id.reset(_ctx_token)

    if not results:
        return {"messages": []}

    # Accumulate tool results into running_context (zero latency — pure string ops)
    section = "\n".join(
        f"[{msg.name}] → {msg.content}"
        for msg in results
        if hasattr(msg, "name") and msg.name
    )
    logger.debug("tool_node: running_context updated (%d chars): %.500s", len(section), section)

    # Sync notepad to state if any write_to_notepad calls were made this batch
    updated_notepad = agent._session_notepads.get(session_id)
    notepad_update = {"session_notepad": updated_notepad} if updated_notepad else {}

    return {"messages": list(results), "running_context": section, **notepad_update}


def post_tool_router(state: AgentState) -> Literal["llm_call"]:
    """
    After tool execution, route back to the LLM.

    Summarization is deferred to memory_writer (runs after the LLM response is
    fully streamed) so it never adds latency to the user-facing response.
    """
    return "llm_call"


def pre_llm_router(state: AgentState) -> Literal["summarize_conversation", "llm_call"]:
    """
    At the start of each conversation turn, pre-emptively summarize if the context
    is approaching the token budget (80% threshold).

    This runs once per turn — before the first LLM call — so it protects against
    hard context-limit errors from the LLM API without causing mid-loop latency spikes.
    post_tool_router no longer triggers summarization; any remaining pruning after
    the turn completes is handled by memory_writer.
    """
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


def _estimate_token_count(messages: Sequence, extra_text: str = "") -> int:
    """Rough token estimate: ~4 chars per token for English text.
    Uses 3 chars/token for ToolMessages (JSON-heavy).

    Pass extra_text for any content injected outside the messages list
    (e.g. state.summary appended to the system prompt) so the estimate
    reflects the real token budget consumed.
    """
    msg_chars = sum(
        len(getattr(m, "content", "") or "") // (3 if isinstance(m, ToolMessage) else 4)
        for m in messages
    )
    return msg_chars + len(extra_text) // 4


def should_continue(state: AgentState) -> "Literal['tool_node', 'llm_call', '__end__'] | Command":
    """
    Decide whether the autonomous agent should keep going:
    - Pending tool calls MUST always be executed first (skipping them leaves
      orphaned tool_calls in the message history which crashes the LLM API).
    - After tool execution (no pending calls), summarize if context is large.
    - If validation fails and budget allows, loop back once with a correction hint.
    - If we've hit the iteration limit or there are no tool calls, stop.

    Does not require agent dependencies — stays a plain function.
    """

    # Stop if we've reached the configured iteration limit
    if state.iteration >= state.max_iterations:
        logger.warning("Iteration limit reached (%d), stopping agent", state.max_iterations)
        return END

    last_message = state.messages[-1]
    has_tool_calls = bool(getattr(last_message, "tool_calls", None))

    # Tool calls MUST be executed before anything else — orphaned tool_calls
    # without matching ToolMessages will cause a 400 from the LLM API.
    if has_tool_calls:
        logger.debug("Routing → tool_node")
        return "tool_node"

    # No tool calls — the LLM produced a final response. Run quality check.
    from agent_sdk.agents.response_validator import validate_response, build_correction_hint
    tool_calls_made = sum(
        1 for m in state.messages
        if getattr(m, "tool_calls", None)
    )
    response_text = last_message.content if hasattr(last_message, "content") else ""
    issues = validate_response(
        response_text,
        tool_calls_made=tool_calls_made,
        require_citations=False,
    )

    # Route back to llm_call with a correction hint if validation failed,
    # budget allows, and we haven't already retried once.
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


# ============================================================================
# ORCHESTRATION — Standard Mode
# ============================================================================

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
    """
    Orchestrate node for standard mode: writes a tool-specific execution plan to scratchpad.
    The llm_call ReAct loop reads scratchpad at every iteration so the LLM never re-plans
    and stops calling tools once the plan is satisfied.

    Trivial queries (≤8 words, single line) skip planning — no latency added.
    """
    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = _strip_context_block(msg.content)
            break

    if not user_query:
        return {}

    # Fast exit: trivial queries don't need a plan
    words = user_query.split()
    if len(words) <= 8 and "\n" not in user_query:
        logger.debug("orchestrate: trivial query (%d words) — skipping plan", len(words))
        return {}

    tools = list(agent.tools_by_name.values())
    if not tools:
        return {}

    tool_names = "\n".join(f"- {getattr(t, 'name', str(t))}" for t in tools)
    llm = _get_phase_llm(agent, state)

    try:
        _t0 = time.monotonic()
        response = await _invoke_with_retry(llm, [
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


# ============================================================================
# FINANCIAL REASONING PIPELINE
# ============================================================================
# Phase executors use native tool binding (no JSON-in-text extraction).
# Every LLM call within a phase sees scratchpad (plan) + running_context (prior work).
# Phase results are accumulated as prose in running_context.
# ============================================================================


async def financial_initialize(state) -> dict:
    """Initialize the financial reasoning pipeline. Delegates to standard initialize."""
    return await initialize(state)


async def financial_orchestrate(agent, state) -> dict:
    """
    Orchestrate node for financial mode: classifies the query, determines phases,
    and writes a comprehensive tool-specific plan — all in a SINGLE LLM call.

    Previously two sequential LLM calls (classify → plan); merged into one combined
    call using FINANCIAL_ORCHESTRATE_COMBINED_PROMPT to eliminate +2-6s of latency.
    Prompt is built dynamically from PHASE_REGISTRY so tool hints never drift.
    """
    from agent_sdk.financial.prompts import FINANCIAL_ORCHESTRATE_COMBINED_PROMPT
    from agent_sdk.financial.schemas import QueryClassification, QueryType
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY

    logger.info("Classifying query and building plan for financial reasoning pipeline")

    llm = _get_phase_llm(agent, state)

    # Build user content for combined prompt
    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    clean_query = _strip_context_block(user_query)

    # Include recent conversational context so classifier can resolve follow-ups
    recent_context: list[str] = []
    for msg in state.messages[:-1]:
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content:
            recent_context.append(f"Assistant: {msg.content}")
        elif isinstance(msg, HumanMessage):
            content = _strip_context_block(msg.content)
            if content:
                recent_context.append(f"User: {content}")
    recent_context = recent_context[-4:]

    if recent_context:
        context_str = "\n".join(recent_context)
        user_content = (
            f"Recent conversation:\n{context_str}\n\n"
            f"Query: {clean_query}"
        )
    else:
        user_content = f"Query: {clean_query}"

    _valid_phases = set(PHASE_REGISTRY.keys()) | {"comparative_analysis"}

    _default_qc = QueryClassification(
        query_type=QueryType.DATA_RETRIEVAL,
        phases=["company_analysis"],
        reasoning="Classification failed — running minimal pipeline",
    )

    qc = _default_qc
    plan_text = ""
    try:
        response = await _invoke_with_retry(llm, [
            SystemMessage(content=FINANCIAL_ORCHESTRATE_COMBINED_PROMPT),
            HumanMessage(content=user_content),
        ])
        combined = _extract_json(response.content)
        if combined:
            plan_obj = combined.pop("plan", "")
            if isinstance(plan_obj, list):
                plan_text = "\n".join(str(step) for step in plan_obj).strip()
            else:
                plan_text = str(plan_obj).strip()
            try:
                normalized = _normalize_classification(combined)
                qc = QueryClassification(**normalized)
            except Exception:
                logger.warning("Could not build QueryClassification from combined response — using default")
        else:
            logger.warning("Combined orchestrate response missing JSON — using default classification")
    except Exception:
        logger.exception("financial_orchestrate: combined LLM call failed — using defaults")

    # Build phases list: use qc.phases if the LLM returned the new format,
    # fall back to the legacy requires_X booleans if present, then append synthesis.
    phases: list[str] = []
    if qc.query_type == QueryType.DATA_RETRIEVAL:
        phases = ["company_analysis"]
    elif qc.query_type == QueryType.COMPARATIVE:
        phases = ["comparative_analysis"]
    elif qc.phases:
        # New format: LLM returned a validated phases list directly.
        phases = [p for p in qc.phases if p in _valid_phases]
    # synthesis is always the terminal phase (appended unconditionally)
    if phases:
        phases.append("synthesis")

    entities = list(qc.entities) if hasattr(qc, "entities") and qc.entities else []
    logger.info("financial_orchestrate: type=%s phases=%s entities=%s plan=%d chars",
                qc.query_type.value, phases, entities, len(plan_text))

    return {
        "scratchpad": plan_text if plan_text else None,
        "phases_to_run": phases,
        "current_phase": phases[0] if phases else "done",
        "query_type": qc.query_type.value,
        "entities": entities,
        "iteration": state.iteration + 1,
    }


async def synthesis_node(agent, state) -> dict:
    """
    Synthesis phase — produce the final user-facing report.

    Prefers state.phase_outputs (typed, structured) over the legacy running_context
    string when available.  Falls back to running_context for backward compat.
    """
    from agent_sdk.financial.prompts import SYNTHESIS_PROMPT, COMPARATIVE_SYNTHESIS_PROMPT
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY
    from datetime import datetime, timezone

    logger.info("Running synthesis phase")
    llm = _get_phase_llm(agent, state)

    query_type = getattr(state, "query_type", None)

    if query_type == "comparative":
        synthesis_prompt = COMPARATIVE_SYNTHESIS_PROMPT
    else:
        synthesis_prompt = SYNTHESIS_PROMPT

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_context = f"\n\nTODAY'S DATE: {today}"

    # Build prior analysis section from typed phase_outputs (preferred) or running_context.
    phase_outputs = getattr(state, "phase_outputs", None) or {}
    if phase_outputs:
        # Render each phase section in pipeline order (registry order = dependency order)
        sections: list[str] = []
        for phase_name in PHASE_REGISTRY:
            po = phase_outputs.get(phase_name)
            if po is None:
                continue
            label = phase_name.upper().replace("_", " ")
            if po.confidence < 0.5:
                sections.append(f"=== {label} ===\n(Phase failed or returned low-confidence data)\n=== END {label} ===")
            else:
                sections.append(f"=== {label} ===\n{po.findings}\n=== END {label} ===")
        # Also include any entity_analysis outputs (comparative mode)
        for key, po in phase_outputs.items():
            if key.startswith("entity_analysis") or (key not in PHASE_REGISTRY):
                label = key.upper().replace("_", " ")
                sections.append(f"=== {label} ===\n{po.findings}\n=== END {label} ===")
        prior_analysis = "\n\n".join(sections) if sections else "(no prior phase results available)"
    else:
        # Fallback: legacy running_context string
        prior_analysis = getattr(state, "running_context", None) or "(no prior phase results available)"

    full_system = (
        synthesis_prompt
        + date_context
        + "\n\n=== PRIOR ANALYSIS ===\n"
        + prior_analysis
        + "\n=== END PRIOR ANALYSIS ==="
    )

    # Build message list — only the current HumanMessage (no prior AI phase outputs)
    user_msg = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_msg = msg
            break

    prompt = [
        SystemMessage(content=full_system),
        *([] if user_msg is None else [user_msg]),
    ]

    _synthesis_timeout = state.tool_timeout
    try:
        response = await asyncio.wait_for(
            _invoke_with_retry(llm, prompt), timeout=_synthesis_timeout
        )
    except asyncio.TimeoutError:
        logger.error("Synthesis LLM call timed out after %.0fs", _synthesis_timeout)
        fallback_content = "Analysis timed out during synthesis. Please try a simpler or more specific query."
        return {
            "messages": [AIMessage(content=fallback_content)],
            "iteration": state.iteration + 1,
        }

    report_msg = AIMessage(content=response.content)
    return {
        "messages": [report_msg],
        "iteration": state.iteration + 1,
    }


# phase_router, phase_advance, and parallel_fan_in have been replaced by the
# dependency-aware phase_scheduler in graph.py (Phase 4 refactor).




# ---------------------------------------------------------------------------
# Financial Pipeline Helpers
# ---------------------------------------------------------------------------

def _get_phase_llm(agent, state):
    """Get the LLM for the current phase, supporting model_id override."""
    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        return get_llm(state.model_id)
    return agent.llm


def _get_phase_tools(agent, phase: str) -> list:
    """
    Get tools appropriate for a specific reasoning phase.

    Phase-specific financial module tools come from PHASE_REGISTRY[phase].financial_tool_names.
    All agent MCP tools (minus research-only ones) are always included.

    Results are cached per (phase, agent_tools_fingerprint) on the agent instance.
    Cache is invalidated on MCP reconnection by clearing agent._phase_tools_cache.
    """
    tool_fingerprint = frozenset(agent.tools_by_name.keys())
    cache_key = (phase, tool_fingerprint)
    cached = agent._phase_tools_cache.get(cache_key)
    if cached is not None:
        return cached

    from agent_sdk.financial.causal_graph import get_causal_graph_tools
    from agent_sdk.financial.ontology import get_ontology_tools
    from agent_sdk.financial.quant_tools import get_quant_tools
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY

    # Build a flat lookup of all financial module tools (computed once per cache miss)
    all_financial: dict[str, object] = {}
    for t in get_quant_tools() + get_causal_graph_tools() + get_ontology_tools():
        all_financial[t.name] = t

    # Pick only the tools listed in the registry for this phase
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


def _build_phase_prompt(state, phase_system_prompt: str) -> list:
    """Build the message list for a phase LLM call, injecting the phase system prompt."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    # Override search year if a historical reference date is provided
    if state.as_of_date:
        year = state.as_of_date[:4]
        date_context = (
            f"\n\nTODAY'S DATE: {today}. HISTORICAL REFERENCE DATE: {state.as_of_date}\n"
            f"Always include the historical year ({year}) in search queries to get results "
            f"relevant to that specific point in time."
        )
    else:
        year = datetime.now(timezone.utc).year
        date_context = (
            f"\n\nTODAY'S DATE: {today}\n"
            f"Always include the current year ({year}) in search queries to get up-to-date results."
        )

    messages = [SystemMessage(content=phase_system_prompt + date_context)]

    # Include all HumanMessages and tool call/result pairs.
    #
    # For AIMessages: prior-turn conversational responses (those appearing BEFORE the last
    # HumanMessage) are included so the LLM can resolve follow-ups like "Yes" back to the
    # question it just asked. Same-turn phase pipeline outputs (those appearing AFTER the
    # last HumanMessage) are still excluded to prevent "opinion contamination"
    # (e.g., regime_assessment's narrative leaking into company_analysis).
    #
    # Safety: strip any AIMessage(tool_calls) that has no matching ToolMessage —
    # orphaned tool_calls cause OpenAI to return a 400 error.
    responded_ids = {
        msg.tool_call_id
        for msg in state.messages
        if isinstance(msg, ToolMessage) and msg.tool_call_id
    }

    # Index of the last HumanMessage — discriminates prior-turn vs same-turn AIMessages.
    last_human_idx = max(
        (i for i, m in enumerate(state.messages) if isinstance(m, HumanMessage)),
        default=-1,
    )

    for i, msg in enumerate(state.messages):
        if isinstance(msg, HumanMessage):
            messages.append(msg)  # Include ALL HumanMessages (not just the first)
        elif isinstance(msg, ToolMessage):
            messages.append(msg)
        elif isinstance(msg, AIMessage):
            if getattr(msg, "tool_calls", None):
                pending = [tc["id"] for tc in msg.tool_calls if tc["id"] not in responded_ids]
                if pending:
                    logger.warning(
                        "_build_phase_prompt: dropping AIMessage with %d unmatched tool_call_id(s) %s "
                        "to prevent OpenAI 400",
                        len(pending), pending,
                    )
                else:
                    messages.append(msg)
            elif msg.content and i < last_human_idx:
                # Conversational AIMessage from a prior turn — include for follow-up context.
                messages.append(msg)
            # AIMessages after the last HumanMessage are same-turn phase outputs — skip.

    return messages


def _format_context(data: dict | None) -> str:
    """Format a dict as readable context for injection into prompts."""
    if not data:
        return "(not yet assessed)"
    import json
    try:
        return json.dumps(data, indent=2, default=str)
    except (TypeError, ValueError):
        return str(data)


def _fix_json_control_chars(s: str) -> str:
    """Escape literal \\n/\\r/\\t inside JSON string values.

    LLMs sometimes emit raw newlines inside JSON string values instead of the
    required \\n escape sequences, causing json.loads to fail.  This helper
    walks the string character-by-character and fixes those characters only
    when inside a quoted JSON string context.
    """
    out: list[str] = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            out.append(ch)
            escape_next = False
        elif ch == '\\':
            out.append(ch)
            escape_next = True
        elif ch == '"':
            out.append(ch)
            in_string = not in_string
        elif in_string and ch == '\n':
            out.append('\\n')
        elif in_string and ch == '\r':
            out.append('\\r')
        elif in_string and ch == '\t':
            out.append('\\t')
        else:
            out.append(ch)
    return ''.join(out)


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from LLM text output."""
    # Try the whole text as JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            return json.loads(_fix_json_control_chars(text))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try to find JSON block in markdown code fences
    matches = _JSON_FENCE_PATTERN.findall(text)
    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            try:
                return json.loads(_fix_json_control_chars(match.strip()))
            except json.JSONDecodeError:
                continue

    # Try to find JSON objects in the text — collect all valid candidates,
    # then return the one with the most keys (most likely the intended phase output).
    candidates: list[dict] = []
    brace_depth = 0
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        candidates.append(obj)
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(_fix_json_control_chars(text[start:i + 1]))
                        if isinstance(obj, dict):
                            candidates.append(obj)
                    except json.JSONDecodeError:
                        pass
                start = None
    if candidates:
        # Heuristic: prefer candidates with string-valued keys (phase outputs have
        # descriptive keys), and among those pick the one with the most keys.
        # Skip tiny objects (e.g., {"type": "tool_call"}) that are likely not phase output.
        scored = []
        for obj in candidates:
            string_keys = sum(1 for v in obj.values() if isinstance(v, str) and len(v) > 10)
            score = len(obj) + string_keys * 2
            scored.append((score, obj))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    logger.warning(
        "_extract_json: all parsing strategies failed (text length=%d, first 500 chars: '%s')",
        len(text), text[:500],
    )
    return None


def _normalize_classification(raw: dict) -> dict:
    """Remap common LLM field-name variations to QueryClassification fields.

    Supports both the new format (phases: list[str]) and the legacy format
    (requires_X_assessment: bool) for backward compatibility during rollout.
    """
    normalized = dict(raw)

    # Remap query_type aliases
    for alias in ("type", "classification", "category"):
        if alias in normalized and "query_type" not in normalized:
            normalized["query_type"] = normalized.pop(alias)

    # Remap reasoning aliases
    for alias in ("reason", "explanation"):
        if alias in normalized and "reasoning" not in normalized:
            normalized["reasoning"] = normalized.pop(alias)

    # Handle legacy requires_X fields: convert to phases list if phases not already present
    _phase_bool_map = {
        "requires_regime_assessment": "regime_assessment",
        "requires_causal_analysis": "causal_analysis",
        "requires_sector_analysis": "sector_analysis",
        "requires_company_analysis": "company_analysis",
        "requires_risk_assessment": "risk_assessment",
    }
    has_legacy_bools = any(k in normalized for k in _phase_bool_map)
    has_phases_key = "phases" in normalized or "reasoning_phases" in normalized

    if has_legacy_bools and not has_phases_key:
        # Convert legacy boolean flags to phases list in registry order
        _registry_order = [
            "regime_assessment", "causal_analysis",
            "sector_analysis", "company_analysis", "risk_assessment",
        ]
        phases_from_bools = [
            phase for phase in _registry_order
            if normalized.get(f"requires_{phase}", False)
        ]
        normalized["phases"] = phases_from_bools

    # If phases came as "reasoning_phases" key, rename it
    if "reasoning_phases" in normalized and "phases" not in normalized:
        normalized["phases"] = normalized.pop("reasoning_phases")

    # Strip unknown keys to avoid Pydantic extra-field errors
    valid_keys = {"query_type", "entities", "phases", "reasoning"}
    return {k: v for k, v in normalized.items() if k in valid_keys}




def _format_tool_catalog(tools: list) -> str:
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






# ============================================================================
# MEMORY NODES
# ============================================================================

_PERSPECTIVE_INJECTION_HEADER = """\
[USER PERSONALITY BACKGROUND]
The following is background context about this user's communication style and preferences.
Use it ONLY to adapt tone, vocabulary, and level of detail in your responses.
Do NOT use this to modify analytical conclusions, filter information, add/remove facts,
or influence task planning in any way. It is read-only personality context.

{perspective}
[/USER PERSONALITY BACKGROUND]"""


async def load_user_context(agent, state: AgentState) -> dict:
    """
    Load perspective memory (Mem0) and semantic memory (Pinecone user-knowledge) for the current
    user and inject as background context into state.perspective_context.

    Both fetches run in parallel. No-op if neither memory tier is configured or user_id is None.
    """
    memory_manager = getattr(agent, "memory_manager", None)
    semantic_memory = getattr(agent, "semantic_memory", None)
    user_id = getattr(state, "user_id", None)

    if not user_id or (memory_manager is None and semantic_memory is None):
        return {}

    # Extract the last user message as query context for semantic retrieval
    last_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_query = _strip_context_block(msg.content)[:200]
            break

    # Run Mem0 + semantic memory in parallel
    async def _get_perspective():
        if memory_manager is None:
            return None
        return await memory_manager.get_perspective(user_id)

    async def _get_semantic():
        if semantic_memory is None:
            return []
        return await semantic_memory.retrieve(user_id, last_query or "user preferences and profile")

    perspective, semantic_facts = await asyncio.gather(
        _get_perspective(), _get_semantic(), return_exceptions=True
    )

    parts = []
    if isinstance(perspective, str) and perspective:
        parts.append(perspective)
    if isinstance(semantic_facts, list) and semantic_facts:
        facts_text = "\n".join(f"• {f}" for f in semantic_facts)
        parts.append(f"Semantic memory (durable facts from past sessions):\n{facts_text}")

    if not parts:
        return {}

    combined = "\n\n".join(parts)
    logger.debug("load_user_context: injecting perspective + %d semantic facts for user=%s",
                 len(semantic_facts) if isinstance(semantic_facts, list) else 0, user_id)
    return {
        "perspective_context": _PERSPECTIVE_INJECTION_HEADER.format(perspective=combined),
    }


async def memory_writer(agent, state: AgentState) -> dict:
    """
    Terminal node: runs deferred summarization then fires the Mem0 pipeline.

    Summarization runs here (not inline in pre/post_tool_router) so it never
    adds latency to the user-facing streaming response — the LLM output is
    fully streamed before this node executes.  The state update (pruned messages
    + new summary) is returned so LangGraph persists it for the next request.

    The Mem0 pipeline is still fire-and-forget via asyncio.create_task.
    """
    # ── Deferred context summarization ──────────────────────────────────────
    # Runs synchronously here because: (a) streaming is already done at this
    # point, and (b) this node's return dict is what gets checkpointed, so the
    # pruned-message state is correctly persisted for the next request.
    summarization_update: dict = {}
    if getattr(state, "enable_summarization", False):
        summary_text = state.summary or ""
        est = _estimate_token_count(state.messages, summary_text)
        needs_summarization = (
            len(state.messages) > state.keep_last_n_messages
            or est > state.max_context_tokens
        )
        if needs_summarization:
            logger.debug("memory_writer: running deferred summarization (messages=%d, est_tokens=%d)",
                         len(state.messages), est)
            summarization_update = await summarize_conversation(agent, state)

    # ── Mem0 long-term memory pipeline ──────────────────────────────────────
    memory_manager = getattr(agent, "memory_manager", None)
    if memory_manager is None:
        return summarization_update

    user_id = getattr(state, "user_id", None)
    if not user_id:
        return summarization_update

    session_id = getattr(state, "session_id", "default")

    # Extract last user query
    query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            query = _strip_context_block(msg.content)
            break

    # Extract last agent response
    response = ""
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            if isinstance(content, str) and content:
                response = content
                break

    if not query or not response:
        logger.debug("memory_writer: no query or response found — skipping")
        return summarization_update

    scratchpad = getattr(state, "scratchpad", None)
    logger.debug("memory_writer: scratchpad passed to snapshot (%d chars): %.500s",
                 len(scratchpad) if scratchpad else 0, scratchpad or "(none)")

    # Schedule background tasks via _tracked_task so they're awaited on shutdown
    agent._tracked_task(
        memory_manager.process_query(
            user_id=user_id,
            session_id=session_id,
            query=query,
            response=response,
            scratchpad=scratchpad,
            llm=agent.llm,
        )
    )
    logger.debug("memory_writer: background Mem0 task scheduled for user=%s", user_id)

    semantic_memory = getattr(agent, "semantic_memory", None)
    if semantic_memory is not None:
        conv_turns = []
        for msg in state.messages[-20:]:
            if isinstance(msg, HumanMessage):
                conv_turns.append(f"User: {_strip_context_block(msg.content)}")
            elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = msg.content
                if isinstance(content, str) and content:
                    conv_turns.append(f"Agent: {content[:500]}")
        if conv_turns:
            agent._tracked_task(
                semantic_memory.consolidate(user_id, "\n".join(conv_turns), agent.llm)
            )
            logger.debug("memory_writer: background semantic consolidation scheduled for user=%s", user_id)

    return summarization_update
