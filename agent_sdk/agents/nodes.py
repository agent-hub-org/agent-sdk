from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Literal, Sequence

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END

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

    # Inject accumulated work done so far this request
    running_ctx = getattr(state, "running_context", None)
    if running_ctx:
        extra_sections.append(
            "WORK DONE SO FAR (tool results / completed phases — use this, don't re-fetch):\n"
            + running_ctx
        )

    perspective = getattr(state, "perspective_context", None)
    if perspective:
        extra_sections.append(perspective)

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
        if hasattr(llm_with_tools, "ainvoke"):
            response = await llm_with_tools.ainvoke(prompt)
        else:
            response = llm_with_tools.invoke(prompt)
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

    return {
        "messages": [response],
        "iteration": state.iteration + 1,
    }


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

    async def _execute(tool_call: dict) -> ToolMessage:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        logger.info("Executing tool '%s' with args: %s", name, args)

        if name not in _lookup:
            return ToolMessage(
                content=f"Error: unknown tool '{name}'. Available tools: {', '.join(sorted(_lookup.keys()))}",
                tool_call_id=tool_call["id"],
            )

        tool = _lookup[name]
        breaker = agent._get_breaker(name)

        if breaker.is_open:
            logger.warning("Circuit breaker OPEN for '%s' — returning error message", name)
            return ToolMessage(
                content=(
                    f"Tool '{name}' is temporarily unavailable (circuit breaker open). "
                    "Please try again later or rephrase your query."
                ),
                tool_call_id=tool_call["id"],
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
            logger.info("Tool '%s' completed — result length: %d chars", name, len(str(observation)))
        except Exception as exc:
            breaker.record_failure(name)
            logger.exception("Tool '%s' failed", name)
            return ToolMessage(
                content=f"Tool '{name}' failed: {exc}. Check the tool schema and retry with correct arguments.",
                tool_call_id=tool_call["id"],
            )

        return ToolMessage(content=str(observation), tool_call_id=tool_call["id"])

    async def _gather_with_timeout():
        import asyncio
        return await asyncio.wait_for(
            asyncio.gather(*[_execute(tc) for tc in tool_calls]),
            timeout=timeout,
        )

    try:
        results = await _gather_with_timeout()
    except Exception as e:
        import asyncio
        if isinstance(e, asyncio.TimeoutError):
            logger.error("Tool execution timed out after %.0fs — emitting error ToolMessages", timeout)
            results = [
                ToolMessage(
                    content=f"Tool execution timed out after {timeout:.0f} seconds.",
                    tool_call_id=tc["id"],
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
                    # Rebuild the lookup dict so the retry uses the new tool objects
                    _lookup.clear()
                    _lookup.update(agent.tools_by_name)
                    if phase_tools:
                        _lookup.update({t.name: t for t in phase_tools})
                    logger.info("Reconnected — retrying %d tool call(s)", len(tool_calls))
                    results = await _gather_with_timeout()
                else:
                    raise
            else:
                logger.error("Unhandled exception in tool execution — returning error messages", exc_info=True)
                results = [
                    ToolMessage(
                        content=f"Tool execution failed: {e}",
                        tool_call_id=tc["id"],
                    )
                    for tc in tool_calls
                ]

    return list(results)


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

    results = await _execute_tool_calls(agent, tool_calls, timeout)

    if not results:
        return {"messages": []}

    # Accumulate tool results into running_context (zero latency — pure string ops)
    section = "\n".join(
        f"[{msg.name}] → {msg.content}"
        for msg in results
        if hasattr(msg, "name") and msg.name
    )
    logger.debug("tool_node: running_context updated (%d chars): %.500s", len(section), section)

    return {"messages": list(results), "running_context": section}


def post_tool_router(state: AgentState) -> Literal["summarize_conversation", "llm_call"]:
    """
    After tool execution, check if context needs summarization before
    handing back to the LLM. This enables mid-loop summarization without
    ever skipping pending tool calls.
    """
    summary_text = state.summary or ""
    est = _estimate_token_count(state.messages, summary_text)
    needs_summarization = (
        state.enable_summarization
        and (
            len(state.messages) > state.keep_last_n_messages
            or est > state.max_context_tokens
        )
    )
    if needs_summarization:
        logger.debug("Post-tool routing → summarize_conversation (messages=%d, est_tokens=%d)",
                     len(state.messages), est)
        return "summarize_conversation"
    return "llm_call"


def pre_llm_router(state: AgentState) -> Literal["summarize_conversation", "llm_call"]:
    """
    At the start of each conversation turn, check whether the context is already
    large enough to warrant summarization before calling the LLM.

    Threshold is 80% of state.max_context_tokens so summarization is deferred as
    long as possible and respects per-agent token budget configuration.
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

    Pass extra_text for any content injected outside the messages list
    (e.g. state.summary appended to the system prompt) so the estimate
    reflects the real token budget consumed.
    """
    msg_chars = sum(len(getattr(m, "content", "") or "") for m in messages)
    return (msg_chars + len(extra_text)) // 4


def should_continue(state: AgentState) -> Literal["tool_node", "summarize_conversation", "__end__"]:
    """
    Decide whether the autonomous agent should keep going:
    - Pending tool calls MUST always be executed first (skipping them leaves
      orphaned tool_calls in the message history which crashes the LLM API).
    - After tool execution (no pending calls), summarize if context is large.
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

    # No tool calls means the LLM produced a final response — run quality check then stop.
    from agent_sdk.agents.response_validator import validate_response
    tool_calls_made = sum(
        1 for m in state.messages
        if getattr(m, "tool_calls", None)
    )
    issues = validate_response(
        last_message.content if hasattr(last_message, "content") else "",
        tool_calls_made=tool_calls_made,
        require_citations=False,
    )
    if issues:
        logger.warning("Response quality issues detected (will still END): %s", issues)

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

    tool_catalog = agent.get_tool_catalog()
    llm = _get_phase_llm(agent, state)

    try:
        _t0 = time.monotonic()
        response = await llm.ainvoke([
            SystemMessage(content=_STANDARD_ORCHESTRATOR_PROMPT.format(tool_catalog=tool_catalog)),
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
    """
    from agent_sdk.financial.prompts import FINANCIAL_ORCHESTRATE_COMBINED_PROMPT
    from agent_sdk.financial.schemas import QueryClassification, QueryType

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

    _default_qc = QueryClassification(
        query_type=QueryType.DATA_RETRIEVAL,
        requires_regime_assessment=False,
        requires_causal_analysis=False,
        requires_sector_analysis=False,
        requires_company_analysis=True,
        requires_risk_assessment=False,
        reasoning="Classification failed — running minimal pipeline",
    )

    qc = _default_qc
    plan_text = ""
    try:
        response = await llm.ainvoke([
            SystemMessage(content=FINANCIAL_ORCHESTRATE_COMBINED_PROMPT),
            HumanMessage(content=user_content),
        ])
        combined = _extract_json(response.content)
        if combined:
            plan_text = combined.pop("plan", "").strip()
            try:
                qc = QueryClassification(**_normalize_classification(combined))
            except Exception:
                logger.warning("Could not build QueryClassification from combined response — using default")
        else:
            logger.warning("Combined orchestrate response missing JSON — using default classification")
    except Exception:
        logger.exception("financial_orchestrate: combined LLM call failed — using defaults")

    # Determine phases from classification
    phases = []
    if qc.query_type == QueryType.DATA_RETRIEVAL:
        phases = ["company_analysis", "synthesis"]
    elif qc.query_type == QueryType.COMPARATIVE:
        phases = ["comparative_analysis", "synthesis"]
    else:
        if qc.requires_regime_assessment:
            phases.append("regime_assessment")
        if qc.requires_causal_analysis:
            phases.append("causal_analysis")
        if qc.requires_sector_analysis:
            phases.append("sector_analysis")
        if qc.requires_company_analysis:
            phases.append("company_analysis")
        if qc.requires_risk_assessment:
            phases.append("risk_assessment")
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


# Per-phase iteration budgets (safety cap — LLM self-regulates via plan+running_context)
_PHASE_BUDGETS: dict[str, int] = {
    "regime_assessment": 4,
    "causal_analysis": 4,
    "sector_analysis": 4,
    "company_analysis": 6,
    "risk_assessment": 4,
}


async def financial_phase_executor(phase_name: str, agent, state) -> dict:
    """
    ReAct executor for a single financial pipeline phase.

    Uses native LangChain tool binding (llm.bind_tools). Every LLM iteration sees:
    - The agent's system prompt (from state.messages[0])
    - scratchpad: the comprehensive plan written by financial_orchestrate
    - running_context: all prior phase results accumulated as prose

    Tool results are appended to running_context after each tool call so the LLM
    self-regulates: once it sees the plan section for this phase is satisfied it stops.

    Phase findings are appended to running_context as a labeled prose block
    so subsequent phases and synthesis can read them.
    """
    from datetime import datetime, timezone

    logger.info("financial_phase_executor: starting phase '%s'", phase_name)
    phase_budget = _PHASE_BUDGETS.get(phase_name, 4)

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, phase_name)
    llm_with_tools = agent.get_bound_llm(llm, tools)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.as_of_date:
        year = state.as_of_date[:4]
        date_context = (
            f"\n\nTODAY'S DATE: {today}. HISTORICAL REFERENCE DATE: {state.as_of_date}\n"
            f"Always include the historical year ({year}) in search queries."
        )
    else:
        year = datetime.now(timezone.utc).year
        date_context = (
            f"\n\nTODAY'S DATE: {today}\n"
            f"Always include the current year ({year}) in search queries for up-to-date results."
        )

    # Build system content: agent system prompt + phase header + date context + plan + prior results
    sys_content = ""
    if state.messages and isinstance(state.messages[0], SystemMessage):
        sys_content = state.messages[0].content

    phase_header = (
        f"\n\n=== CURRENT PHASE: {phase_name.upper().replace('_', ' ')} ===\n"
        f"Focus on the {phase_name.replace('_', ' ')} section of your EXECUTION PLAN. "
        f"Call the tools needed for this phase. Stop when this phase section is complete."
    )

    plan = getattr(state, "scratchpad", None)
    running_ctx = getattr(state, "running_context", None)

    def _extract_phase_plan(full_plan: str, phase: str) -> str:
        """Return the single plan line for this phase; fall back to full plan."""
        phase_key = phase.replace("_", " ")
        for line in full_plan.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith(phase + ":") or stripped.startswith(phase_key + ":"):
                return line.strip()
        return full_plan

    phase_plan = _extract_phase_plan(plan, phase_name) if plan else None

    extra_sections: list[str] = []
    if phase_plan:
        extra_sections.append(
            "PHASE PLAN (follow this):\n" + phase_plan
        )
    if running_ctx:
        extra_sections.append(
            "PRIOR PHASE RESULTS (already done — do not re-fetch):\n" + running_ctx
        )

    def _build_system(extra_ctx_parts: list[str]) -> str:
        content = sys_content + phase_header + date_context
        if extra_ctx_parts:
            content += "\n\n" + "\n\n".join(extra_ctx_parts)
        return content

    # Only include the current HumanMessage — no prior AI phase outputs (avoids opinion contamination)
    user_msg = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_msg = msg
            break

    phase_messages: list = [
        SystemMessage(content=_build_system(extra_sections)),
        *([] if user_msg is None else [user_msg]),
    ]

    phase_running_ctx_parts: list[str] = []
    accumulated_running_ctx = running_ctx or ""
    phase_tool_calls_log: list[dict] = []

    for iteration in range(phase_budget):
        try:
            _t0 = time.monotonic()
            response = await llm_with_tools.ainvoke(phase_messages)
            _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
            llm_call_duration.labels(agent="sdk", model=_model_name, phase=phase_name).observe(
                time.monotonic() - _t0
            )
        except Exception:
            logger.exception("financial_phase_executor: LLM call failed in phase '%s'", phase_name)
            break

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # LLM produced final prose — phase complete
            if response.content:
                phase_running_ctx_parts.append(response.content)
            break

        logger.info(
            "financial_phase_executor [%s] iter %d: %d tool call(s): %s",
            phase_name, iteration, len(tool_calls), [tc["name"] for tc in tool_calls],
        )
        phase_messages.append(response)

        # Record tool calls for the execution trace
        for tc in tool_calls:
            phase_tool_calls_log.append({
                "action": "tool_call",
                "phase": phase_name,
                "tool": tc["name"],
                "args": tc.get("args", {}),
            })

        # Execute tools
        tool_results = await _execute_tool_calls(agent, tool_calls, state.tool_timeout, phase_tools=tools)

        for tc, tr in zip(tool_calls, tool_results):
            phase_messages.append(tr)
            phase_running_ctx_parts.append(f"[{tc['name']}] → {tr.content}")
            phase_tool_calls_log.append({
                "action": "tool_result",
                "phase": phase_name,
                "tool": tc["name"],
                "result_length": len(tr.content),
                "result_preview": tr.content[:500] if len(tr.content) > 500 else tr.content,
            })

        # Rebuild system message so next iteration sees updated context
        accumulated_running_ctx = (
            (running_ctx or "")
            + ("\n" if running_ctx else "")
            + "\n".join(phase_running_ctx_parts)
        )
        updated_extra = []
        if phase_plan:
            updated_extra.append(
                "PHASE PLAN (follow this):\n" + phase_plan
            )
        updated_extra.append(
            "PRIOR PHASE RESULTS (already done — do not re-fetch):\n" + accumulated_running_ctx
        )
        phase_messages[0] = SystemMessage(content=_build_system(updated_extra))
    else:
        logger.warning(
            "financial_phase_executor: phase '%s' hit budget (%d iterations)", phase_name, phase_budget
        )

    # Compile phase findings into a labeled block appended to running_context
    phase_findings = "\n".join(phase_running_ctx_parts) if phase_running_ctx_parts else "(no data retrieved)"
    phase_block = (
        f"=== {phase_name.upper().replace('_', ' ')} ===\n"
        f"{phase_findings}\n"
        f"=== END {phase_name.upper().replace('_', ' ')} ==="
    )
    logger.info(
        "financial_phase_executor: phase '%s' complete — %d chars added to running_context",
        phase_name, len(phase_block),
    )

    return {
        "running_context": phase_block,
        "iteration": state.iteration + 1,
        "tool_calls_log": phase_tool_calls_log,
    }


async def synthesis_node(agent, state) -> dict:
    """
    Synthesis phase — produce the final user-facing report.

    Reads from state.running_context (accumulated prose from all prior phases) and
    state.query_type (set by financial_orchestrate) to select the right synthesis prompt.
    No tools — pure LLM reasoning step.
    """
    from agent_sdk.financial.prompts import SYNTHESIS_PROMPT, COMPARATIVE_SYNTHESIS_PROMPT
    from datetime import datetime, timezone

    logger.info("Running synthesis phase")
    llm = _get_phase_llm(agent, state)

    running_ctx = getattr(state, "running_context", None) or "(no prior phase results available)"
    query_type = getattr(state, "query_type", None)

    if query_type == "comparative":
        synthesis_prompt = COMPARATIVE_SYNTHESIS_PROMPT
    else:
        synthesis_prompt = SYNTHESIS_PROMPT

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_context = f"\n\nTODAY'S DATE: {today}"

    # Inject running_context as a labeled block in the system prompt
    full_system = (
        synthesis_prompt
        + date_context
        + "\n\n=== PRIOR ANALYSIS ===\n"
        + running_ctx
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
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=_synthesis_timeout)
    except asyncio.TimeoutError:
        logger.error("Synthesis LLM call timed out after %.0fs", _synthesis_timeout)
        fallback_content = "Analysis timed out during synthesis. Please try a simpler or more specific query."
        return {
            "messages": [AIMessage(content=fallback_content)],
            "iteration": state.iteration + 1,
        }

    synthesis_data = _extract_json(response.content) or {}

    if synthesis_data and "full_report" in synthesis_data:
        report_msg = AIMessage(content=synthesis_data["full_report"])
    else:
        logger.warning(
            "Synthesis did not return structured JSON; using raw content (%d chars). First 200: %s",
            len(response.content), response.content[:200],
        )
        report_msg = AIMessage(content=response.content)

    return {
        "messages": [report_msg],
        "iteration": state.iteration + 1,
    }


def phase_router(state) -> dict:
    """
    Pass-through routing node for the financial reasoning pipeline.
    Actual routing logic is handled by _route_phase conditional edges.
    """
    return {}


def phase_advance(state) -> dict:
    """
    Advance to the next phase in the pipeline.
    Removes the current phase from phases_to_run and sets current_phase.
    """
    remaining = list(state.phases_to_run)
    if remaining:
        remaining.pop(0)  # Remove completed phase

    next_phase = remaining[0] if remaining else "done"
    logger.info("Phase advance: %s → %s (remaining: %s)", state.current_phase, next_phase, remaining)

    return {
        "phases_to_run": remaining,
        "current_phase": next_phase,
    }


def parallel_fan_in(state) -> dict:
    """
    Fan-in node after parallel sector_analysis + company_analysis.

    _causal_analysis_router fans out directly (bypassing phase_advance), so
    phases_to_run still contains causal_analysis, sector_analysis, and company_analysis
    when this node runs. Pop all three at once so phase_router advances correctly.

    Also handles the sequential case (only one of sector/company in the plan):
    phase_advance already popped causal_analysis, so only the completed phase is
    removed from the set — anything absent is a no-op.
    """
    to_remove = {"causal_analysis", "sector_analysis", "company_analysis"}
    remaining = [p for p in state.phases_to_run if p not in to_remove]
    next_phase = remaining[0] if remaining else "done"
    logger.info(
        "parallel_fan_in: sector+company complete → next=%s, remaining=%s",
        next_phase, remaining,
    )
    return {
        "phases_to_run": remaining,
        "current_phase": next_phase,
    }




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
    Returns a filtered subset of agent tools plus phase-specific financial tools.

    Results are cached per (phase, agent_tools_fingerprint) on the agent instance
    to avoid re-importing financial modules and scanning all tools on every phase
    iteration (financial mode runs 20-36 tool lookups per request).
    Cache is invalidated on MCP reconnection by clearing agent._phase_tools_cache.
    """
    # Cache key: phase name + frozenset of current tool names (detects MCP reconnections)
    tool_fingerprint = frozenset(agent.tools_by_name.keys())
    cache_key = (phase, tool_fingerprint)
    cached = agent._phase_tools_cache.get(cache_key)
    if cached is not None:
        return cached

    from agent_sdk.financial.causal_graph import get_causal_graph_tools
    from agent_sdk.financial.ontology import get_ontology_tools
    from agent_sdk.financial.quant_tools import get_quant_tools

    # Financial tools organized by phase (computed once per cache miss)
    quant = get_quant_tools()
    causal = get_causal_graph_tools()
    ontology = get_ontology_tools()

    phase_financial_tools: dict[str, list] = {
        "regime_assessment": [t for t in quant if t.name == "detect_market_regime"],
        "causal_analysis": causal + [t for t in quant if t.name == "run_scenario_simulation"],
        "sector_analysis": ontology,
        "company_analysis": (
            ontology + [t for t in quant if t.name in (
                "run_dcf", "run_comparable_valuation", "calculate_technical_signals", "calculate_risk_metrics"
            )]
        ),
        "risk_assessment": (
            [t for t in quant if t.name in ("run_scenario_simulation", "calculate_risk_metrics")]
            + causal
        ),
    }

    financial_tools = phase_financial_tools.get(phase, [])
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
    """Remap common LLM field-name variations to QueryClassification fields."""
    normalized = dict(raw)

    # Remap query_type aliases
    for alias in ("type", "classification", "category"):
        if alias in normalized and "query_type" not in normalized:
            normalized["query_type"] = normalized.pop(alias)

    # Remap reasoning aliases
    for alias in ("reason", "explanation"):
        if alias in normalized and "reasoning" not in normalized:
            normalized["reasoning"] = normalized.pop(alias)

    # Convert reasoning_phases / phases list to individual booleans
    phases_list = normalized.pop("reasoning_phases", None) or normalized.pop("phases", None)
    if phases_list and isinstance(phases_list, list):
        phase_mapping = {
            "regime_assessment": "requires_regime_assessment",
            "causal_analysis": "requires_causal_analysis",
            "sector_analysis": "requires_sector_analysis",
            "company_analysis": "requires_company_analysis",
            "risk_assessment": "requires_risk_assessment",
        }
        for field in phase_mapping.values():
            normalized.setdefault(field, False)
        for phase in phases_list:
            if phase in phase_mapping:
                normalized[phase_mapping[phase]] = True

    # Strip unknown keys to avoid Pydantic extra-field errors
    valid_keys = {
        "query_type", "entities",
        "requires_regime_assessment", "requires_causal_analysis",
        "requires_sector_analysis", "requires_company_analysis",
        "requires_risk_assessment", "reasoning",
    }
    return {k: v for k, v in normalized.items() if k in valid_keys}




async def comparative_analysis_node(agent, state) -> dict:
    """
    Comparative analysis phase — run isolated ReAct loops per entity in parallel
    and append the combined results to running_context for synthesis to consume.
    """
    from datetime import datetime, timezone

    logger.info("Running comparative analysis phase for multiple entities")
    entities = getattr(state, "entities", []) or []
    if not entities:
        logger.warning("No entities found for comparative analysis, falling back to synthesis")
        return {
            "current_phase": "synthesis",
            "iteration": state.iteration + 1,
        }

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "company_analysis")
    llm_with_tools = agent.get_bound_llm(llm, tools)

    # Build base system content (agent prompt + prior context)
    sys_content = ""
    if state.messages and isinstance(state.messages[0], SystemMessage):
        sys_content = state.messages[0].content

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_ctx = f"\n\nTODAY'S DATE: {today}. Include the current year in search queries."

    running_ctx = getattr(state, "running_context", None)
    plan = getattr(state, "scratchpad", None)
    extra: list[str] = []
    if plan:
        extra.append("EXECUTION PLAN:\n" + plan)
    if running_ctx:
        extra.append("PRIOR PHASE RESULTS:\n" + running_ctx)

    base_system = sys_content + date_ctx
    if extra:
        base_system += "\n\n" + "\n\n".join(extra)

    async def analyze_entity(entity: str) -> str:
        entity_system = (
            base_system
            + f"\n\n=== FOCUS ENTITY: {entity} ===\n"
            f"Perform a thorough company analysis for {entity}. "
            f"Retrieve fundamentals, valuation, recent performance, and key risks."
        )
        messages: list = [SystemMessage(content=entity_system)]

        # Include the current user query
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                messages.append(msg)
                break

        llm_timeout = state.tool_timeout
        seen_calls: dict[tuple, str] = {}

        for _ in range(5):
            try:
                response = await asyncio.wait_for(
                    llm_with_tools.ainvoke(messages), timeout=llm_timeout
                )
            except asyncio.TimeoutError:
                logger.warning("LLM timed out for entity '%s'", entity)
                return f"Analysis timed out for {entity}"

            if not getattr(response, "tool_calls", None):
                return response.content or f"(no analysis produced for {entity})"

            messages.append(response)

            deduped_calls, cached_msgs = [], []
            for tc in response.tool_calls:
                key = (tc["name"], str(sorted(tc.get("args", {}).items())))
                if key in seen_calls:
                    logger.info("Skipping duplicate tool '%s' for entity '%s'", tc["name"], entity)
                    cached_msgs.append(ToolMessage(content=seen_calls[key], tool_call_id=tc["id"]))
                else:
                    deduped_calls.append(tc)

            if deduped_calls:
                fresh = await _execute_tool_calls(agent, deduped_calls, llm_timeout, phase_tools=tools)
                for tc, tr in zip(deduped_calls, fresh):
                    key = (tc["name"], str(sorted(tc.get("args", {}).items())))
                    seen_calls[key] = tr.content
                messages.extend(fresh)

            messages.extend(cached_msgs)

        return "Analysis incomplete (iteration budget reached)"

    _entity_budget = state.tool_timeout * 1.5

    async def _bounded(e: str) -> str:
        try:
            return await asyncio.wait_for(analyze_entity(e), timeout=_entity_budget)
        except asyncio.TimeoutError:
            logger.warning("Entity '%s' hit wall-clock budget (%.0fs)", e, _entity_budget)
            return f"Analysis budget exceeded for {e} — partial data only."

    results = await asyncio.gather(*(_bounded(e) for e in entities))
    logger.info("Comparative analysis complete — %d entities processed", len(entities))

    combined = "\n".join(f"## {e}\n{r}\n---" for e, r in zip(entities, results))
    phase_block = (
        "=== COMPARATIVE ANALYSIS ===\n"
        + combined
        + "\n=== END COMPARATIVE ANALYSIS ==="
    )

    return {
        "running_context": phase_block,
        "iteration": state.iteration + 1,
    }


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
    Load perspective memory for the current user and inject it as background
    context into state.perspective_context.

    Runs after initialize, before triager. No-op if:
    - agent has no memory_manager
    - state.user_id is None
    - no perspective memory exists yet for this user
    """
    memory_manager = getattr(agent, "memory_manager", None)
    if memory_manager is None:
        return {}

    user_id = getattr(state, "user_id", None)
    if not user_id:
        return {}

    perspective = await memory_manager.get_perspective(user_id)
    if not perspective:
        return {}

    logger.debug("load_user_context: injecting perspective for user=%s (%d chars)", user_id, len(perspective))
    return {
        "perspective_context": _PERSPECTIVE_INJECTION_HEADER.format(perspective=perspective),
    }


async def memory_writer(agent, state: AgentState) -> dict:
    """
    Terminal node that fires the memory pipeline as a background asyncio task.

    Creates a snapshot memory summary of the completed query, then (if the
    episodic threshold is reached) compiles episodic and perspective memories.

    Does NOT block: schedules work via asyncio.create_task and returns {} immediately,
    so no latency is added to the user-facing response.
    """
    memory_manager = getattr(agent, "memory_manager", None)
    if memory_manager is None:
        return {}

    user_id = getattr(state, "user_id", None)
    if not user_id:
        return {}

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
        return {}

    scratchpad = getattr(state, "scratchpad", None)
    logger.debug("memory_writer: scratchpad passed to snapshot (%d chars): %.500s",
                 len(scratchpad) if scratchpad else 0, scratchpad or "(none)")

    # Fire-and-forget: memory pipeline runs in background, never blocks response
    asyncio.create_task(
        memory_manager.process_query(
            user_id=user_id,
            session_id=session_id,
            query=query,
            response=response,
            scratchpad=scratchpad,
            llm=agent.llm,
        )
    )
    logger.debug("memory_writer: background memory task scheduled for user=%s", user_id)
    return {}