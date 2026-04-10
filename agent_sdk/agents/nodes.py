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

from agent_sdk.agents.state import AgentState
from agent_sdk.config import settings
from agent_sdk.mcp.exceptions import MCPSessionError
from agent_sdk.metrics import llm_call_duration, tool_call_duration, raw_fallback_total

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
    # If a SystemMessage already exists, check whether it needs updating
    if state.messages and isinstance(state.messages[0], SystemMessage):
        if state.system_prompt and state.messages[0].content != state.system_prompt:
            logger.info("Updating system prompt (%d chars)", len(state.system_prompt))
            return {"messages": [RemoveMessage(id=state.messages[0].id),
                                 SystemMessage(content=state.system_prompt)]}
        logger.debug("System prompt already present, skipping initialization")
        return {}

    content = state.system_prompt or (
        "You are an autonomous assistant. "
        "You may call tools to achieve the user's goal, "
        "or respond directly when tools are not needed."
    )
    logger.info("Initialized agent with system prompt (%d chars)", len(content))
    return {"messages": [SystemMessage(content=content)]}


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
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

    # Build appended context sections (summary + perspective) without mutating state.messages
    extra_sections: list[str] = []

    if state.summary:
        extra_sections.append(
            "INTERNAL CONTEXT (for your reference only — do NOT include, repeat, "
            "or paraphrase any of this in your response to the user):\n"
            f"Previous conversation summary: {state.summary}"
        )

    # Inject financial phase outputs if present (financial_analyst mode follow-up queries)
    _FINANCIAL_PHASE_FIELDS = [
        ("regime_context", "Market Regime"),
        ("causal_analysis", "Causal Analysis"),
        ("sector_findings", "Sector Analysis"),
        ("company_analysis", "Company Analysis"),
        ("risk_assessment", "Risk Assessment"),
    ]
    phase_parts = []
    for field, label in _FINANCIAL_PHASE_FIELDS:
        val = getattr(state, field, None)
        if val:
            serialized = json.dumps(val, default=str)
            phase_parts.append(f"**{label}**: {serialized[:800]}")
    if phase_parts:
        extra_sections.append(
            "PRIOR ANALYSIS CONTEXT (use to answer follow-up questions accurately — "
            "do not re-fetch data already captured here):\n" + "\n".join(phase_parts)
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
        except Exception:
            breaker.record_failure(name)
            logger.exception("Tool '%s' failed", name)
            raise

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
                    logger.info("Reconnected — retrying %d tool call(s)", len(tool_calls))
                    results = await _gather_with_timeout()
                else:
                    raise
            else:
                raise

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

    # Accumulate tool results into scratchpad (zero latency — pure string ops)
    section = "\n".join(
        f"[{msg.name}] → {msg.content}"
        for msg in results
        if hasattr(msg, "name") and msg.name
    )
    existing = getattr(state, "scratchpad", None) or ""
    new_scratchpad = (existing + "\n\n" + section).strip() if existing else section
    logger.debug("tool_node: scratchpad updated (%d chars total): %.500s",
                 len(new_scratchpad), new_scratchpad)

    return {"messages": list(results), "scratchpad": new_scratchpad}


def post_tool_router(state: AgentState) -> Literal["summarize_conversation", "llm_call"]:
    """
    After tool execution, check if context needs summarization before
    handing back to the LLM. This enables mid-loop summarization without
    ever skipping pending tool calls.
    """
    summary_text = state.summary or ""
    needs_summarization = (
        state.enable_summarization
        and (
            len(state.messages) > state.keep_last_n_messages
            or _estimate_token_count(state.messages, summary_text) > state.max_context_tokens
        )
    )
    if needs_summarization:
        logger.debug("Post-tool routing → summarize_conversation (messages=%d, est_tokens=%d)",
                     len(state.messages), _estimate_token_count(state.messages, summary_text))
        return "summarize_conversation"
    return "llm_call"


def pre_llm_router(state: AgentState) -> Literal["summarize_conversation", "llm_call"]:
    """
    At the start of each conversation turn, check whether the context is already
    large enough to warrant summarization before calling the LLM.

    Uses a looser threshold (80% of max_context_tokens) to act preventively.
    This catches long conversational sessions that never trigger tool calls
    and therefore never hit post_tool_router.
    """
    summary_text = state.summary or ""
    needs_summarization = (
        state.enable_summarization
        and (
            len(state.messages) > state.keep_last_n_messages
            or _estimate_token_count(state.messages, summary_text) > state.max_context_tokens * 0.8
        )
    )
    if needs_summarization:
        logger.debug("Pre-LLM routing → summarize_conversation (messages=%d, est_tokens=%d)",
                     len(state.messages), _estimate_token_count(state.messages, summary_text))
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
# FINANCIAL REASONING PIPELINE — Phase Nodes
# ============================================================================
# These nodes are only used by create_financial_reasoning_graph().
# They implement the multi-step cognitive pipeline for structured financial
# analysis. Each node represents one analytical phase with its own system
# prompt, tool set, and structured output schema.
# ============================================================================


def _build_phase_return(
    response,
    data_key: str,
    data: dict,
    state,
    phase_name: str,
    extra: dict | None = None,
) -> dict:
    """
    Shared helper for all financial pipeline phase nodes.

    Builds the standard state-update dict for a completed phase, handling the
    JSON-fallback pattern that was previously copy-pasted across 5+ nodes.
    If *data* is falsy the phase fell back to raw text — increment the fallback
    counter and log a warning so operators can detect degraded analysis.
    """
    fallback_inc = 0
    if not data:
        logger.warning("%s fell back to raw_analysis — LLM did not return parseable JSON", phase_name)
        fallback_inc = 1
        raw_fallback_total.labels(phase=phase_name).inc()
    result: dict = {
        "messages": [response],
        data_key: data if data else {"raw_analysis": response.content},
        "raw_fallback_count": state.raw_fallback_count + fallback_inc,
        "iteration": state.iteration + 1,
        "phase_iterations": {phase_name: state.phase_iterations.get(phase_name, 0) + 1},
    }
    if extra:
        result.update(extra)
    return result

async def financial_initialize(state) -> dict:
    """Initialize the financial reasoning pipeline."""
    # Reuse standard initialize for system message setup
    result = await initialize(state)
    
    # Initialize phase_iterations with 0 for all phases
    result["phase_iterations"] = {
        "query_classification": 0,
        "regime_assessment": 0,
        "causal_analysis": 0,
        "sector_analysis": 0,
        "company_analysis": 0,
        "comparative_analysis": 0,
        "risk_assessment": 0,
        "synthesis": 0,
    }
    return result


async def classify_query_node(agent, state) -> dict:
    """
    Classify the user's query to determine which pipeline phases to activate.
    Uses the LLM with a classification-specific prompt.
    """
    from agent_sdk.financial.prompts import QUERY_CLASSIFIER_PROMPT
    from agent_sdk.financial.schemas import QueryClassification, QueryType

    logger.info("Classifying query for financial reasoning pipeline")

    llm = _get_phase_llm(agent, state)

    # Build classification prompt
    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # Strip [CONTEXT] wrapper so the classifier sees the raw user text
    clean_query = _strip_context_block(user_query)

    # Include recent conversational context (last 4 turns before current query)
    # so the classifier can resolve follow-ups like "Yes" or "Go ahead".
    recent_context: list[str] = []
    for msg in state.messages[:-1]:  # All messages except the current HumanMessage
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and msg.content:
            recent_context.append(f"Assistant: {msg.content}")
        elif isinstance(msg, HumanMessage):
            content = _strip_context_block(msg.content)
            if content:
                recent_context.append(f"User: {content}")
    recent_context = recent_context[-4:]  # Keep last 4 conversational turns

    if recent_context:
        context_str = "\n".join(recent_context)
        classify_content = (
            f"Recent conversation:\n{context_str}\n\n"
            f"Classify this query (in context of the conversation above):\n\n{clean_query}"
        )
    else:
        classify_content = f"Classify this query:\n\n{clean_query}"

    classification_prompt = [
        SystemMessage(content=QUERY_CLASSIFIER_PROMPT),
        HumanMessage(content=classify_content),
    ]

    try:
        response = await llm.ainvoke(classification_prompt)
        # Try to parse structured classification from response
        import json
        content = response.content
        # Extract JSON from response
        classification = _extract_json(content)
        if classification:
            qc = QueryClassification(**_normalize_classification(classification))
        else:
            # Default to data_retrieval (minimal pipeline) if parsing fails
            logger.warning("Could not parse query classification, defaulting to data_retrieval")
            qc = QueryClassification(
                query_type=QueryType.DATA_RETRIEVAL,
                requires_regime_assessment=False,
                requires_causal_analysis=False,
                requires_sector_analysis=False,
                requires_company_analysis=True,
                requires_risk_assessment=False,
                reasoning="Classification parsing failed — running minimal pipeline",
            )
    except Exception:
        logger.exception("Query classification failed, defaulting to data_retrieval")
        qc = QueryClassification(
            query_type=QueryType.DATA_RETRIEVAL,
            requires_regime_assessment=False,
            requires_causal_analysis=False,
            requires_sector_analysis=False,
            requires_company_analysis=True,
            requires_risk_assessment=False,
            reasoning="Classification exception — running minimal pipeline",
        )

    # Determine phases to run based on classification
    phases = []
    if qc.query_type == QueryType.DATA_RETRIEVAL:
        # Simple data retrieval — still needs tool access for fetching
        # and synthesis to produce a user-facing response
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
        # Always add synthesis if we have any analysis phases
        if phases:
            phases.append("synthesis")

    logger.info("Query classified as %s — phases: %s", qc.query_type.value, phases)

    return {
        "query_classification": qc.model_dump(),
        "phases_to_run": phases,
        "current_phase": phases[0] if phases else "done",
        "iteration": state.iteration + 1,
        "phase_iterations": {"query_classification": 1},
        "as_of_date": state.as_of_date,  # Persist as_of_date
    }


async def regime_assessment_node(agent, state) -> dict:
    """
    Regime assessment phase — assess macro/market/monetary regime.
    Uses regime-specific tools (detect_market_regime) and system prompt.
    """
    from agent_sdk.financial.prompts import REGIME_ASSESSMENT_PROMPT

    logger.info("Running regime assessment phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "regime_assessment")
    logger.info("regime_assessment — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True) if tools else llm

    prompt = _build_phase_prompt(state, REGIME_ASSESSMENT_PROMPT)
    response = await llm_with_tools.ainvoke(prompt)

    # Check if LLM wants to call tools
    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "messages": [response], 
            "iteration": state.iteration + 1,
            "phase_iterations": {"regime_assessment": state.phase_iterations.get("regime_assessment", 0) + 1}
        }

    # Phase complete — extract structured regime context from response
    regime_data = _extract_json(response.content) or {}
    logger.info("Regime assessment complete: %s", regime_data.get("market_regime", "unknown"))
    return _build_phase_return(response, "regime_context", regime_data, state, "regime_assessment")


async def causal_analysis_node(agent, state) -> dict:
    """
    Causal analysis phase — trace causal chains from trigger events.
    Uses causal graph tools and references regime context from prior phase.
    """
    from agent_sdk.financial.prompts import CAUSAL_ANALYSIS_PROMPT

    logger.info("Running causal analysis phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "causal_analysis")
    logger.info("causal_analysis — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True) if tools else llm

    # Inject prior phase context into prompt
    prompt_template = CAUSAL_ANALYSIS_PROMPT.format(
        regime_context=_format_context(state.findings.get("regime_assessment")),
    )
    prompt = _build_phase_prompt(state, prompt_template)
    response = await llm_with_tools.ainvoke(prompt)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "messages": [response], 
            "iteration": state.iteration + 1,
            "phase_iterations": {"causal_analysis": state.phase_iterations.get("causal_analysis", 0) + 1}
        }

    causal_data = _extract_json(response.content) or {}
    return _build_phase_return(response, "causal_analysis", causal_data, state, "causal_analysis")


async def sector_analysis_node(agent, state) -> dict:
    """Sector analysis phase — analyze relevant sectors."""
    from agent_sdk.financial.prompts import SECTOR_ANALYSIS_PROMPT

    logger.info("Running sector analysis phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "sector_analysis")
    logger.info("sector_analysis — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True) if tools else llm

    prompt_template = SECTOR_ANALYSIS_PROMPT.format(
        regime_context=_format_context(state.findings.get("regime_assessment")),
        causal_analysis=_format_context(state.findings.get("causal_analysis")),
    )
    prompt = _build_phase_prompt(state, prompt_template)
    response = await llm_with_tools.ainvoke(prompt)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "messages": [response], 
            "iteration": state.iteration + 1,
            "phase_iterations": {"sector_analysis": state.phase_iterations.get("sector_analysis", 0) + 1}
        }

    sector_data = _extract_json(response.content) or {}
    return _build_phase_return(response, "sector_findings", sector_data, state, "sector_analysis")


async def company_analysis_node(agent, state) -> dict:
    """Company analysis phase — deep fundamental analysis."""
    from agent_sdk.financial.prompts import COMPANY_ANALYSIS_PROMPT

    logger.info("Running company analysis phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "company_analysis")
    logger.info("company_analysis — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True) if tools else llm

    prompt_template = COMPANY_ANALYSIS_PROMPT.format(
        regime_context=_format_context(state.findings.get("regime_assessment")),
        causal_analysis=_format_context(state.findings.get("causal_analysis")),
        sector_analysis=_format_context(state.findings.get("sector_analysis")),
    )
    prompt = _build_phase_prompt(state, prompt_template)
    response = await llm_with_tools.ainvoke(prompt)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "messages": [response],
            "iteration": state.iteration + 1,
            "phase_iterations": {"company_analysis": state.phase_iterations.get("company_analysis", 0) + 1}
        }

    company_data = _extract_json(response.content) or {}

    # Run symbolic validation on company analysis so warnings reach the risk phase
    company_warnings = _run_phase_validation(state)

    return _build_phase_return(
        response, "company_analysis", company_data, state, "company_analysis",
        extra={"validation_warnings": state.validation_warnings + company_warnings},
    )


async def risk_assessment_node(agent, state) -> dict:
    """Risk assessment phase — stress-test the thesis."""
    from agent_sdk.financial.prompts import RISK_ASSESSMENT_PROMPT

    logger.info("Running risk assessment phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "risk_assessment")
    logger.info("risk_assessment — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True) if tools else llm

    prompt_template = RISK_ASSESSMENT_PROMPT.format(
        regime_context=_format_context(state.findings.get("regime_assessment")),
        causal_analysis=_format_context(state.findings.get("causal_analysis")),
        sector_analysis=_format_context(state.findings.get("sector_analysis")),
        company_analysis=_format_context(state.findings.get("company_analysis")),
    )

    # Inject validation warnings from company_analysis into this phase's prompt
    if state.validation_warnings:
        prompt_template += "\n\nVALIDATION WARNINGS FROM PRIOR PHASES (address these in your risk assessment):\n"
        for w in state.validation_warnings:
            prompt_template += f"- {w}\n"

    prompt = _build_phase_prompt(state, prompt_template)
    response = await llm_with_tools.ainvoke(prompt)

    tool_calls = getattr(response, "tool_calls", None) or []
    if tool_calls:
        return {
            "messages": [response], 
            "iteration": state.iteration + 1,
            "phase_iterations": {"risk_assessment": state.phase_iterations.get("risk_assessment", 0) + 1}
        }

    # Run symbolic validation on the accumulated analysis (adds new warnings from this phase)
    validation_warnings = _run_phase_validation(state)

    risk_data = _extract_json(response.content) or {}
    return _build_phase_return(
        response, "risk_assessment", risk_data, state, "risk_assessment",
        extra={"validation_warnings": state.validation_warnings + validation_warnings},
    )


async def synthesis_node(agent, state) -> dict:
    """Synthesis phase — combine all phases into final report."""
    from agent_sdk.financial.prompts import SYNTHESIS_PROMPT, COMPARATIVE_SYNTHESIS_PROMPT

    logger.info("Running synthesis phase")

    llm = _get_phase_llm(agent, state)
    
    query_type = state.query_classification.get("query_type") if state.query_classification else None

    if query_type == "comparative":
        prompt_template = COMPARATIVE_SYNTHESIS_PROMPT.format(
            company_analysis=_format_context(state.findings.get("company_analysis"))
        )
    else:
        prompt_template = SYNTHESIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
            sector_analysis=_format_context(state.findings.get("sector_analysis")),
            company_analysis=_format_context(state.findings.get("company_analysis")),
            risk_assessment=_format_context(state.findings.get("risk_assessment")),
        )

    # Add validation warnings to synthesis
    if state.validation_warnings:
        prompt_template += "\n\nVALIDATION WARNINGS (must address in your synthesis):\n"
        for w in state.validation_warnings:
            prompt_template += f"- {w}\n"

    if state.raw_fallback_count > 0:
        prompt_template += (
            f"\n\nDATA QUALITY NOTE: {state.raw_fallback_count} phase(s) returned unstructured "
            f"analysis instead of structured data. Some findings above may be less precise — "
            f"acknowledge any gaps in your synthesis."
        )

    prompt = _build_phase_prompt(state, prompt_template)

    # Synthesis doesn't use tools — it's a pure reasoning step
    _synthesis_timeout = state.tool_timeout  # 120s default
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=_synthesis_timeout)
    except asyncio.TimeoutError:
        logger.error("Synthesis LLM call timed out after %.0fs", _synthesis_timeout)
        fallback_content = "Analysis timed out during synthesis. Please try a simpler or more specific query."
        return {
            "messages": [AIMessage(content=fallback_content)],
            "synthesis_report": {"full_report": fallback_content},
            "overall_confidence": 1.0,
            "iteration": state.iteration + 1,
            "phase_iterations": {"synthesis": state.phase_iterations.get("synthesis", 0) + 1},
        }

    synthesis_data = _extract_json(response.content) or {}
    
    base_confidence = 10.0
    penalty = (
        len(state.validation_warnings) * settings.confidence_penalty_per_warning
        + state.raw_fallback_count * settings.confidence_penalty_per_fallback
    )
    confidence_score = max(1.0, round(base_confidence - penalty, 1))

    factors = []
    if state.validation_warnings:
        factors.append(f"{len(state.validation_warnings)} validation issue(s)")
    if state.raw_fallback_count > 0:
        factors.append(f"{state.raw_fallback_count} data fallback(s)")
    
    factors_str = ", ".join(factors) if factors else "All data validated"
    confidence_text = f"\n\n### Analysis Confidence: {confidence_score}/10 — {factors_str}"

    # Ensure the message added to state contains a user-facing narrative,
    # not raw JSON.  If the LLM returned structured JSON with a full_report
    # field, use that as the message content.
    if synthesis_data and "full_report" in synthesis_data:
        synthesis_data["full_report"] += confidence_text
        report_msg = AIMessage(content=synthesis_data["full_report"])
    else:
        logger.warning(
            "Synthesis did not return structured JSON; falling back to raw content (%d chars). "
            "First 200 chars: %s",
            len(response.content),
            response.content[:200],
        )
        content_with_confidence = response.content + confidence_text
        report_msg = AIMessage(content=content_with_confidence)
        if not synthesis_data:
            synthesis_data = {"full_report": content_with_confidence}

    return {
        "messages": [report_msg],
        "synthesis_report": synthesis_data,
        "overall_confidence": confidence_score,
        "iteration": state.iteration + 1,
        "phase_iterations": {"synthesis": state.phase_iterations.get("synthesis", 0) + 1}
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


async def financial_tool_node(agent, state) -> dict:
    """
    Execute tool calls within the financial reasoning pipeline.
    Temporarily registers phase-specific financial tools so tool_node can find them.
    Always emits a ToolMessage for every pending tool_call_id — even on failure —
    so state.messages never contains an orphaned AIMessage(tool_calls).
    """
    phase_tools = _get_phase_tools(agent, state.current_phase)
    original_tools_by_name = dict(agent.tools_by_name)

    for t in phase_tools:
        if t.name not in agent.tools_by_name:
            agent.tools_by_name[t.name] = t

    try:
        return await tool_node(agent, state)
    except Exception as e:
        # tool_node raised — build error ToolMessages for every pending tool_call_id
        # so the message history stays valid for the next LLM call.
        last_message = state.messages[-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        error_messages = [
            ToolMessage(
                content=f"Tool execution failed: {e}",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
        if error_messages:
            logger.warning(
                "financial_tool_node: tool_node raised %s — emitting %d error ToolMessage(s) to keep history valid",
                e, len(error_messages),
            )
            return {"messages": error_messages}
        raise  # no tool_calls to recover from — propagate
    finally:
        agent.tools_by_name = original_tools_by_name


def financial_should_continue(phase_name: str, state) -> str:
    """
    Routing function for financial phase nodes.
    If the LLM requested tools, route to financial_tool_node.
    Otherwise, phase is complete — route to phase_advance.
    
    Supports per-phase iteration budgets via phase_iteration_budgets dict.
    Falls back to global max_iterations if not configured.
    """
    last_message = state.messages[-1]
    has_tool_calls = bool(getattr(last_message, "tool_calls", None))
    
    # Check per-phase budget first
    phase_budgets = state.phase_iteration_budgets
    phase_count = state.phase_iterations.get(phase_name, 0)
    
    if phase_budgets:
        phase_limit = phase_budgets.get(phase_name, 3)
        if phase_count >= phase_limit:
            logger.warning("Iteration limit reached in phase %s (per-phase budget: %d, used: %d)", phase_name, phase_limit, phase_count)
            if has_tool_calls:
                logger.warning(
                    "Phase %s hit iteration limit with pending tool_calls — routing to tool_node to clear them",
                    phase_name,
                )
                return "financial_tool_node"
            return "phase_advance"
    
    # Global safety check (fallback)
    if state.iteration >= state.max_iterations:
        logger.warning("Global iteration limit reached in phase %s (global budget: %d)", phase_name, state.max_iterations)
        if has_tool_calls:
            logger.warning(
                "Phase %s hit global iteration limit with pending tool_calls — routing to tool_node to clear them",
                phase_name,
            )
            return "financial_tool_node"
        return "phase_advance"

    if has_tool_calls:
        return "financial_tool_node"

    return "phase_advance"


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
    """
    from agent_sdk.financial.causal_graph import get_causal_graph_tools
    from agent_sdk.financial.ontology import get_ontology_tools
    from agent_sdk.financial.quant_tools import get_quant_tools

    # Financial tools organized by phase
    phase_financial_tools = {
        "regime_assessment": [
            t for t in get_quant_tools() if t.name == "detect_market_regime"
        ],
        "causal_analysis": get_causal_graph_tools() + [
            t for t in get_quant_tools() if t.name == "run_scenario_simulation"
        ],
        "sector_analysis": get_ontology_tools(),
        "company_analysis": (
            get_ontology_tools()
            + [t for t in get_quant_tools() if t.name in (
                "run_dcf", "run_comparable_valuation", "calculate_technical_signals", "calculate_risk_metrics"
            )]
        ),
        "risk_assessment": [
            t for t in get_quant_tools() if t.name in ("run_scenario_simulation", "calculate_risk_metrics")
        ] + get_causal_graph_tools(),
    }

    financial_tools = phase_financial_tools.get(phase, [])

    # Also include agent's own tools (e.g., MCP data-fetching tools)
    # so phases can retrieve live data — exclude research-only tools
    agent_tools = [t for t in agent.tools_by_name.values() if t.name not in _RESEARCH_ONLY_TOOLS]

    # Combine, deduplicating by name
    seen = set()
    combined = []
    for t in financial_tools + agent_tools:
        if t.name not in seen:
            combined.append(t)
            seen.add(t.name)

    return agent.get_available_tools(phase_tools=combined)


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
        return max(candidates, key=lambda x: len(x))

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


def _run_phase_validation(state) -> list[str]:
    """
    Run symbolic validators on accumulated analysis state.
    Returns list of warning messages.
    """
    from agent_sdk.financial.validators import validate_logical_consistency, validate_confidence

    warnings = []

    # Validate company analysis if available
    ca = state.findings.get("company_analysis")
    if ca and isinstance(ca, dict):
        # Extract metrics for validation
        val = ca.get("valuation", {})
        fund = ca.get("fundamentals", {})

        results = validate_logical_consistency(
            valuation_assessment=ca.get("valuation_assessment"),
            recommendation=ca.get("recommendation"),
            pe=val.get("pe_trailing"),
            roe=fund.get("roe"),
            debt_to_equity=fund.get("debt_to_equity"),
            interest_coverage=fund.get("interest_coverage"),
        )

        for r in results:
            if not r.passed:
                warnings.append(r.message)

    # Validate confidence calibration
    regime = state.findings.get("regime_assessment")
    if regime and isinstance(regime, dict):
        conf = regime.get("confidence", 0.5)
        data_points = len([v for v in regime.values() if v is not None and v != ""])
        result = validate_confidence(
            stated_confidence=conf,
            data_points_available=data_points,
        )
        if not result.passed:
            warnings.append(result.message)

    return warnings



_LARGE_RESULT_THRESHOLD = settings.large_result_threshold  # chars; results larger than this are distilled immediately


async def _summarize_large_tool_result(agent, tool_result, tool_name: str, entity: str, timeout: float):
    """
    Distill a single large tool result to key financial facts right after it arrives,
    before it is appended to the message history. This keeps per-iteration context
    bounded so the global compressor never receives an input that exceeds model limits.
    All numerical values are explicitly preserved in the summariser prompt.
    """
    import asyncio
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

    content = tool_result.content if hasattr(tool_result, "content") else ""
    content = content or ""
    if len(content) < _LARGE_RESULT_THRESHOLD:
        return tool_result

    summarizer = agent.summarizer or agent.llm
    prompt = [
        SystemMessage(content=(
            f"The following is raw financial data for {entity} retrieved via '{tool_name}'. "
            "Extract and preserve ALL key metrics verbatim: revenue, profit, margins (OPM/NPM/EBITDA%), "
            "P/E, P/B, EV/EBITDA, ROE, ROCE, debt/equity, FCF, promoter holding%, pledge%, "
            "dividend yield, capex, working capital, and any guidance or commentary. "
            "Format as a structured bullet-point summary. Do NOT drop any numerical values."
        )),
        HumanMessage(content=content),
    ]
    try:
        resp = await asyncio.wait_for(summarizer.ainvoke(prompt), timeout=timeout)
        summary = resp.content if hasattr(resp, "content") else str(resp)
        logger.info(
            "Pre-compressed large tool result '%s' for '%s': %d → %d chars",
            tool_name, entity, len(content), len(summary),
        )
        return ToolMessage(content=summary, tool_call_id=tool_result.tool_call_id)
    except Exception:
        logger.warning(
            "Pre-compression failed for tool '%s' / entity '%s' — using original result",
            tool_name, entity,
        )
        return tool_result


async def _compress_entity_messages(agent, messages: list, entity: str, timeout: float) -> list:
    """
    Summarize intermediate tool results in the analyze_entity message list to prevent
    context window overflow. Always preserves the leading SystemMessage and the most
    recent AI+tool round so the LLM retains fresh context.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    import asyncio

    has_system = messages and isinstance(messages[0], SystemMessage)
    system_msg = messages[0] if has_system else None
    body = messages[1:] if has_system else list(messages)

    # Need at least 2 rounds (AI msg + tools) before compression makes sense
    if len(body) < 4:
        return messages

    # Keep the 4 most recent messages (≈1 full AI+tool round) verbatim
    keep_tail = min(4, len(body))
    to_summarize = body[:-keep_tail]
    tail = body[-keep_tail:]

    if not to_summarize:
        return messages

    summarizer = agent.summarizer or agent.llm
    summarizer_input = [
        SystemMessage(content=(
            f"The following are financial data retrieval results for {entity}. "
            "Extract and preserve ALL key financial metrics, ratios, figures, dates, and facts verbatim. "
            "Be comprehensive — do not drop any numerical data. Format as a structured bullet-point summary."
        )),
        *to_summarize,
        SystemMessage(content="Provide the structured financial data summary now."),
    ]

    try:
        if hasattr(summarizer, "ainvoke"):
            resp = await asyncio.wait_for(summarizer.ainvoke(summarizer_input), timeout=timeout)
        else:
            resp = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, summarizer.invoke, summarizer_input),
                timeout=timeout,
            )
        summary_text = resp.content if hasattr(resp, "content") else str(resp)
        logger.info(
            "Compressed %d messages for entity '%s' → summary (%d chars)",
            len(to_summarize), entity, len(summary_text),
        )
        compressed = []
        if system_msg:
            compressed.append(system_msg)
        compressed.append(HumanMessage(content=f"[Previously retrieved data for {entity}]\n{summary_text}"))
        compressed.extend(tail)
        return compressed
    except Exception:
        logger.warning(
            "Context compression failed for entity '%s' — hard-truncating to last 3 rounds",
            entity,
        )
        # Keep SystemMessage + last 3 AI+tool rounds (≈12 messages) to stay within limits
        keep = body[-12:]
        if has_system:
            return [system_msg] + keep
        return keep


async def comparative_analysis_node(agent, state) -> dict:
    from agent_sdk.financial.prompts import COMPANY_ANALYSIS_PROMPT
    from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
    import asyncio

    logger.info("Running comparative analysis phase for multiple entities")
    entities = state.query_classification.get("entities", [])
    if not entities:
        logger.warning("No entities found for comparative analysis, falling back to synthesis")
        return {
            "current_phase": "synthesis", 
            "iteration": state.iteration + 1,
            "phase_iterations": {"comparative_analysis": state.phase_iterations.get("comparative_analysis", 0) + 1}
        }
        
    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "company_analysis")
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True) if tools else llm

    async def analyze_entity(entity: str):
        prompt_template = COMPANY_ANALYSIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
            sector_analysis=_format_context(state.findings.get("sector_analysis")),
        )
        prompt_template += f"\n\nFOCUS ENTITY: {entity}. Follow all standard company analysis instructions for this specific entity."
        
        messages = [SystemMessage(content=prompt_template)]
        
        # Isolated tool loop for this entity
        llm_timeout = state.tool_timeout
        seen_tool_calls: dict[tuple, str] = {}  # (name, args_key) → cached result
        compression_failures = 0

        for _ in range(5):
            try:
                response = await asyncio.wait_for(
                    llm_with_tools.ainvoke(messages),
                    timeout=llm_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning("LLM call timed out after %.0fs in analyze_entity for '%s'", llm_timeout, entity)
                return f"Analysis timed out for {entity}"

            if not getattr(response, "tool_calls", None):
                return response.content

            messages.append(response)

            # Deduplicate tool calls — return cached result for identical (name, args) pairs
            tool_timeout = state.tool_timeout
            deduped_calls, cached_msgs = [], []
            for tc in response.tool_calls:
                key = (tc["name"], str(sorted(tc.get("args", {}).items())))
                if key in seen_tool_calls:
                    logger.info("Skipping duplicate tool call '%s' — returning cached result", tc["name"])
                    cached_msgs.append(ToolMessage(content=seen_tool_calls[key], tool_call_id=tc["id"]))
                else:
                    deduped_calls.append(tc)

            fresh_results: list = []
            if deduped_calls:
                raw_results = await _execute_tool_calls(agent, deduped_calls, tool_timeout, phase_tools=tools)
                # Distill large results immediately to prevent context bloat
                fresh_results = list(await asyncio.gather(*(
                    _summarize_large_tool_result(agent, tr, tc["name"], entity, llm_timeout)
                    for tc, tr in zip(deduped_calls, raw_results)
                )))
                for tc, tr in zip(deduped_calls, fresh_results):
                    key = (tc["name"], str(sorted(tc.get("args", {}).items())))
                    seen_tool_calls[key] = tr.content

            messages.extend(fresh_results + cached_msgs)

            # Compress if context is still approaching the model's limit (safety net)
            phase_token_limit = int(state.max_context_tokens * 0.70)
            if _estimate_token_count(messages) > phase_token_limit:
                before_len = len(messages)
                logger.info(
                    "Entity '%s' context approaching limit (%d est. tokens) — compressing",
                    entity, _estimate_token_count(messages),
                )
                messages = await _compress_entity_messages(agent, messages, entity, llm_timeout)
                if len(messages) >= before_len:
                    compression_failures += 1
                    if compression_failures >= 2:
                        logger.warning(
                            "Entity '%s' — breaking loop after %d consecutive compression failures",
                            entity, compression_failures,
                        )
                        for msg in reversed(messages):
                            if (hasattr(msg, "content") and msg.content
                                    and not getattr(msg, "tool_calls", None)):
                                return msg.content
                        return f"Analysis partial — context limit reached for {entity}"
                else:
                    compression_failures = 0

        return "Analysis incomplete (exceeded iterations)"

    # Execute entity analyses in parallel with a wall-clock budget per entity
    _entity_budget = state.tool_timeout * 1.5  # e.g. 120s × 1.5 = 180s per entity

    async def _bounded_analyze(e: str) -> str:
        try:
            return await asyncio.wait_for(analyze_entity(e), timeout=_entity_budget)
        except asyncio.TimeoutError:
            logger.warning(
                "Entity '%s' hit wall-clock budget (%.0fs) — returning partial analysis", e, _entity_budget
            )
            return f"Analysis budget exceeded for {e} — partial data only."

    results = await asyncio.gather(*(_bounded_analyze(e) for e in entities))
    logger.info("Comparative analysis complete — %d entities processed", len(entities))
    
    combined_content = ""
    for entity, res in zip(entities, results):
        combined_content += f"\n\n## {entity}\n{res}\n---\n"
        
    # Set fallback count for transparency
    state_fallback_count = state.raw_fallback_count
    
    return {
        "messages": [AIMessage(content=combined_content)],
        "company_analysis": {"raw_analysis": combined_content},  # We inject into company_analysis so synthesis sees it
        "raw_fallback_count": state_fallback_count,
        "iteration": state.iteration + 1,
        "phase_iterations": {"comparative_analysis": state.phase_iterations.get("comparative_analysis", 0) + 1}
    }


# ============================================================================
# STRUCTURED PLANNING — Standard Mode Nodes
# ============================================================================
# These nodes implement the 4-phase Triager → Parallel Planner →
# Stateless Executor → Synthesizer flow for mode="standard".
# Simple ("opaque") queries skip planning and go directly to llm_call.
# Complex ("analytical") queries are broken into parallel tool batches,
# executed without any LLM involvement, then synthesized in one final call.
# ============================================================================

_TRIAGER_SYSTEM_PROMPT = """\
Classify the user query as either "opaque" or "analytical".

opaque: Can be answered directly or with at most one tool call.
  Examples: math, greetings, factual single-entity lookups, simple follow-ups.

analytical: Requires multiple distinct tool calls, comparing sources, or multi-step research.
  Examples: "Compare X and Y", "Research A, B, and C and summarize", "Analyze X in depth".

Output ONLY valid JSON — no explanation, no markdown:
{"type": "opaque"}
OR
{"type": "analytical"}
"""

_PLANNER_SYSTEM_PROMPT = """\
You are a task planner. Break the user query into ordered batches of tool calls.

{tool_catalog}

RULES:
- Group tool calls that are INDEPENDENT into the same batch — they run in parallel.
- Batches are sequential — later batches may depend on earlier results.
- Each call must use a tool name from the catalog above with exact argument names.
- Maximum 4 batches. Maximum 6 calls per batch.
- Do NOT include reasoning or synthesis steps — only tool calls.

Output ONLY valid JSON:
{{"batches": [[{{"tool": "<name>", "args": {{<key>: <value>}}}}, ...], ...]}}
"""

_SYNTHESIZER_SYSTEM_PROMPT = """\
You are a synthesis assistant. Based on the tool findings below, answer the user's question clearly and accurately.
Do not reference tool names, internal steps, or JSON structure — respond naturally and directly.

TOOL FINDINGS:
{scratchpad}
"""


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


async def triager(agent, state: AgentState) -> dict:
    """
    Classify the incoming query as 'opaque' (answer directly / single tool call)
    or 'analytical' (multi-step plan + parallel tool execution).

    A heuristic fast-exit skips the LLM call for trivially short messages to
    avoid unnecessary latency on greetings and simple follow-ups.
    """
    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = _strip_context_block(msg.content)
            break

    if not user_query:
        return {"query_type": "opaque"}

    # Heuristic: short, single-line messages without complex interrogatives → skip LLM
    words = user_query.split()
    if len(words) <= 8 and "\n" not in user_query:
        logger.debug("triager: heuristic fast-exit → opaque (%d words)", len(words))
        return {"query_type": "opaque"}

    llm = agent.llm
    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        llm = get_llm(state.model_id)

    try:
        _t0 = time.monotonic()
        response = await llm.ainvoke([
            SystemMessage(content=_TRIAGER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query: {user_query}"),
        ])
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase="triager").observe(
            time.monotonic() - _t0
        )
        result = _extract_json(response.content)
        query_type = (result or {}).get("type", "opaque")
        if query_type not in ("opaque", "analytical"):
            query_type = "opaque"
    except Exception:
        logger.exception("triager: classification failed — defaulting to opaque")
        query_type = "opaque"

    logger.info("triager: query_type=%s", query_type)
    return {"query_type": query_type}


def triager_router(state: AgentState) -> Literal["llm_call", "parallel_planner"]:
    """Route after triager: opaque → llm_call (existing ReAct loop), analytical → parallel_planner.

    When enable_analytical_path is False the planner is bypassed entirely —
    all queries use the fast opaque path (scratchpad still captured in tool_node).
    """
    if not getattr(state, "enable_analytical_path", True):
        return "llm_call"
    qt = getattr(state, "query_type", None)
    if qt == "analytical":
        return "parallel_planner"
    return "llm_call"


async def parallel_planner(agent, state: AgentState) -> dict:
    """
    For analytical queries: generate an ordered plan of parallel tool-call batches.

    Each batch groups independent calls that can run concurrently via asyncio.gather.
    Later batches may depend on earlier results. The plan is stored in state.execution_plan
    and executed by stateless_executor without any further LLM involvement.

    Falls back to query_type='opaque' (standard llm_call loop) on any parse error.
    """
    tools = list(agent.tools_by_name.values())
    tool_catalog = _format_tool_catalog(tools) if tools else "No tools available."

    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = _strip_context_block(msg.content)
            break

    llm = agent.llm
    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        llm = get_llm(state.model_id)

    planner_prompt = _PLANNER_SYSTEM_PROMPT.format(tool_catalog=tool_catalog)

    try:
        _t0 = time.monotonic()
        response = await llm.ainvoke([
            SystemMessage(content=planner_prompt),
            HumanMessage(content=f"Query: {user_query}"),
        ])
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase="parallel_planner").observe(
            time.monotonic() - _t0
        )
        result = _extract_json(response.content)
        batches = (result or {}).get("batches")
        if not batches or not isinstance(batches, list):
            raise ValueError(f"Invalid batches from planner: {result}")
        # Validate each entry has 'tool' and 'args'
        for batch in batches:
            for call in batch:
                if "tool" not in call:
                    raise ValueError(f"Missing 'tool' key in call: {call}")
                if call["tool"] not in agent.tools_by_name:
                    logger.warning("parallel_planner: unknown tool '%s' in plan — removing", call["tool"])
        # Filter out unknown tools
        batches = [
            [c for c in batch if c.get("tool") in agent.tools_by_name]
            for batch in batches
        ]
        batches = [b for b in batches if b]  # remove empty batches
        if not batches:
            raise ValueError("No valid tool calls in plan after filtering")
        logger.info("parallel_planner: %d batches, %d total calls",
                    len(batches), sum(len(b) for b in batches))
        return {"execution_plan": batches, "current_batch_index": 0}
    except Exception:
        logger.exception("parallel_planner: failed — falling back to opaque (standard llm_call)")
        return {"query_type": "opaque", "execution_plan": None, "current_batch_index": 0}


async def stateless_executor(agent, state: AgentState) -> dict:
    """
    Execute the current batch of tool calls in parallel (asyncio.gather).

    NO LLM involvement — pure concurrent tool dispatch using the agent's
    existing tool registry and circuit-breaker infrastructure.
    Results are appended to state.scratchpad and current_batch_index is advanced.
    """
    plan = getattr(state, "execution_plan", None) or []
    idx = getattr(state, "current_batch_index", 0)

    if idx >= len(plan):
        logger.warning("stateless_executor: called with no remaining batches (idx=%d, plan len=%d)", idx, len(plan))
        return {"current_batch_index": idx}

    batch = plan[idx]
    timeout = state.tool_timeout

    async def run_one(call: dict) -> str:
        tool_name = call.get("tool", "")
        args = call.get("args", {})
        tool = agent.tools_by_name.get(tool_name)
        if tool is None:
            return f"[{tool_name}] ERROR: tool not found"

        breaker = agent._get_breaker(tool_name)
        if breaker.is_open:
            return f"[{tool_name}] SKIPPED: circuit breaker open"

        try:
            _t0 = time.monotonic()
            if hasattr(tool, "ainvoke"):
                result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke if hasattr(tool, "invoke") else tool.run, args),
                    timeout=timeout,
                )
            breaker.record_success()
            tool_call_duration.labels(agent="sdk", tool_name=tool_name).observe(time.monotonic() - _t0)
            logger.info("stateless_executor: '%s' completed (%d chars)", tool_name, len(str(result)))
            return f"[{tool_name}({args})] → {result}"
        except asyncio.TimeoutError:
            breaker.record_failure(tool_name)
            return f"[{tool_name}] TIMEOUT after {timeout:.0f}s"
        except Exception as e:
            breaker.record_failure(tool_name)
            logger.exception("stateless_executor: tool '%s' failed", tool_name)
            return f"[{tool_name}] ERROR: {e}"

    logger.info("stateless_executor: batch %d/%d — %d parallel calls: %s",
                idx + 1, len(plan), len(batch), [c.get("tool") for c in batch])

    results = await asyncio.gather(*[run_one(c) for c in batch])

    batch_section = f"\n\n--- Batch {idx + 1} ---\n" + "\n\n".join(results)
    existing_scratchpad = getattr(state, "scratchpad", None) or ""

    return {
        "scratchpad": existing_scratchpad + batch_section,
        "current_batch_index": idx + 1,
    }


def after_planner_router(state: AgentState) -> Literal["stateless_executor", "llm_call"]:
    """
    After parallel_planner: if the plan is valid route to stateless_executor,
    otherwise fall back to the standard llm_call loop (e.g. if planner failed).
    """
    plan = getattr(state, "execution_plan", None)
    if plan:
        return "stateless_executor"
    logger.info("after_planner_router: no valid plan — falling back to llm_call")
    return "llm_call"


def batch_check(state: AgentState) -> Literal["stateless_executor", "synthesizer"]:
    """After each executor run: proceed to next batch or move to synthesis."""
    plan = getattr(state, "execution_plan", None) or []
    idx = getattr(state, "current_batch_index", 0)
    if idx < len(plan):
        logger.debug("batch_check: more batches remaining (%d/%d)", idx + 1, len(plan))
        return "stateless_executor"
    logger.debug("batch_check: all batches done — routing to synthesizer")
    return "synthesizer"


async def synthesizer(agent, state: AgentState) -> dict:
    """
    Final synthesis: one LLM call (no tools) that turns the scratchpad into
    a user-facing response. Streamed to the client.
    """
    scratchpad = getattr(state, "scratchpad", None) or "(no tool results)"

    # Find the user's original query
    user_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            user_query = _strip_context_block(msg.content)
            break

    # Include agent system prompt so synthesizer maintains agent persona
    system_content = state.system_prompt or "You are a helpful assistant."
    if state.summary:
        system_content += (
            f"\n\nCONVERSATION CONTEXT: {state.summary}"
        )

    synthesis_system = system_content + "\n\n" + _SYNTHESIZER_SYSTEM_PROMPT.format(
        scratchpad=scratchpad
    )

    llm = agent.llm
    if state.model_id:
        from agent_sdk.llm_services.model_registry import get_llm
        llm = get_llm(state.model_id)

    try:
        _t0 = time.monotonic()
        response = await llm.ainvoke([
            SystemMessage(content=synthesis_system),
            HumanMessage(content=user_query),
        ])
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase="synthesizer").observe(
            time.monotonic() - _t0
        )
    except Exception:
        logger.exception("synthesizer: LLM call failed")
        response = AIMessage(content="I encountered an error synthesizing the results. Please try again.")

    logger.info("synthesizer: response (%d chars)", len(getattr(response, "content", "")))
    return {
        "messages": [response],
        "iteration": state.iteration + 1,
    }


# ============================================================================
# STRUCTURED PLANNING — Financial Mode Nodes
# ============================================================================
# Each financial phase node is split into three sub-nodes:
#   1. {phase}_plan  — 1 LLM call: output a list of tool calls needed
#   2. financial_stateless_executor_node  — 0 LLM calls: run them all in parallel
#   3. {phase}_synth — 1 LLM call: synthesize results into structured phase output
#
# comparative_analysis and synthesis phases are unchanged (they have their own
# parallel entity logic and no tools respectively).
# ============================================================================

_FINANCIAL_PHASE_PLANNER_PREFIX = """\
PLANNING STEP — DO NOT ANALYZE YET. DO NOT write explanations, markdown, or ask questions.

Your ONLY task is to output a JSON list of tool calls needed for this phase.
The calls will be executed in parallel and their results returned to you for synthesis.

AUTONOMY RULE: Make all decisions yourself — never ask the user for clarification.
- For Indian companies where NSE/BSE is not specified, default to NSE (ticker.NS suffix).
- If a ticker is ambiguous, pick the most likely one and proceed.

{tool_catalog}

REQUIRED ARGS RULE: Args marked with * are REQUIRED. You MUST provide a concrete, non-empty value
for every required arg. Never output an empty string "", null, or {{}} for a required field.
Example of a CORRECT call:
  {{"tool": "tavily_quick_search", "args": {{"query": "RBI rate cut impact on Indian NBFCs 2026"}}}}
Example of an INVALID call (will be rejected):
  {{"tool": "tavily_quick_search", "args": {{}}}}

OUTPUT RULE (overrides any other format instruction): Output ONLY valid JSON — no explanation, no markdown:
{{"calls": [{{"tool": "<name>", "args": {{<key>: <value>}}}}, ...]}}

If no tool calls are needed, output: {{"calls": []}}
"""

# Maps phase name → state field that stores its structured output
_PHASE_STATE_KEY: dict[str, str] = {
    "regime_assessment": "regime_context",
    "causal_analysis": "causal_analysis",
    "sector_analysis": "sector_findings",
    "company_analysis": "company_analysis",
    "risk_assessment": "risk_assessment",
}


async def financial_phase_planner(phase_name: str, agent, state) -> dict:
    """
    Planning sub-node for a financial pipeline phase.
    Calls the LLM once to produce a list of tool calls needed for this phase.
    Tool calls are stored in state.phase_tool_plan; phase_scratchpad is reset.
    """
    from agent_sdk.financial.prompts import (
        REGIME_ASSESSMENT_PROMPT, CAUSAL_ANALYSIS_PROMPT,
        SECTOR_ANALYSIS_PROMPT, COMPANY_ANALYSIS_PROMPT, RISK_ASSESSMENT_PROMPT,
    )

    # Deterministic phases bypass the planner and go straight to execution
    # These are phases where the tool set is effectively constant or handled by a specific logic
    if phase_name == "query_classification":
        # We simulate a plan for the classifier tool
        return {
            "phase_tool_plan": [{"tool": "classify_query", "args": {}}],
            "phase_scratchpad": "",
            "iteration": state.iteration + 1,
            "phase_iterations": {phase_name: state.phase_iterations.get(phase_name, 0) + 1},
        }

    _phase_prompts = {
        "regime_assessment": REGIME_ASSESSMENT_PROMPT,
        "causal_analysis": CAUSAL_ANALYSIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment"))
        ),
        "sector_analysis": SECTOR_ANALYSIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
        ),
        "company_analysis": COMPANY_ANALYSIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
            sector_analysis=_format_context(state.findings.get("sector_analysis")),
        ),
        "risk_assessment": RISK_ASSESSMENT_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
            sector_analysis=_format_context(state.findings.get("sector_analysis")),
            company_analysis=_format_context(state.findings.get("company_analysis")),
        ),
    }

    phase_prompt = _phase_prompts.get(phase_name, "")
    tools = _get_phase_tools(agent, phase_name)
    tool_catalog = _format_tool_catalog(tools) if tools else "No tools available for this phase."

    planning_prompt = _FINANCIAL_PHASE_PLANNER_PREFIX.format(tool_catalog=tool_catalog)

    llm = _get_phase_llm(agent, state)
    prompt = _build_phase_prompt(state, phase_prompt)
    # Phase prompt first, then planning constraints last so JSON-only rule takes precedence
    prompt[0] = SystemMessage(content=phase_prompt + "\n\n" + planning_prompt)

    logger.info("financial_phase_planner: planning %s phase", phase_name)
    try:
        _t0 = time.monotonic()
        response = await llm.ainvoke(prompt)
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase=f"{phase_name}_plan").observe(
            time.monotonic() - _t0
        )
        result = _extract_json(response.content)
        calls = (result or {}).get("calls", [])
        # Filter to known tools only
        known_names = {t.name for t in tools}
        valid_calls = [c for c in calls if c.get("tool") in known_names]
        if len(valid_calls) != len(calls):
            logger.warning(
                "financial_phase_planner: dropped %d unknown tool(s) from %s plan",
                len(calls) - len(valid_calls), phase_name,
            )
        # Guard: drop calls with missing required args (e.g. tavily_quick_search with empty args)
        tools_by_name_map = {t.name: t for t in tools}

        def _args_complete(call: dict) -> bool:
            tool = tools_by_name_map.get(call.get("tool", ""))
            if not tool:
                return False
            try:
                schema = tool.get_input_schema().model_json_schema() if hasattr(tool, "get_input_schema") else {}
                required = set(schema.get("required", []))
                args = call.get("args") or {}
                return all(k in args and args[k] not in (None, "", {}, []) for k in required)
            except Exception:
                return True  # can't introspect schema — let it through

        complete_calls = [c for c in valid_calls if _args_complete(c)]
        if len(complete_calls) != len(valid_calls):
            dropped = [c.get("tool") for c in valid_calls if not _args_complete(c)]
            logger.warning(
                "financial_phase_planner: dropped %d call(s) with missing required args from %s plan: %s",
                len(valid_calls) - len(complete_calls), phase_name, dropped,
            )
        valid_calls = complete_calls
        logger.info("financial_phase_planner: %s → %d tool call(s)", phase_name, len(valid_calls))
    except Exception:
        logger.exception("financial_phase_planner: %s planning failed — empty plan", phase_name)
        valid_calls = []

    return {
        "phase_tool_plan": valid_calls,
        "phase_scratchpad": "",
        "iteration": state.iteration + 1,
        "phase_iterations": {phase_name: state.phase_iterations.get(phase_name, 0) + 1},
    }


async def financial_stateless_executor_node(phase_name: str, agent, state) -> dict:
    """
    Stateless executor for a financial pipeline phase.
    Runs all calls from state.phase_tool_plan in parallel using asyncio.gather.
    Results are stored in state.phase_scratchpad. No LLM call.
    """
    calls = getattr(state, "phase_tool_plan", None) or []
    if not calls:
        logger.info("financial_stateless_executor: %s — no tool calls planned, skipping", phase_name)
        return {"phase_scratchpad": "(no tool calls)"}

    tools = _get_phase_tools(agent, phase_name)
    tools_by_name = {t.name: t for t in tools}
    # Also include agent tools
    tools_by_name.update(agent.tools_by_name)
    timeout = state.tool_timeout

    async def run_one(call: dict) -> str:
        tool_name = call.get("tool", "")
        args = call.get("args", {})
        tool = tools_by_name.get(tool_name)
        if tool is None:
            return f"[{tool_name}] ERROR: tool not found"

        breaker = agent._get_breaker(tool_name)
        if breaker.is_open:
            return f"[{tool_name}] SKIPPED: circuit breaker open"

        try:
            _t0 = time.monotonic()
            if hasattr(tool, "ainvoke"):
                result = await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke if hasattr(tool, "invoke") else tool.run, args),
                    timeout=timeout,
                )
            breaker.record_success()
            tool_call_duration.labels(agent="sdk", tool_name=tool_name).observe(time.monotonic() - _t0)
            return f"[{tool_name}] → {result}"
        except asyncio.TimeoutError:
            breaker.record_failure(tool_name)
            return f"[{tool_name}] TIMEOUT after {timeout:.0f}s"
        except Exception as e:
            breaker.record_failure(tool_name)
            logger.exception("financial_stateless_executor: tool '%s' failed", tool_name)
            return f"[{tool_name}] ERROR: {e}"

    logger.info("financial_stateless_executor: %s — %d parallel calls: %s",
                phase_name, len(calls), [c.get("tool") for c in calls])

    results = await asyncio.gather(*[run_one(c) for c in calls])
    scratchpad = "\n\n".join(results)
    return {"phase_scratchpad": scratchpad}


async def financial_phase_synthesizer(phase_name: str, agent, state) -> dict:
    """
    Synthesis sub-node for a financial pipeline phase.
    Calls the LLM once with the phase system prompt + tool results from phase_scratchpad
    to produce the structured phase output (regime_context, causal_analysis, etc.).
    """
    from agent_sdk.financial.prompts import (
        REGIME_ASSESSMENT_PROMPT, CAUSAL_ANALYSIS_PROMPT,
        SECTOR_ANALYSIS_PROMPT, COMPANY_ANALYSIS_PROMPT, RISK_ASSESSMENT_PROMPT,
    )

    _phase_prompts = {
        "regime_assessment": REGIME_ASSESSMENT_PROMPT,
        "causal_analysis": CAUSAL_ANALYSIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment"))
        ),
        "sector_analysis": SECTOR_ANALYSIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
        ),
        "company_analysis": COMPANY_ANALYSIS_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
            sector_analysis=_format_context(state.findings.get("sector_analysis")),
        ),
        "risk_assessment": RISK_ASSESSMENT_PROMPT.format(
            regime_context=_format_context(state.findings.get("regime_assessment")),
            causal_analysis=_format_context(state.findings.get("causal_analysis")),
            sector_analysis=_format_context(state.findings.get("sector_analysis")),
            company_analysis=_format_context(state.findings.get("company_analysis")),
        ),
    }

    phase_prompt = _phase_prompts.get(phase_name, "")
    phase_scratchpad = getattr(state, "phase_scratchpad", None) or "(no tool results retrieved)"

    # Inject tool results into the system prompt
    synthesis_prefix = (
        f"TOOL RESULTS FOR THIS PHASE:\n{phase_scratchpad}\n\n"
        "Now synthesize the above results into the required structured output. "
        "Output the structured JSON as specified in your instructions."
    )

    # Add validation warnings for risk_assessment
    if phase_name == "risk_assessment" and state.validation_warnings:
        synthesis_prefix += "\n\nVALIDATION WARNINGS FROM PRIOR PHASES (address these):\n"
        for w in state.validation_warnings:
            synthesis_prefix += f"- {w}\n"

    llm = _get_phase_llm(agent, state)
    prompt = _build_phase_prompt(state, phase_prompt)
    # Append tool results before handing to LLM (inject into system message)
    prompt[0] = SystemMessage(content=f"{phase_prompt}\n\n{synthesis_prefix}")

    logger.info("financial_phase_synthesizer: synthesizing %s phase", phase_name)
    try:
        _t0 = time.monotonic()
        response = await llm.ainvoke(prompt)
        _model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"
        llm_call_duration.labels(agent="sdk", model=_model_name, phase=f"{phase_name}_synth").observe(
            time.monotonic() - _t0
        )
    except Exception:
        logger.exception("financial_phase_synthesizer: %s synthesis LLM call failed", phase_name)
        response = AIMessage(content=f"{{\"error\": \"synthesis failed for {phase_name}\"}}")

    data_key = phase_name
    phase_data = _extract_json(response.content) or {}

    # Store result in the dynamic findings map
    findings = state.findings.copy()
    findings[phase_name] = phase_data

    result = _build_phase_return(response, data_key, phase_data, state, phase_name)
    result["findings"] = findings

    # Run symbolic validation for phases that accumulate warnings
    if phase_name in ("company_analysis", "risk_assessment"):
        new_warnings = _run_phase_validation(state)
        result["validation_warnings"] = state.validation_warnings + new_warnings

    return result


def _financial_after_plan(phase_name: str, state) -> str:
    """
    Conditional edge from a financial phase planner node.
    Routes to executor if there are tool calls, otherwise directly to synthesizer.
    """
    calls = getattr(state, "phase_tool_plan", None) or []
    if calls:
        return f"{phase_name}_exec"
    logger.info("_financial_after_plan: %s has no tool calls — skipping executor", phase_name)
    return f"{phase_name}_synth"


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