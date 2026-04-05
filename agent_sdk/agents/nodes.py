from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Literal, Sequence

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from agent_sdk.agents.state import AgentState

logger = logging.getLogger("agent_sdk.nodes")

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
    pattern = r'<function=(\w+)(.*?)</function>'
    matches = re.findall(pattern, failed_generation, re.DOTALL)

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

    # Merge summary into the existing system message to avoid dual SystemMessages
    if state.summary:
        summary_text = (
            "INTERNAL CONTEXT (for your reference only — do NOT include, repeat, "
            "or paraphrase any of this in your response to the user):\n"
            f"Previous conversation summary: {state.summary}"
        )
        messages = list(state.messages)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(
                content=f"{messages[0].content}\n\n{summary_text}"
            )
        else:
            messages.insert(0, SystemMessage(content=summary_text))
        prompt = messages
    else:
        prompt = list(state.messages)

    try:
        if hasattr(llm_with_tools, "ainvoke"):
            response = await llm_with_tools.ainvoke(prompt)
        else:
            response = llm_with_tools.invoke(prompt)
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
    existing_summary = state.summary or ""
    if existing_summary:
        summary_message = (
            f"This is a summary of the conversation to date: {existing_summary}\n\n"
            "Extend the summary by taking into account the new messages below. "
            "IMPORTANT: Just state the facts. Do NOT mention that you were asked to summarize."
        )
    else:
        summary_message = (
            "Create a concise summary of the facts and context from the conversation below. "
            "IMPORTANT: Just state the facts. Do NOT mention that you were asked to summarize."
        )

    summarizer_input = (
        [SystemMessage(content=summary_message)]
        + messages_to_prune
        + [SystemMessage(content="Provide a concise summary capturing the key facts, decisions, and results. Omit raw data dumps. IMPORTANT: Just state the facts. Do NOT mention that you were asked to summarize.")]
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
            # Check if this is an MCP session termination error
            error_msg = str(e).lower()
            if "session terminated" in error_msg or "session" in error_msg and "closed" in error_msg:
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

    `agent` is bound via functools.partial at graph build time.
    """

    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    timeout = getattr(state, "tool_timeout", 120.0)

    results = await _execute_tool_calls(agent, tool_calls, timeout)

    return {"messages": list(results)}


def post_tool_router(state: AgentState) -> Literal["summarize_conversation", "llm_call"]:
    """
    After tool execution, check if context needs summarization before
    handing back to the LLM. This enables mid-loop summarization without
    ever skipping pending tool calls.
    """
    needs_summarization = (
        state.enable_summarization
        and (
            len(state.messages) > state.keep_last_n_messages
            or _estimate_token_count(state.messages) > state.max_context_tokens
        )
    )
    if needs_summarization:
        logger.debug("Post-tool routing → summarize_conversation (messages=%d, est_tokens=%d)",
                     len(state.messages), _estimate_token_count(state.messages))
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
    needs_summarization = (
        state.enable_summarization
        and (
            len(state.messages) > state.keep_last_n_messages
            or _estimate_token_count(state.messages) > state.max_context_tokens * 0.8
        )
    )
    if needs_summarization:
        logger.debug("Pre-LLM routing → summarize_conversation (messages=%d, est_tokens=%d)",
                     len(state.messages), _estimate_token_count(state.messages))
        return "summarize_conversation"
    logger.debug("Pre-LLM routing → llm_call")
    return "llm_call"


def _estimate_token_count(messages: Sequence) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return sum(len(getattr(m, "content", "") or "") for m in messages) // 4


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
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

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

    return {
        "messages": [response],
        "regime_context": regime_data if regime_data else {"raw_assessment": response.content},
        "iteration": state.iteration + 1,
        "phase_iterations": {"regime_assessment": state.phase_iterations.get("regime_assessment", 0) + 1}
    }


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
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

    # Inject prior phase context into prompt
    prompt_template = CAUSAL_ANALYSIS_PROMPT.format(
        regime_context=_format_context(state.regime_context),
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
    fallback_inc = 0
    if not causal_data:
        logger.warning("causal_analysis fell back to raw_analysis — LLM did not return parseable JSON")
        fallback_inc = 1
    return {
        "messages": [response],
        "causal_analysis": causal_data if causal_data else {"raw_analysis": response.content},
        "raw_fallback_count": state.raw_fallback_count + fallback_inc,
        "iteration": state.iteration + 1,
        "phase_iterations": {"causal_analysis": state.phase_iterations.get("causal_analysis", 0) + 1}
    }


async def sector_analysis_node(agent, state) -> dict:
    """Sector analysis phase — analyze relevant sectors."""
    from agent_sdk.financial.prompts import SECTOR_ANALYSIS_PROMPT

    logger.info("Running sector analysis phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "sector_analysis")
    logger.info("sector_analysis — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

    prompt_template = SECTOR_ANALYSIS_PROMPT.format(
        regime_context=_format_context(state.regime_context),
        causal_analysis=_format_context(state.causal_analysis),
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
    fallback_inc = 0
    if not sector_data:
        logger.warning("sector_analysis fell back to raw_analysis — LLM did not return parseable JSON")
        fallback_inc = 1
    return {
        "messages": [response],
        "sector_findings": sector_data if sector_data else {"raw_analysis": response.content},
        "raw_fallback_count": state.raw_fallback_count + fallback_inc,
        "iteration": state.iteration + 1,
        "phase_iterations": {"sector_analysis": state.phase_iterations.get("sector_analysis", 0) + 1}
    }


async def company_analysis_node(agent, state) -> dict:
    """Company analysis phase — deep fundamental analysis."""
    from agent_sdk.financial.prompts import COMPANY_ANALYSIS_PROMPT

    logger.info("Running company analysis phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "company_analysis")
    logger.info("company_analysis — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

    prompt_template = COMPANY_ANALYSIS_PROMPT.format(
        regime_context=_format_context(state.regime_context),
        causal_analysis=_format_context(state.causal_analysis),
        sector_analysis=_format_context(state.sector_findings),
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
    fallback_inc = 0
    if not company_data:
        logger.warning(
            "company_analysis fell back to raw_analysis — LLM did not return parseable JSON. "
            "Raw output (first 500 chars): %s", response.content[:500]
        )
        fallback_inc = 1

    # Run symbolic validation on company analysis so warnings reach the risk phase
    company_warnings = _run_phase_validation(state)

    return {
        "messages": [response],
        "company_analysis": company_data if company_data else {"raw_analysis": response.content},
        "validation_warnings": state.validation_warnings + company_warnings,
        "raw_fallback_count": state.raw_fallback_count + fallback_inc,
        "iteration": state.iteration + 1,
        "phase_iterations": {"company_analysis": state.phase_iterations.get("company_analysis", 0) + 1}
    }


async def risk_assessment_node(agent, state) -> dict:
    """Risk assessment phase — stress-test the thesis."""
    from agent_sdk.financial.prompts import RISK_ASSESSMENT_PROMPT

    logger.info("Running risk assessment phase")

    llm = _get_phase_llm(agent, state)
    tools = _get_phase_tools(agent, "risk_assessment")
    logger.info("risk_assessment — %d tools available: %s", len(tools), [t.name for t in tools])
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False) if tools else llm

    prompt_template = RISK_ASSESSMENT_PROMPT.format(
        regime_context=_format_context(state.regime_context),
        causal_analysis=_format_context(state.causal_analysis),
        sector_analysis=_format_context(state.sector_findings),
        company_analysis=_format_context(state.company_analysis),
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
    fallback_inc = 0
    if not risk_data:
        logger.warning("risk_assessment fell back to raw_analysis — LLM did not return parseable JSON")
        fallback_inc = 1
    return {
        "messages": [response],
        "risk_assessment": risk_data if risk_data else {"raw_analysis": response.content},
        "validation_warnings": state.validation_warnings + validation_warnings,
        "raw_fallback_count": state.raw_fallback_count + fallback_inc,
        "iteration": state.iteration + 1,
        "phase_iterations": {"risk_assessment": state.phase_iterations.get("risk_assessment", 0) + 1}
    }


async def synthesis_node(agent, state) -> dict:
    """Synthesis phase — combine all phases into final report."""
    from agent_sdk.financial.prompts import SYNTHESIS_PROMPT, COMPARATIVE_SYNTHESIS_PROMPT

    logger.info("Running synthesis phase")

    llm = _get_phase_llm(agent, state)
    
    query_type = state.query_classification.get("query_type") if state.query_classification else None

    if query_type == "comparative":
        prompt_template = COMPARATIVE_SYNTHESIS_PROMPT.format(
            company_analysis=_format_context(state.company_analysis)
        )
    else:
        prompt_template = SYNTHESIS_PROMPT.format(
            regime_context=_format_context(state.regime_context),
            causal_analysis=_format_context(state.causal_analysis),
            sector_analysis=_format_context(state.sector_findings),
            company_analysis=_format_context(state.company_analysis),
            risk_assessment=_format_context(state.risk_assessment),
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
    response = await llm.ainvoke(prompt)

    synthesis_data = _extract_json(response.content) or {}
    
    base_confidence = 10.0
    penalty = (len(state.validation_warnings) * 1.0) + (state.raw_fallback_count * 1.5)
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
    
    # Check per-phase budget first (if configured) — use getattr for Pydantic
    phase_budgets = getattr(state, "phase_iteration_budgets", {})
    phase_count = getattr(state, "phase_iterations", {}).get(phase_name, 0)
    
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
    if getattr(state, "iteration", 0) >= getattr(state, "max_iterations", 10):
        logger.warning("Global iteration limit reached in phase %s (global budget: %d)", phase_name, getattr(state, "max_iterations", 10))
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

    return combined


def _build_phase_prompt(state, phase_system_prompt: str) -> list:
    """Build the message list for a phase LLM call, injecting the phase system prompt."""
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from LLM text output."""
    import json

    # Try the whole text as JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON block in markdown code fences
    import re
    json_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match.strip())
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
    if state.company_analysis and isinstance(state.company_analysis, dict):
        ca = state.company_analysis

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
    if state.regime_context and isinstance(state.regime_context, dict):
        conf = state.regime_context.get("confidence", 0.5)
        data_points = len([v for v in state.regime_context.values() if v is not None and v != ""])
        result = validate_confidence(
            stated_confidence=conf,
            data_points_available=data_points,
        )
        if not result.passed:
            warnings.append(result.message)

    return warnings


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
        logger.warning("Context compression failed for entity '%s' — proceeding with full context", entity)
        return messages


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
            regime_context=_format_context(state.regime_context),
            causal_analysis=_format_context(state.causal_analysis),
            sector_analysis=_format_context(state.sector_findings),
        )
        prompt_template += f"\n\nFOCUS ENTITY: {entity}. Follow all standard company analysis instructions for this specific entity."
        
        messages = [SystemMessage(content=prompt_template)]
        
        # Isolated tool loop for this entity
        llm_timeout = getattr(state, "tool_timeout", 120.0)
        seen_tool_calls: dict[tuple, str] = {}  # (name, args_key) → cached result

        for _ in range(8):
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
            tool_timeout = getattr(state, "tool_timeout", 120.0)
            deduped_calls, cached_msgs = [], []
            for tc in response.tool_calls:
                key = (tc["name"], str(sorted(tc.get("args", {}).items())))
                if key in seen_tool_calls:
                    logger.info("Skipping duplicate tool call '%s' — returning cached result", tc["name"])
                    cached_msgs.append(ToolMessage(content=seen_tool_calls[key], tool_call_id=tc["id"]))
                else:
                    deduped_calls.append(tc)

            fresh_results = []
            if deduped_calls:
                fresh_results = await _execute_tool_calls(agent, deduped_calls, tool_timeout, phase_tools=tools)
                for tc, tr in zip(deduped_calls, fresh_results):
                    key = (tc["name"], str(sorted(tc.get("args", {}).items())))
                    seen_tool_calls[key] = tr.content

            messages.extend(fresh_results + cached_msgs)

            # Compress if context is approaching the model's limit
            phase_token_limit = int(getattr(state, "max_context_tokens", 32768) * 0.70)
            if _estimate_token_count(messages) > phase_token_limit:
                logger.info(
                    "Entity '%s' context approaching limit (%d est. tokens) — compressing",
                    entity, _estimate_token_count(messages),
                )
                messages = await _compress_entity_messages(agent, messages, entity, llm_timeout)

        return "Analysis incomplete (exceeded iterations)"

    # Execute entity analyses in parallel
    results = await asyncio.gather(*(analyze_entity(e) for e in entities))
    
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