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
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    # Merge summary into the existing system message to avoid dual SystemMessages
    if state.summary:
        summary_text = f"Conversation summary:\n{state.summary}"
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

            logger.error("Could not recover tool calls from failed generation")

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

    async def _execute(tool_call: dict) -> ToolMessage:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        logger.info("Executing tool '%s' with args: %s", name, args)

        tool = agent.tools_by_name[name]

        try:
            if hasattr(tool, "ainvoke"):
                observation = await tool.ainvoke(args)
            elif hasattr(tool, "arun"):
                observation = await tool.arun(args)
            else:
                observation = await asyncio.to_thread(
                    tool.invoke if hasattr(tool, "invoke") else tool.run, args
                )
            logger.info("Tool '%s' completed — result length: %d chars", name, len(str(observation)))
        except Exception:
            logger.exception("Tool '%s' failed", name)
            raise

        return ToolMessage(content=str(observation), tool_call_id=tool_call["id"])

    try:
        results = await asyncio.gather(*[_execute(tc) for tc in tool_calls])
    except Exception as e:
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
                results = await asyncio.gather(*[_execute(tc) for tc in tool_calls])
            else:
                raise
        else:
            raise

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

    # No tool calls means the LLM produced a final response — always stop.
    # Summarization after the final response is unnecessary (the conversation
    # is over) and would create an infinite loop: summarize → llm_call →
    # final response → summarize → llm_call → ...
    logger.debug("Routing → END")
    return END