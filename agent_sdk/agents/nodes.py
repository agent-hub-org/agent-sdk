from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from agent_sdk.agents.state import AgentState

logger = logging.getLogger("agent_sdk.nodes")


async def initialize(state: AgentState) -> dict:
    """
    Runs once at START before the agent loop.
    Inserts the system prompt as the first message in state so it is
    persisted by the checkpointer and never re-sent on subsequent LLM calls.
    If no system_prompt is provided, a default is used.

    Does not require agent dependencies — stays a plain function.
    """
    # Only inject if no SystemMessage is already present
    if state.messages and isinstance(state.messages[0], SystemMessage):
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

    logger.info("LLM call — iteration %d/%d, message count: %d",
                state.iteration + 1, state.max_iterations, len(state.messages))

    tools = list(agent.tools_by_name.values())
    llm_with_tools = agent.llm.bind_tools(tools) if tools else agent.llm

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

    if hasattr(llm_with_tools, "ainvoke"):
        response = await llm_with_tools.ainvoke(prompt)
    else:
        response = llm_with_tools.invoke(prompt)

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

    existing_summary = state.summary or ""
    if existing_summary:
        summary_message = (
            f"This is a summary of the conversation to date: {existing_summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Pass full message history + instruction to the summarizer
    messages = list(state.messages) + [HumanMessage(content=summary_message)]

    summarizer = agent.summarizer or agent.llm
    if hasattr(summarizer, "ainvoke"):
        response = await summarizer.ainvoke(messages)
    else:
        response = summarizer.invoke(messages)

    # Prune old messages from state — keep only the most recent N
    keep_n = max(state.keep_last_n_messages, 1)
    delete_messages = [RemoveMessage(id=m.id) for m in state.messages[:-keep_n]]

    logger.info("Summarization complete — pruned %d messages, keeping last %d",
                len(delete_messages), keep_n)

    return {"summary": response.content, "messages": delete_messages}


async def tool_node(agent, state: AgentState) -> dict:
    """
    Execute any tool calls from the last assistant message and
    return the resulting tool messages.

    `agent` is bound via functools.partial at graph build time.
    """

    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    results = []
    for tool_call in tool_calls:
        name = tool_call["name"]
        args = tool_call.get("args", {})
        logger.info("Executing tool '%s' with args: %s", name, args)

        tool = agent.tools_by_name[name]

        try:
            # Prefer async tool invocation when available
            if hasattr(tool, "ainvoke"):
                observation = await tool.ainvoke(args)
            elif hasattr(tool, "arun"):
                observation = await tool.arun(args)
            else:
                observation = tool.invoke(args) if hasattr(tool, "invoke") else tool.run(args)

            logger.info("Tool '%s' completed — result length: %d chars", name, len(str(observation)))
        except Exception:
            logger.exception("Tool '%s' failed", name)
            raise

        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tool_node", "summarize_conversation", "__end__"]:
    """
    Decide whether the autonomous agent should keep going:
    - If the last assistant message requested tools, route to the tool node.
    - If the conversation is too long and summarization is enabled, summarize.
    - If we've hit the iteration limit or there are no tool calls, stop.

    Does not require agent dependencies — stays a plain function.
    """

    # Stop if we've reached the configured iteration limit
    if state.iteration >= state.max_iterations:
        logger.warning("Iteration limit reached (%d), stopping agent", state.max_iterations)
        return END

    last_message = state.messages[-1]

    # If the LLM makes a tool call, perform the action before summarizing
    if getattr(last_message, "tool_calls", None):
        logger.debug("Routing → tool_node")
        return "tool_node"

    # Summarize if conversation has grown beyond the retention window
    if (
        state.enable_summarization
        and len(state.messages) > state.keep_last_n_messages
    ):
        logger.debug("Routing → summarize_conversation (%d messages > %d limit)",
                     len(state.messages), state.keep_last_n_messages)
        return "summarize_conversation"

    logger.debug("Routing → END")
    return END