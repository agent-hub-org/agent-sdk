from __future__ import annotations

from typing import Literal, Sequence

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END

from agent_sdk.agents.state import AgentState


def _estimate_token_usage(messages: Sequence) -> int:
    """Very rough token estimate based on character length."""

    total_chars = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total_chars += len(content)
    # ~4 characters per token as a crude heuristic
    return total_chars // 4


async def llm_call(state: AgentState) -> dict:
    """
    Core autonomous agent step:
    - Optionally summarize long history into `summary` when context is too large.
    - Call the LLM (bound with tools) with summary + recent messages.
    - Return a new assistant message (which may include tool calls).
    - Increment the iteration counter.
    """

    tools = list(state.tools_by_name.values())
    llm_with_tools = state.llm.bind_tools(tools) if tools else state.llm

    messages = list(state.messages)
    new_summary = None

    # Summarization logic: collapse long history into a short summary + recent turns
    if state.enable_summarization and state.summarizer_llm is not None:
        estimated_tokens = _estimate_token_usage(messages)
        if estimated_tokens > state.max_context_tokens:
            keep_n = max(state.keep_last_n_messages, 0)
            if keep_n > 0 and len(messages) > keep_n:
                old_messages = messages[:-keep_n]
                recent_messages = messages[-keep_n:]
            else:
                old_messages = messages
                recent_messages = messages

            if old_messages:
                summary_prompt: Sequence = [
                    SystemMessage(
                        content=(
                            "Summarize the following conversation history "
                            "concisely so it can be used as context for future turns."
                        )
                    )
                ]
                if state.summary:
                    summary_prompt.append(
                        SystemMessage(
                            content=f"Previous summary:\n{state.summary}"
                        )
                    )
                summary_prompt.extend(old_messages)

                # Prefer async if available for low latency
                if hasattr(state.summarizer_llm, "ainvoke"):
                    summary_msg = await state.summarizer_llm.ainvoke(summary_prompt)
                else:
                    summary_msg = state.summarizer_llm.invoke(summary_prompt)
                new_summary = getattr(summary_msg, "content", str(summary_msg))

                # For the LLM prompt, use the summary plus the recent messages.
                messages = [
                    SystemMessage(content=f"Conversation summary:\n{new_summary}")
                ]
                messages.extend(recent_messages)

    prompt: Sequence = [
        SystemMessage(
            content=(
                "You are an autonomous assistant. "
                "You may call tools to achieve the user's goal, "
                "or respond directly when tools are not needed."
            )
        ),
        *messages,
    ]

    # Prefer async model call when available
    if hasattr(llm_with_tools, "ainvoke"):
        response = await llm_with_tools.ainvoke(prompt)
    else:
        response = llm_with_tools.invoke(prompt)

    result = {
        "messages": [response],
        "iteration": state.iteration + 1,
    }
    if new_summary is not None:
        result["summary"] = new_summary

    return result


async def tool_node(state: AgentState) -> dict:
    """
    Execute any tool calls from the last assistant message and
    return the resulting tool messages.
    """

    last_message = state.messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    results = []
    for tool_call in tool_calls:
        tool = state.tools_by_name[tool_call["name"]]
        args = tool_call.get("args", {})

        # Prefer async tool invocation when available
        if hasattr(tool, "ainvoke"):
            observation = await tool.ainvoke(args)
        elif hasattr(tool, "arun"):
            observation = await tool.arun(args)
        else:
            observation = tool.invoke(args) if hasattr(tool, "invoke") else tool.run(args)

        results.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        )

    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """
    Decide whether the autonomous agent should keep going:
    - If the last assistant message requested tools, route to the tool node.
    - If we've hit the iteration limit or there are no tool calls, stop.
    """

    # Stop if we've reached the configured iteration limit
    if state.iteration >= state.max_iterations:
        return END

    last_message = state.messages[-1]

    # If the LLM makes a tool call, then perform an action
    if getattr(last_message, "tool_calls", None):
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END
