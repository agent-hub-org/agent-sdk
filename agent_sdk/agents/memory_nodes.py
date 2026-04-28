"""LangGraph memory nodes: load_user_context and memory_writer."""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage

from agent_sdk.agents.state import AgentState

logger = logging.getLogger("agent_sdk.nodes.memory")

_PERSPECTIVE_INJECTION_HEADER = """\
[USER PERSONALITY BACKGROUND]
The following is background context about this user's communication style and preferences.
Use it ONLY to adapt tone, vocabulary, and level of detail in your responses.
Do NOT use this to modify analytical conclusions, filter information, add/remove facts,
or influence task planning in any way. It is read-only personality context.

{perspective}
[/USER PERSONALITY BACKGROUND]"""


def _strip_context_block(text: str) -> str:
    marker = "[/CONTEXT]"
    if marker in text:
        return text[text.find(marker) + len(marker):].strip()
    return text.strip()


async def load_user_context(agent, state: AgentState) -> dict:
    """Load Mem0 perspective and semantic memory, inject into state.perspective_context."""
    memory_manager = getattr(agent, "memory_manager", None)
    semantic_memory = getattr(agent, "semantic_memory", None)
    user_id = getattr(state, "user_id", None)

    if not user_id or (memory_manager is None and semantic_memory is None):
        return {}

    last_query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            last_query = _strip_context_block(msg.content)[:200]
            break

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
    """Terminal node: runs deferred summarization then fires the Mem0 pipeline."""
    from agent_sdk.agents.nodes import summarize_conversation, _estimate_token_count

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

    memory_manager = getattr(agent, "memory_manager", None)
    if memory_manager is None:
        return summarization_update

    user_id = getattr(state, "user_id", None)
    if not user_id:
        return summarization_update

    session_id = getattr(state, "session_id", "default")

    query = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            query = _strip_context_block(msg.content)
            break

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
