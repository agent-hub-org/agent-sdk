"""Tool execution engine: parallel dispatch, timeout, circuit breaker, summarization."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent_sdk.config import settings
from agent_sdk.mcp.exceptions import MCPSessionError
from agent_sdk.metrics import tool_call_duration

if TYPE_CHECKING:
    from agent_sdk.agents.base_agent import BaseAgent

logger = logging.getLogger("agent_sdk.agents.tool_executor")


async def execute_tool_calls(
    agent: "BaseAgent",
    tool_calls: list[dict],
    timeout: float,
    phase_tools: list | None = None,
) -> list[ToolMessage]:
    """Execute tool calls in parallel with per-tool timeouts and circuit breakers."""
    _lookup = {**agent.tools_by_name, **({t.name: t for t in phase_tools} if phase_tools else {})}

    async def _execute(tool_call: dict) -> tuple[ToolMessage, bool, str]:
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
                observation = await asyncio.to_thread(
                    tool.invoke if hasattr(tool, "invoke") else tool.run, args
                )
            breaker.record_success()
            tool_call_duration.labels(agent="sdk", tool_name=name).observe(time.monotonic() - _tool_t0)
            obs_str = str(observation)
            raw_len = len(obs_str)
            logger.info("Tool '%s' completed — result length: %d chars", name, raw_len)

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
        elif isinstance(e, MCPSessionError):
            logger.warning("MCP session dropped — attempting reconnect and retry")
            if agent._mcp_manager is not None:
                new_tools = await agent._mcp_manager.reconnect()
                agent.tools = list(agent.tools)
                for t in new_tools:
                    agent.tools_by_name[t.name] = t
                agent._phase_tools_cache.clear()
                agent._bound_llm_cache.clear()
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
