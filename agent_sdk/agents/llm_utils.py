"""LLM invocation utilities: retry logic and running-context compression."""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent_sdk.config import settings

logger = logging.getLogger("agent_sdk.agents.llm_utils")

_RETRYABLE_HTTP_CODES = frozenset({429, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True
    status = (
        getattr(exc, "status_code", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
    )
    return status in _RETRYABLE_HTTP_CODES


async def invoke_with_retry(llm_bound, prompt, max_retries: int | None = None, base_delay: float | None = None):
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


async def compress_running_context(agent, text: str) -> str:
    """Compress running_context to ~2000 chars. Runs as a background task during tool execution."""
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
