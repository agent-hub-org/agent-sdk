"""
MemoryManager — orchestrates the 3-tier memory system.

Tier 1 — Snapshot Memory (per query, per session)
    A 2-4 sentence LLM summary of what the user asked, what tools were used,
    and what was answered. Created after every completed query.

Tier 2 — Episodic Memory (per N snapshots, per user)
    A narrative compilation of `episodic_threshold` (default: 15) snapshots
    covering what topics were explored and what patterns emerged in the
    conversation. Created when the snapshot threshold is reached.

Tier 3 — Perspective Memory (per user, rolling)
    A personality profile of the user derived from all episodic memories.
    Captures communication style, interests, and behavioral tendencies.
    Updated every time a new episodic memory is created.

    CRITICAL: Perspective memory is injected as background context ONLY.
    It must NEVER influence analytical conclusions, factual content, or
    planning decisions — only communication style adaptation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Any

from agent_sdk.memory.backend import MemoryBackend

logger = logging.getLogger("agent_sdk.memory.manager")

_PERSPECTIVE_CACHE: dict[str, tuple[Optional[str], float]] = {}
_PERSPECTIVE_CACHE_TTL = 300.0  # seconds

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SNAPSHOT_PROMPT = """\
Summarize the following question-and-answer exchange in 2-4 sentences.
Include: what the user asked, key findings from the research (if any), and what was answered.
Be factual and concise. Do not add opinions or analysis.

User query:
{query}

Tool findings (may be empty for simple queries):
{scratchpad}

Agent response:
{response}"""

_EPISODIC_PROMPT = """\
Below are {n} Q&A summaries from a user's conversation session.
Write a single 3-5 sentence narrative that captures:
- What topics the user explored in this session
- What types of questions they asked most
- What they found most valuable or returned to repeatedly

Summaries:
{summaries}"""

_PERSPECTIVE_PROMPT = """\
Based on the conversation history summaries below, write a concise 3-5 sentence user personality profile.
Focus exclusively on HOW this user interacts (communication style, vocabulary level, detail preference,
follow-up patterns, domain expertise signals) — NOT what specific topics they asked about.

If there is insufficient data for a confident assessment, say so briefly and describe only what is clear.

Conversation history (episodic summaries):
{episodic}"""


class MemoryManager:
    """
    Orchestrates the snapshot → episodic → perspective memory pipeline.

    Usage:
        manager = MemoryManager(backend=InMemoryBackend(), llm=agent.llm)
        agent = BaseAgent(..., memory_manager=manager)

    The manager is accessed inside graph nodes and fires background tasks
    so memory writes never add latency to the user-facing response.
    """

    def __init__(
        self,
        backend: MemoryBackend,
        llm: Any = None,
        episodic_threshold: int = 15,
    ) -> None:
        self.backend = backend
        self.llm = llm  # set by BaseAgent after LLM is initialized
        self.episodic_threshold = episodic_threshold

    # ------------------------------------------------------------------
    # Public API called by memory_writer node
    # ------------------------------------------------------------------

    async def process_query(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        scratchpad: Optional[str],
        llm: Any,  # agent.llm (passed at call time so node doesn't need to store it)
    ) -> None:
        """
        Full pipeline called after every completed query:
          1. Create and persist snapshot memory.
          2. If snapshot threshold reached → compile episodic memory.
          3. If new episodic → update perspective memory.

        This runs in a background asyncio task — it does NOT block the response.
        """
        effective_llm = self.llm or llm
        if effective_llm is None:
            logger.warning("MemoryManager.process_query: no LLM available — skipping memory write")
            return

        try:
            # --- Tier 1: Snapshot ---
            snapshot = await self._create_snapshot(
                query=query,
                response=response,
                scratchpad=scratchpad,
                llm=effective_llm,
            )
            await self.backend.save_snapshot(user_id, session_id, snapshot)
            logger.info("MemoryManager: snapshot saved for user=%s session=%s", user_id, session_id)

            # --- Tier 2: Episodic (threshold check) ---
            # Requires at least episodic_threshold snapshots accumulated this session.
            snapshots = await self.backend.get_snapshots(user_id, session_id)
            if len(snapshots) >= self.episodic_threshold:
                # Use the oldest un-compiled batch of threshold-sized snapshots
                batch = snapshots[-self.episodic_threshold:]
                episodic_content = await self._compile_episodic(batch, effective_llm)
                await self.backend.save_episodic(user_id, episodic_content)
                logger.info("MemoryManager: episodic memory compiled for user=%s (%d snapshots)",
                            user_id, len(batch))

                # Clear the session snapshot cache after compilation so the counter
                # resets and the next episodic window starts fresh.
                await self._reset_session_snapshots(user_id, session_id)

                # --- Tier 3: Perspective (updated after each new episodic) ---
                await self._update_perspective(user_id, effective_llm)

        except Exception:
            logger.exception(
                "MemoryManager.process_query: error during memory pipeline "
                "(user=%s session=%s) — ignoring to avoid disrupting agent",
                user_id, session_id,
            )

    async def _reset_session_snapshots(self, user_id: str, session_id: str) -> None:
        """Clear the in-session snapshot cache so the episodic counter starts fresh."""
        if hasattr(self.backend, "reset_snapshots"):
            await self.backend.reset_snapshots(user_id, session_id)

    async def get_perspective(self, user_id: str) -> Optional[str]:
        """Retrieve the current perspective memory for injection into the system prompt."""
        cached_result, fetched_at = _PERSPECTIVE_CACHE.get(user_id, (None, 0))
        if cached_result and time.monotonic() - fetched_at < _PERSPECTIVE_CACHE_TTL:
            return cached_result

        try:
            result = await self.backend.get_perspective(user_id)
            if len(_PERSPECTIVE_CACHE) > 1000:
                _PERSPECTIVE_CACHE.clear()  # simple bound
            _PERSPECTIVE_CACHE[user_id] = (result, time.monotonic())
            return result
        except Exception:
            logger.exception("MemoryManager.get_perspective: error reading perspective for user=%s", user_id)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _create_snapshot(
        self,
        query: str,
        response: str,
        scratchpad: Optional[str],
        llm: Any,
    ) -> dict:
        """Call the LLM to produce a 2-4 sentence snapshot summary."""
        from langchain_core.messages import HumanMessage, SystemMessage

        scratchpad_text = scratchpad or "(no tool calls — direct answer)"
        # Truncate to avoid overloading the summarizer
        if len(scratchpad_text) > 3000:
            scratchpad_text = scratchpad_text[:3000] + "\n[truncated]"
        if len(response) > 2000:
            response = response[:2000] + "[truncated]"

        prompt = [
            SystemMessage(content=(
                "You are a concise assistant that summarizes Q&A exchanges. "
                "Output only the summary — no preamble, no markdown."
            )),
            HumanMessage(content=_SNAPSHOT_PROMPT.format(
                query=query[:500],
                scratchpad=scratchpad_text,
                response=response,
            )),
        ]

        try:
            resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=30.0)
            summary = resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            logger.warning("MemoryManager: snapshot LLM call failed — using truncated query as fallback")
            summary = f"User asked: {query[:200]}"

        return {
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _compile_episodic(self, snapshots: list[dict], llm: Any) -> str:
        """Compile a list of snapshots into a single episodic narrative."""
        from langchain_core.messages import HumanMessage, SystemMessage

        summaries_text = "\n\n".join(
            f"{i + 1}. {s.get('summary', '')}"
            for i, s in enumerate(snapshots)
        )

        prompt = [
            SystemMessage(content=(
                "You are a concise assistant that summarizes conversation sessions. "
                "Output only the narrative — no preamble, no markdown."
            )),
            HumanMessage(content=_EPISODIC_PROMPT.format(
                n=len(snapshots),
                summaries=summaries_text,
            )),
        ]

        try:
            resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=30.0)
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            logger.warning("MemoryManager: episodic compilation LLM call failed — using summary concat")
            return f"Session covering {len(snapshots)} queries."

    async def _update_perspective(self, user_id: str, llm: Any) -> None:
        """Recompile perspective memory from all episodic memories for this user."""
        from langchain_core.messages import HumanMessage, SystemMessage

        episodic_entries = await self.backend.get_episodic(user_id, limit=20)
        if not episodic_entries:
            return

        episodic_text = "\n\n---\n\n".join(
            f"Session {i + 1}:\n{e}"
            for i, e in enumerate(episodic_entries)
        )

        prompt = [
            SystemMessage(content=(
                "You are a concise assistant that builds user personality profiles. "
                "Focus only on communication style and behavior — not content of queries. "
                "Output only the profile — no preamble, no markdown headers."
            )),
            HumanMessage(content=_PERSPECTIVE_PROMPT.format(episodic=episodic_text)),
        ]

        try:
            resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=30.0)
            perspective = resp.content if hasattr(resp, "content") else str(resp)
            await self.backend.save_perspective(user_id, perspective)
            _PERSPECTIVE_CACHE[user_id] = (perspective, time.monotonic())
            logger.info("MemoryManager: perspective memory updated for user=%s (%d chars)",
                        user_id, len(perspective))
        except Exception:
            logger.warning("MemoryManager: perspective update LLM call failed for user=%s", user_id)
