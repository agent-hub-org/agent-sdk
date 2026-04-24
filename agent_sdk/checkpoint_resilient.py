"""
ResilientCheckpointer — wraps any BaseCheckpointSaver (typically AsyncMongoDBSaver)
and falls back to InMemorySaver on write failures, preventing agent hangs when
MongoDB is temporarily unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger("agent_sdk.checkpoint_resilient")


class ResilientCheckpointer(BaseCheckpointSaver):
    """Wraps a primary checkpointer and falls back to InMemorySaver on write failure.

    On the first write failure the checkpointer logs a WARNING and switches to
    the in-memory fallback for the remainder of the process lifetime. This keeps
    the agent functional even when MongoDB is unreachable, at the cost of losing
    cross-restart session persistence for that instance.

    Read methods always try the primary first; if degraded, they use the fallback
    since recent state is in-memory.
    """

    def __init__(self, primary: BaseCheckpointSaver) -> None:
        super().__init__()
        self._primary = primary
        self._fallback = InMemorySaver()
        self._degraded = False

    @property
    def _active_writer(self) -> BaseCheckpointSaver:
        return self._fallback if self._degraded else self._primary

    def _mark_degraded(self, exc: Exception) -> None:
        if not self._degraded:
            logger.warning(
                "Checkpointer write failed — switching to InMemorySaver for this process: %s",
                exc,
            )
            self._degraded = True

    # ── Sync stubs (not used in async agents) ─────────────────────────────────

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError("Sync methods not supported in ResilientCheckpointer")

    def list(self, config, *, filter=None, before=None, limit=None):
        raise NotImplementedError("Sync methods not supported in ResilientCheckpointer")

    def put(self, config, checkpoint, metadata, new_versions):
        raise NotImplementedError("Sync methods not supported in ResilientCheckpointer")

    def put_writes(self, config, writes, task_id, task_path=""):
        raise NotImplementedError("Sync methods not supported in ResilientCheckpointer")

    # ── Async reads — try primary; use fallback if degraded ───────────────────

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if self._degraded:
            return await self._fallback.aget_tuple(config)
        try:
            return await self._primary.aget_tuple(config)
        except Exception as exc:
            logger.warning("Checkpointer read failed, trying fallback: %s", exc)
            return await self._fallback.aget_tuple(config)

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        source = self._fallback if self._degraded else self._primary
        async for item in source.alist(config, filter=filter, before=before, limit=limit):
            yield item

    # ── Async writes — fall back to InMemorySaver on any error ───────────────

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        try:
            return await self._active_writer.aput(config, checkpoint, metadata, new_versions)
        except Exception as exc:
            self._mark_degraded(exc)
            return await self._fallback.aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        try:
            await self._active_writer.aput_writes(config, writes, task_id, task_path)
        except Exception as exc:
            self._mark_degraded(exc)
            await self._fallback.aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        try:
            await self._primary.adelete_thread(thread_id)
        except Exception as exc:
            logger.warning("Checkpointer delete_thread failed: %s", exc)
        if self._degraded:
            await self._fallback.adelete_thread(thread_id)
