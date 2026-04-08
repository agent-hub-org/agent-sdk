"""
Memory backends for the 3-tier memory system.

MemoryBackend defines the storage interface.
InMemoryBackend is the default (dev/test) implementation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("agent_sdk.memory.backend")


class MemoryBackend(ABC):
    """
    Abstract storage interface for all three memory tiers.

    Implementations should be safe to call concurrently from asyncio tasks.
    All methods are async to support remote/database backends.
    """

    @abstractmethod
    async def save_snapshot(self, user_id: str, session_id: str, snapshot: dict) -> None:
        """Persist a snapshot memory entry for (user_id, session_id)."""

    @abstractmethod
    async def get_snapshots(self, user_id: str, session_id: str) -> list[dict]:
        """Retrieve all snapshot memories for (user_id, session_id), oldest first."""

    @abstractmethod
    async def save_episodic(self, user_id: str, content: str) -> None:
        """Append an episodic memory entry for user_id."""

    @abstractmethod
    async def get_episodic(self, user_id: str, limit: int = 10) -> list[str]:
        """Retrieve the most recent `limit` episodic memories for user_id."""

    async def reset_snapshots(self, user_id: str, session_id: str) -> None:
        """
        Clear accumulated snapshots for (user_id, session_id) after episodic compilation
        so the threshold counter resets for the next window.
        Default implementation is a no-op (backends that persist to external stores
        should override this if they maintain a local cache).
        """

    @abstractmethod
    async def save_perspective(self, user_id: str, content: str) -> None:
        """Overwrite the perspective memory for user_id."""

    @abstractmethod
    async def get_perspective(self, user_id: str) -> Optional[str]:
        """Retrieve the current perspective memory for user_id, or None."""


class InMemoryBackend(MemoryBackend):
    """
    In-process, non-persistent backend for development and testing.

    Data is lost when the process exits. For production use, replace with
    a MongoDB or Redis backend that persists across restarts.
    """

    def __init__(self) -> None:
        # snapshots keyed by "user_id:session_id"
        self._snapshots: dict[str, list[dict]] = {}
        # episodic keyed by user_id (list of strings)
        self._episodic: dict[str, list[str]] = {}
        # perspective keyed by user_id (single string, latest wins)
        self._perspective: dict[str, str] = {}

    async def save_snapshot(self, user_id: str, session_id: str, snapshot: dict) -> None:
        key = f"{user_id}:{session_id}"
        self._snapshots.setdefault(key, []).append(snapshot)
        logger.debug("InMemoryBackend: saved snapshot for %s (total: %d)",
                     key, len(self._snapshots[key]))

    async def get_snapshots(self, user_id: str, session_id: str) -> list[dict]:
        return list(self._snapshots.get(f"{user_id}:{session_id}", []))

    async def reset_snapshots(self, user_id: str, session_id: str) -> None:
        self._snapshots.pop(f"{user_id}:{session_id}", None)
        logger.debug("InMemoryBackend: reset snapshot cache for %s:%s", user_id, session_id)

    async def save_episodic(self, user_id: str, content: str) -> None:
        self._episodic.setdefault(user_id, []).append(content)
        logger.debug("InMemoryBackend: saved episodic for %s (total: %d)",
                     user_id, len(self._episodic[user_id]))

    async def get_episodic(self, user_id: str, limit: int = 10) -> list[str]:
        return list(self._episodic.get(user_id, []))[-limit:]

    async def save_perspective(self, user_id: str, content: str) -> None:
        self._perspective[user_id] = content
        logger.debug("InMemoryBackend: updated perspective for %s (%d chars)",
                     user_id, len(content))

    async def get_perspective(self, user_id: str) -> Optional[str]:
        return self._perspective.get(user_id)
