"""
Production memory backend.

Mem0MongoMemoryBackend:
  - Snapshot memory  → Mem0  (tagged per user + session)
  - Episodic memory  → Mem0  (tagged per user)
  - Perspective      → MongoDB  (upserted per user_id, latest wins)

Snapshot summaries are also held in an in-process dict so the manager can
compile episodic content from the actual text without an extra Mem0 round-trip.
This dict resets on process restart — the episodic threshold counter simply
resets as well, which is acceptable.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from agent_sdk.memory.backend import MemoryBackend

logger = logging.getLogger("agent_sdk.memory.prod_backend")

# Mem0 metadata tags
_TYPE_SNAPSHOT = "snapshot"
_TYPE_EPISODIC  = "episodic"


class Mem0MongoMemoryBackend(MemoryBackend):
    """
    Production 3-tier memory backend.

    Storage layout:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Mem0 (cloud vector store)                                       │
    │    user_id memories — metadata: {memory_type: "snapshot",        │
    │                                   session_id: "<id>"}            │
    │    user_id memories — metadata: {memory_type: "episodic"}        │
    ├──────────────────────────────────────────────────────────────────┤
    │  MongoDB  collection: perspective_memory                         │
    │    { user_id, content, updated_at }   ← upserted per user       │
    └──────────────────────────────────────────────────────────────────┘

    In-process cache (resets on restart):
      _session_snapshots  "user_id:session_id" → list[{summary, timestamp}]
    """

    def __init__(
        self,
        mongo_uri: str | None = None,
        db_name: str = "agent_memory",
        perspective_collection: str = "perspective_memory",
    ) -> None:
        self._mongo_uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self._db_name = db_name
        self._perspective_col = perspective_collection

        self._mem0_client = None
        self._motor_client = None

        # In-process snapshot cache: "user_id:session_id" → [{"summary": ..., "timestamp": ...}]
        self._session_snapshots: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mem0(self):
        if self._mem0_client is None:
            from mem0 import MemoryClient
            api_key = os.getenv("MEM0_API_KEY")
            if not api_key:
                raise ValueError("MEM0_API_KEY environment variable is not set.")
            logger.info("Mem0MongoMemoryBackend: initialising Mem0 client")
            self._mem0_client = MemoryClient(api_key=api_key)
        return self._mem0_client

    def _get_collection(self, name: str):
        if self._motor_client is None:
            from motor.motor_asyncio import AsyncIOMotorClient
            logger.info("Mem0MongoMemoryBackend: connecting to MongoDB")
            self._motor_client = AsyncIOMotorClient(self._mongo_uri)
        return self._motor_client[self._db_name][name]

    async def _mem0_add(self, messages: list[dict], user_id: str, metadata: dict) -> None:
        """Run synchronous Mem0 client.add() in a thread pool."""
        client = self._get_mem0()
        await asyncio.to_thread(
            client.add,
            messages,
            user_id=user_id,
            metadata=metadata,
        )

    async def _mem0_get_all(self, user_id: str) -> list[dict]:
        """Run synchronous Mem0 client.get_all() in a thread pool."""
        client = self._get_mem0()
        result = await asyncio.to_thread(client.get_all, user_id=user_id)
        # Mem0 returns either a list or a dict with a "results" key
        if isinstance(result, dict):
            return result.get("results", [])
        return result or []

    # ------------------------------------------------------------------
    # MemoryBackend implementation
    # ------------------------------------------------------------------

    async def save_snapshot(self, user_id: str, session_id: str, snapshot: dict) -> None:
        """
        1. Cache snapshot locally (for episodic compilation this session).
        2. Persist to Mem0 with memory_type=snapshot tag.
        """
        key = f"{user_id}:{session_id}"
        self._session_snapshots.setdefault(key, []).append(snapshot)

        messages = [{"role": "assistant", "content": snapshot.get("summary", "")}]
        metadata = {"memory_type": _TYPE_SNAPSHOT, "session_id": session_id}

        try:
            await self._mem0_add(messages, user_id=user_id, metadata=metadata)
            logger.info(
                "Mem0MongoMemoryBackend: snapshot saved to Mem0 — user=%s session=%s",
                user_id, session_id,
            )
        except Exception:
            logger.exception(
                "Mem0MongoMemoryBackend: failed to save snapshot to Mem0 — user=%s",
                user_id,
            )

    async def get_snapshots(self, user_id: str, session_id: str) -> list[dict]:
        """Return snapshots for this session.

        Reads from the in-process cache first (fast path). On cache miss — e.g.
        after a process restart — falls back to Mem0, repopulates the cache, and
        returns the restored list so the episodic threshold counter is correct.
        """
        key = f"{user_id}:{session_id}"
        cached = self._session_snapshots.get(key)
        if cached:
            return list(cached)

        # Cache miss: reconstruct from Mem0 (process restart scenario)
        try:
            all_memories = await self._mem0_get_all(user_id=user_id)
            snapshots = [
                {"summary": r["memory"], "timestamp": r.get("created_at", "")}
                for r in all_memories
                if isinstance(r.get("metadata"), dict)
                and r["metadata"].get("memory_type") == _TYPE_SNAPSHOT
                and r["metadata"].get("session_id") == session_id
                and r.get("memory")
            ]
            self._session_snapshots[key] = snapshots
            logger.info(
                "get_snapshots: restored %d snapshot(s) from Mem0 after cache miss (user=%s session=%s)",
                len(snapshots), user_id, session_id,
            )
            return list(snapshots)
        except Exception:
            logger.exception(
                "get_snapshots: Mem0 fallback failed — user=%s session=%s", user_id, session_id
            )
            return []

    async def reset_snapshots(self, user_id: str, session_id: str) -> None:
        """Clear the local cache after episodic compilation so the counter resets."""
        key = f"{user_id}:{session_id}"
        self._session_snapshots.pop(key, None)
        logger.debug("Mem0MongoMemoryBackend: reset snapshot cache for %s", key)

    async def save_episodic(self, user_id: str, content: str) -> None:
        """Persist episodic narrative to Mem0 with memory_type=episodic tag."""
        messages = [{"role": "assistant", "content": content}]
        metadata = {"memory_type": _TYPE_EPISODIC}

        try:
            await self._mem0_add(messages, user_id=user_id, metadata=metadata)
            logger.info(
                "Mem0MongoMemoryBackend: episodic memory saved to Mem0 — user=%s",
                user_id,
            )
        except Exception:
            logger.exception(
                "Mem0MongoMemoryBackend: failed to save episodic to Mem0 — user=%s",
                user_id,
            )

    async def get_episodic(self, user_id: str, limit: int = 10) -> list[str]:
        """
        Retrieve episodic memories from Mem0, filtered by memory_type=episodic.
        Returns the `memory` field (Mem0's processed/compressed version).
        """
        try:
            all_memories = await self._mem0_get_all(user_id=user_id)
            episodic = [
                r["memory"]
                for r in all_memories
                if isinstance(r.get("metadata"), dict)
                and r["metadata"].get("memory_type") == _TYPE_EPISODIC
                and r.get("memory")
            ]
            return episodic[-limit:]
        except Exception:
            logger.exception(
                "Mem0MongoMemoryBackend: failed to get episodic from Mem0 — user=%s",
                user_id,
            )
            return []

    async def save_perspective(self, user_id: str, content: str) -> None:
        """Upsert perspective document in MongoDB (latest wins)."""
        col = self._get_collection(self._perspective_col)
        try:
            await col.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "user_id": user_id,
                        "content": content,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
                upsert=True,
            )
            logger.info(
                "Mem0MongoMemoryBackend: perspective saved to MongoDB — user=%s (%d chars)",
                user_id, len(content),
            )
        except Exception:
            logger.exception(
                "Mem0MongoMemoryBackend: failed to save perspective to MongoDB — user=%s",
                user_id,
            )

    async def get_perspective(self, user_id: str) -> Optional[str]:
        """Retrieve perspective document from MongoDB."""
        col = self._get_collection(self._perspective_col)
        try:
            doc = await col.find_one({"user_id": user_id}, {"content": 1})
            return doc["content"] if doc else None
        except Exception:
            logger.exception(
                "Mem0MongoMemoryBackend: failed to get perspective from MongoDB — user=%s",
                user_id,
            )
            return None
