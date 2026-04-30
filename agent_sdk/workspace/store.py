from __future__ import annotations

import json
import logging
from cachetools import TTLCache

logger = logging.getLogger("agent_sdk.workspace.store")


class WorkspaceStore:
    """Redis-backed request-scoped store for sub-agent outputs.

    Falls back to in-process TTLCache when Redis is unavailable.
    Keys follow the pattern: workspace:{workspace_id}:{agent_name}
    """

    def __init__(
        self,
        redis_url: str | None = None,
        fallback_ttl: int = 1800,
        fallback_maxsize: int = 500,
    ) -> None:
        self._redis = None
        self._redis_url = redis_url
        self._fallback: TTLCache = TTLCache(maxsize=fallback_maxsize, ttl=fallback_ttl)
        self._degraded = True

    async def init(self) -> None:
        if not self._redis_url:
            logger.info("WorkspaceStore: no Redis URL — using in-memory fallback")
            return
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            self._degraded = False
            logger.info("WorkspaceStore: connected to Redis at %s", self._redis_url)
        except Exception as exc:
            logger.warning("WorkspaceStore: Redis unavailable (%s) — using in-memory fallback", exc)
            self._degraded = True

    async def read(self, workspace_id: str, key: str) -> dict | None:
        full_key = f"workspace:{workspace_id}:{key}"
        if not self._degraded and self._redis:
            raw = await self._redis.get(full_key)
            return json.loads(raw) if raw else None
        return self._fallback.get(full_key)

    async def write(
        self, workspace_id: str, key: str, value: dict, ttl: int = 1800
    ) -> None:
        full_key = f"workspace:{workspace_id}:{key}"
        if not self._degraded and self._redis:
            await self._redis.setex(full_key, ttl, json.dumps(value))
        else:
            self._fallback[full_key] = value

    async def flush(self, workspace_id: str) -> None:
        prefix = f"workspace:{workspace_id}:"
        if not self._degraded and self._redis:
            keys = [k async for k in self._redis.scan_iter(f"{prefix}*")]
            if keys:
                await self._redis.delete(*keys)
        else:
            to_delete = [k for k in list(self._fallback.keys()) if k.startswith(prefix)]
            for k in to_delete:
                self._fallback.pop(k, None)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
