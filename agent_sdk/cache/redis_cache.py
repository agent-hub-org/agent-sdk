"""
Redis-backed TTL cache with automatic in-memory fallback (cachetools.TTLCache).

Usage:
    cache = RedisCache(prefix="embed", ttl=600, maxsize=512)
    await cache.init()              # call in lifespan startup
    await cache.set(key, value)     # value must be JSON-serializable
    result = await cache.get(key)   # None on miss
    await cache.close()             # call in lifespan teardown
"""

import json
import logging
import os
from typing import Any

logger = logging.getLogger("agent_sdk.cache.redis_cache")


class RedisCache:
    def __init__(self, prefix: str, ttl: int, maxsize: int = 1000,
                 redis_url: str | None = None) -> None:
        self._prefix = prefix
        self._ttl = ttl
        self._maxsize = maxsize
        self._redis_url = redis_url
        self._redis = None
        self._fallback = None
        self._degraded = False

    async def init(self) -> bool:
        """Connect to Redis. Returns True if successful, False if using in-memory fallback."""
        url = self._redis_url or os.getenv("REDIS_URL")
        if not url:
            self._use_fallback("REDIS_URL not set")
            return False
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(url, decode_responses=True, socket_connect_timeout=3)
            await client.ping()
            self._redis = client
            logger.info("RedisCache '%s': connected to Redis", self._prefix)
            return True
        except Exception as exc:  # noqa: BLE001
            self._use_fallback(str(exc))
            return False

    def _use_fallback(self, reason: str) -> None:
        from cachetools import TTLCache
        self._fallback = TTLCache(maxsize=self._maxsize, ttl=self._ttl)
        self._degraded = True
        logger.warning(
            "RedisCache '%s': using in-memory fallback (%s)", self._prefix, reason
        )

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        if self._degraded:
            return self._fallback.get(key)
        try:
            raw = await self._redis.get(self._key(key))
            return json.loads(raw) if raw is not None else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("RedisCache.get key=%s: %s", key, exc)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._ttl
        if self._degraded:
            self._fallback[key] = value
            return
        try:
            await self._redis.setex(self._key(key), effective_ttl, json.dumps(value))
        except Exception as exc:  # noqa: BLE001
            logger.debug("RedisCache.set key=%s: %s", key, exc)

    async def delete(self, key: str) -> None:
        if self._degraded:
            self._fallback.pop(key, None)
            return
        try:
            await self._redis.delete(self._key(key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("RedisCache.delete key=%s: %s", key, exc)

    async def clear(self) -> None:
        if self._degraded:
            self._fallback.clear()
            return
        try:
            cursor = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match=f"{self._prefix}:*", count=100)
                if keys:
                    await self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.debug("RedisCache.clear prefix=%s: %s", self._prefix, exc)

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
