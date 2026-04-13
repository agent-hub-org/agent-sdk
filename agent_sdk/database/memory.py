import hashlib
import logging
import os
import time
from typing import Optional

from mem0 import MemoryClient

from agent_sdk.config import settings

logger = logging.getLogger("agent_sdk.database.memory")

_client: Optional[MemoryClient] = None

# In-process TTL cache for Mem0 search results.
# Key: (user_id, query_hash), Value: (results_list, expiry_timestamp)
_mem_cache: dict[tuple[str, str], tuple[list[str], float]] = {}
_MEM_CACHE_TTL = 300  # 5 minutes
_MEM_CACHE_MAX = 256  # max entries before LRU eviction


def _get_client() -> MemoryClient:
    global _client
    if _client is None:
        api_key = os.getenv("MEM0_API_KEY")
        if not api_key:
            raise ValueError("MEM0_API_KEY environment variable is not set.")
        logger.info("Initializing Mem0 client")
        _client = MemoryClient(api_key=api_key)
    return _client


def _cache_key(user_id: str, query: str) -> tuple[str, str]:
    q_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()
    return (user_id, q_hash)


def _evict_cache() -> None:
    """Remove expired entries; if still over limit, drop oldest by insertion order."""
    now = time.monotonic()
    expired = [k for k, (_, exp) in _mem_cache.items() if exp <= now]
    for k in expired:
        del _mem_cache[k]
    # If still over limit, drop oldest entries (dict preserves insertion order in 3.7+)
    if len(_mem_cache) >= _MEM_CACHE_MAX:
        overage = len(_mem_cache) - _MEM_CACHE_MAX + 1
        for k in list(_mem_cache.keys())[:overage]:
            del _mem_cache[k]


_MEMORY_SCORE_THRESHOLD = settings.memory_score_threshold


def get_memories(user_id: str, query: str) -> tuple[list[str], str | None]:
    """
    Search Mem0 for facts relevant to the user and the current query.
    Returns (memories, error_msg) — error_msg is None on success, a user-friendly
    string on failure so the caller can surface degradation to the user.

    Results are cached in-process for 5 minutes per (user_id, query) to avoid
    redundant network calls across repeated or similar requests.
    """
    key = _cache_key(user_id, query)
    now = time.monotonic()
    cached = _mem_cache.get(key)
    if cached is not None:
        memories, expiry = cached
        if now < expiry:
            logger.debug("Mem0 cache hit for user='%s'", user_id)
            return memories, None
        # Expired — remove stale entry
        del _mem_cache[key]

    try:
        client = _get_client()
        results = client.search(
            query=query,
            version="v2",
            filters={"user_id": user_id},
            limit=settings.memory_max_results,
        )
        memories = [
            r["memory"][:settings.memory_truncate_chars]
            for r in results.get("results", [])
            if r.get("memory") and r.get("score", 0) >= _MEMORY_SCORE_THRESHOLD
        ]
        if memories:
            logger.info("Retrieved %d memories for user='%s'", len(memories), user_id)
        else:
            logger.info("No relevant memories found for user='%s'", user_id)

        _evict_cache()
        _mem_cache[key] = (memories, now + _MEM_CACHE_TTL)
        return memories, None
    except Exception as e:
        # Memory retrieval failures should never break the agent — degrade gracefully
        logger.warning("Failed to retrieve memories for user='%s': %s", user_id, e)
        return [], "Personalization temporarily unavailable — memory service unreachable."


def save_memory(user_id: str, query: str, response: str) -> None:
    """
    Add the latest conversation turn to Mem0. Mem0 automatically extracts 
    and stores any relevant facts about the user from the exchange.
    """
    try:
        client = _get_client()
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ]
        client.add(messages=messages, user_id=user_id)
        logger.info("Saved conversation to Mem0 for user='%s'", user_id)
    except Exception as e:
        # Memory saving failures should never break the agent — degrade gracefully
        logger.warning("Failed to save memory for user='%s': %s", user_id, e)
