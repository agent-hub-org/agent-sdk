"""SemanticMemoryManager: direct Pinecone client for durable user-fact storage.

Provides two operations:
  retrieve(user_id, query) → list of fact strings (at session start, runs in parallel with Mem0)
  consolidate(user_id, conversation, llm) → fire-and-forget extraction + upsert (at session end)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger("agent_sdk.memory.semantic")

_DEDUP_THRESHOLD = 0.88
_DIMENSIONS = 3072  # text-embedding-3-large

_CONSOLIDATION_SYSTEM_PROMPT = (
    "Extract durable facts about the user from this conversation. "
    "Focus on: preferences, risk tolerance, interests, goals, budget, demographics, "
    "domain expertise, and investment style. "
    "Return a JSON array of short fact strings (max 15 words each). "
    'Example: ["prefers large-cap Indian equities", "monthly SIP budget ₹3000", '
    '"risk-conservative investor", "software engineer interested in tech stocks"]. '
    "Return [] if no durable facts are present."
)


class SemanticMemoryManager:
    """Direct Pinecone client for semantic (durable fact) memory.

    Usage:
        mgr = SemanticMemoryManager()
        facts = await mgr.retrieve(user_id="u123", query="investment preferences")
        # fire-and-forget at session end:
        asyncio.create_task(mgr.consolidate(user_id="u123", conversation="...", llm=agent.llm))
    """

    def __init__(self):
        self._pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self._embeddings = OpenAIEmbeddings(
            base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
            api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
            model="text-embedding-3-large",
        )
        self._index = self._ensure_index()

    def _ensure_index(self):
        existing = {idx.name for idx in self._pinecone.list_indexes()}
        if "user-knowledge" not in existing:
            self._pinecone.create_index(
                name="user-knowledge",
                dimension=_DIMENSIONS,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        return self._pinecone.Index("user-knowledge")

    async def retrieve(self, user_id: str, query: str, top_k: int = 5) -> list[str]:
        """Fetch durable semantic facts for a user. Returns [] on error (non-blocking)."""
        try:
            vec = await asyncio.to_thread(self._embeddings.embed_query, query)
            results = await asyncio.to_thread(
                self._index.query,
                vector=vec,
                top_k=top_k,
                filter={"user_id": {"$eq": user_id}},
                include_metadata=True,
            )
            return [
                m.metadata.get("fact", "")
                for m in results.matches
                if m.metadata.get("fact")
            ]
        except Exception as e:
            logger.warning("SemanticMemoryManager.retrieve failed: %s", e)
            return []

    async def consolidate(self, user_id: str, conversation: str, llm) -> None:
        """Extract durable facts from conversation and upsert to user-knowledge index.
        Designed to run as a fire-and-forget background task — errors are logged, not raised."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            resp = await llm.ainvoke([
                SystemMessage(content=_CONSOLIDATION_SYSTEM_PROMPT),
                HumanMessage(content=conversation[:8000]),
            ])
            raw = resp.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()

            facts: list[str] = json.loads(raw)
            if not facts or not isinstance(facts, list):
                logger.debug("consolidate: no durable facts extracted for user=%s", user_id)
                return

            n = await asyncio.to_thread(self._upsert_facts, user_id, facts)
            logger.info("SemanticMemoryManager.consolidate for user=%s: upserted %d facts", user_id, n)
        except Exception as e:
            logger.warning("SemanticMemoryManager.consolidate failed for user=%s: %s", user_id, e)

    def _upsert_facts(self, user_id: str, facts: list[str]) -> int:
        """Sync helper — called via asyncio.to_thread. Upserts with dedup."""
        upserted = 0
        for fact in [f for f in facts if f]:
            vec = self._embeddings.embed_query(fact)
            existing = self._index.query(
                vector=vec,
                top_k=1,
                filter={"user_id": {"$eq": user_id}},
                include_metadata=True,
            )
            if existing.matches and existing.matches[0].score >= _DEDUP_THRESHOLD:
                continue
            self._index.upsert(vectors=[{
                "id": f"{user_id}_{uuid.uuid4().hex[:12]}",
                "values": vec,
                "metadata": {"user_id": user_id, "fact": fact},
            }])
            upserted += 1
        return upserted
