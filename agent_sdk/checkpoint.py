"""
AsyncMongoDBSaver — a LangGraph checkpointer that uses pymongo's native
AsyncMongoClient instead of wrapping sync calls in run_in_executor.

Subclasses MongoDBSaver so sync methods (put/get_tuple/list) still work
for any callers that need them. Only the async methods are overridden.
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
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from pymongo.asynchronous.mongo_client import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.operations import UpdateOne

logger = logging.getLogger("agent_sdk.checkpoint")


class AsyncMongoDBSaver(MongoDBSaver):
    """MongoDBSaver that uses pymongo AsyncMongoClient for all async operations.

    Sync methods (put, get_tuple, list) are inherited from MongoDBSaver as-is.
    Async methods (aput, aget_tuple, alist, aput_writes, adelete_thread) use
    the native async pymongo driver — no thread-pool wrapping.

    Usage::

        saver = AsyncMongoDBSaver.from_conn_string(
            "mongodb://localhost:27017", db_name="agent_research"
        )
        agent = BaseAgent(..., checkpointer=saver)
    """

    _async_checkpoints: AsyncCollection
    _async_writes: AsyncCollection

    def __init__(
        self,
        conn_string: str,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints",
        writes_collection_name: str = "checkpoint_writes",
        ttl: Optional[int] = None,
        serde=None,
    ) -> None:
        # Sync client passed to parent for inherited sync methods
        sync_client = MongoClient(conn_string)
        super().__init__(
            client=sync_client,
            db_name=db_name,
            checkpoint_collection_name=checkpoint_collection_name,
            writes_collection_name=writes_collection_name,
            ttl=ttl,
            serde=serde,
        )

        # Async client for all overridden async methods
        async_client = AsyncMongoClient(conn_string)
        async_db = async_client[db_name]
        self._async_checkpoints = async_db[checkpoint_collection_name]
        self._async_writes = async_db[writes_collection_name]
        self._conn_string = conn_string

        logger.info(
            "AsyncMongoDBSaver initialised — db='%s', checkpoints='%s', writes='%s'",
            db_name, checkpoint_collection_name, writes_collection_name,
        )

    # ------------------------------------------------------------------
    # Convenience constructor (mirrors MongoDBSaver.from_conn_string but
    # returns an instance directly instead of a context manager)
    # ------------------------------------------------------------------

    @classmethod
    def from_conn_string(  # type: ignore[override]
        cls,
        conn_string: str,
        db_name: str = "checkpointing_db",
        checkpoint_collection_name: str = "checkpoints",
        writes_collection_name: str = "checkpoint_writes",
        ttl: Optional[int] = None,
        serde=None,
    ) -> "AsyncMongoDBSaver":
        return cls(
            conn_string=conn_string,
            db_name=db_name,
            checkpoint_collection_name=checkpoint_collection_name,
            writes_collection_name=writes_collection_name,
            ttl=ttl,
            serde=serde,
        )

    # ------------------------------------------------------------------
    # Async overrides — native AsyncMongoClient, no run_in_executor
    # ------------------------------------------------------------------

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async checkpoint retrieval using native pymongo async driver."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        query: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if checkpoint_id:
            query["checkpoint_id"] = checkpoint_id

        doc = await self._async_checkpoints.find_one(
            query, sort=[("checkpoint_id", -1)]
        )
        if not doc:
            return None

        checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))

        write_docs = await self._async_writes.find(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": doc["checkpoint_id"],
            }
        ).sort("idx", 1).to_list(None)

        pending_writes = [
            (
                w["task_id"],
                w["channel"],
                self.serde.loads_typed((w["type"], w["value"])),
            )
            for w in write_docs
        ]

        parent_config = (
            {
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": doc["parent_checkpoint_id"],
                },
            }
            if doc.get("parent_checkpoint_id")
            else None
        )

        return CheckpointTuple(
            config={
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": doc["checkpoint_id"],
                },
            },
            checkpoint=checkpoint,
            metadata=doc.get("metadata", {}),
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async checkpoint listing using native pymongo async driver."""
        query: dict[str, Any] = {}
        if config:
            query["thread_id"] = config["configurable"]["thread_id"]
            query["checkpoint_ns"] = config["configurable"].get("checkpoint_ns", "")

        if filter:
            for key, value in filter.items():
                query[f"metadata.{key}"] = value

        if before:
            query["checkpoint_id"] = {
                "$lt": before["configurable"]["checkpoint_id"]
            }

        cursor = self._async_checkpoints.find(query).sort("checkpoint_id", -1)
        if limit:
            cursor = cursor.limit(limit)

        async for doc in cursor:
            checkpoint = self.serde.loads_typed((doc["type"], doc["checkpoint"]))
            parent_config = (
                {
                    **(config or {}),
                    "configurable": {
                        **(config or {}).get("configurable", {}),
                        "checkpoint_id": doc["parent_checkpoint_id"],
                    },
                }
                if doc.get("parent_checkpoint_id")
                else None
            )
            yield CheckpointTuple(
                config={
                    **(config or {}),
                    "configurable": {
                        **(config or {}).get("configurable", {}),
                        "checkpoint_id": doc["checkpoint_id"],
                    },
                },
                checkpoint=checkpoint,
                metadata=doc.get("metadata", {}),
                parent_config=parent_config,
                pending_writes=[],
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Async checkpoint persistence using native pymongo async driver."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        type_, serialized = self.serde.dumps_typed(checkpoint)
        doc: dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint["id"],
            "parent_checkpoint_id": config["configurable"].get("checkpoint_id"),
            "type": type_,
            "checkpoint": serialized,
            "metadata": metadata,
        }

        await self._async_checkpoints.update_one(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            },
            {"$set": doc},
            upsert=True,
        )

        return {
            **config,
            "configurable": {
                **config["configurable"],
                "checkpoint_id": checkpoint["id"],
            },
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async intermediate writes persistence using native pymongo async driver."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        operations = []
        for idx, (channel, value) in enumerate(writes):
            type_, serialized = self.serde.dumps_typed(value)
            doc: dict[str, Any] = {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
                "task_id": task_id,
                "task_path": task_path,
                "idx": idx,
                "channel": channel,
                "type": type_,
                "value": serialized,
            }
            operations.append(
                UpdateOne(
                    {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                        "task_id": task_id,
                        "idx": idx,
                    },
                    {"$set": doc},
                    upsert=True,
                )
            )

        if operations:
            await self._async_writes.bulk_write(operations)

    async def adelete_thread(self, thread_id: str) -> None:
        """Async thread deletion using native pymongo async driver."""
        await self._async_checkpoints.delete_many({"thread_id": thread_id})
        await self._async_writes.delete_many({"thread_id": thread_id})
