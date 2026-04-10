import logging
from typing import Any, Optional

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import Task
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger("agent_sdk.a2a.mongodb_task_store")


class AsyncMongoDBTaskStore(TaskStore):
    """
    A TaskStore implementation that uses MongoDB for persistence.
    Replaces the default InMemoryTaskStore to ensure A2A tasks persist across restarts.
    """

    def __init__(
        self,
        conn_string: str,
        db_name: str = "a2a_tasks",
        collection_name: str = "tasks",
    ):
        self._client = AsyncIOMotorClient(
            conn_string,
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=30000,
        )
        self._db = self._client[db_name]
        self._collection = self._db[collection_name]
        logger.info(
            "AsyncMongoDBTaskStore initialized — db='%s', collection='%s'",
            db_name,
            collection_name,
        )

    async def get(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> Task | None:
        """Retrieve a task from MongoDB."""
        doc = await self._collection.find_one({"task_id": task_id})
        if doc and (task_data := doc.get("task_data")):
            return Task.model_validate(task_data)
        return None

    async def save(
        self, task: Task, context: ServerCallContext | None = None
    ) -> None:
        """Save or update a task in MongoDB."""
        await self._collection.update_one(
            {"task_id": task.id},
            {"$set": {"task_id": task.id, "task_data": task.model_dump()}},
            upsert=True,
        )

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        """Remove a task from MongoDB."""
        await self._collection.delete_one({"task_id": task_id})

    async def close(self):
        """Close the MongoDB client connection."""
        self._client.close()
