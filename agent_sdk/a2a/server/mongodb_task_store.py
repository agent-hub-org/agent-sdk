import logging
from typing import Any, Optional

from a2a.server.tasks import TaskStore
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

    async def get(self, task_id: str) -> Optional[Any]:
        """Retrieve a task from MongoDB."""
        doc = await self._collection.find_one({"task_id": task_id})
        if doc:
            # Note: We assume the task object is stored in 'task_data'
            # Depending on how a2a-sdk expects the object (instance vs dict),
            # we might need to reconstruct the object here.
            # However, since we don't have the class info, we store/return as is.
            # If a2a-sdk expects a specific type, this might need an adapter.
            return doc.get("task_data")
        return None

    async def save(self, task_id: str, task: Any) -> None:
        """Save or update a task in MongoDB."""
        # Handle Pydantic models if present
        task_data = task
        if hasattr(task, "model_dump"):
            task_data = task.model_dump()
        elif hasattr(task, "dict"):
            task_data = task.dict()

        await self._collection.update_one(
            {"task_id": task_id},
            {"$set": {"task_id": task_id, "task_data": task_data}},
            upsert=True,
        )

    async def delete(self, task_id: str) -> None:
        """Remove a task from MongoDB."""
        await self._collection.delete_one({"task_id": task_id})

    async def close(self):
        """Close the MongoDB client connection."""
        self._client.close()
