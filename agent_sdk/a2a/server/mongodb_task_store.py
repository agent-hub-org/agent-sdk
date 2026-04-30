import logging

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import Task
from a2a.types.a2a_pb2 import ListTasksRequest, ListTasksResponse
from google.protobuf.json_format import MessageToDict, ParseDict
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
        doc = await self._collection.find_one({"task_id": task_id})
        if doc and (task_data := doc.get("task_data")):
            return ParseDict(task_data, Task())
        return None

    async def save(
        self, task: Task, context: ServerCallContext | None = None
    ) -> None:
        await self._collection.update_one(
            {"task_id": task.id},
            {"$set": {"task_id": task.id, "task_data": MessageToDict(task)}},
            upsert=True,
        )

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        await self._collection.delete_one({"task_id": task_id})

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext | None = None,
    ) -> ListTasksResponse:
        query: dict = {}
        if params.context_id:
            query["task_data.contextId"] = params.context_id

        page_size = params.page_size or 50
        cursor = self._collection.find(query).limit(page_size)
        tasks = []
        async for doc in cursor:
            if task_data := doc.get("task_data"):
                tasks.append(ParseDict(task_data, Task()))

        return ListTasksResponse(tasks=tasks)

    async def close(self):
        self._client.close()
