import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient
from agent_sdk.config import settings

logger = logging.getLogger("agent_sdk.database.mongo")
_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

class BaseMongoDatabase:
    """
    Base generic MongoDB class for storing agent conversations.
    Agents should subclass this and implement db_name() to define their specific database.
    """
    _client: AsyncIOMotorClient | None = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        if cls._client is None:
            logger.info("Connecting to MongoDB at %s", _MONGO_URI.split("@")[-1] if "@" in _MONGO_URI else _MONGO_URI.split("//")[-1].split(":")[0])
            cls._client = AsyncIOMotorClient(_MONGO_URI)
        return cls._client

    @classmethod
    def db_name(cls) -> str:
        raise NotImplementedError("Subclasses must implement db_name()")

    @classmethod
    def _conversations(cls):
        return cls.get_client()[cls.db_name()]["conversations"]

    @classmethod
    def generate_session_id(cls) -> str:
        return uuid.uuid4().hex

    @classmethod
    async def save_conversation(
        cls,
        session_id: str,
        query: str,
        response: str,
        steps: list[dict] | None = None,
        user_id: str | None = None,
    ) -> str:
        doc = {
            "session_id": session_id,
            "query": query,
            "response": response,
            "steps": steps or [],
            "tools_used": list({s["tool"] for s in (steps or []) if s.get("action") == "tool_call"}),
            "total_tool_calls": sum(1 for s in (steps or []) if s.get("action") == "tool_call"),
            "created_at": datetime.now(timezone.utc),
        }
        if user_id:
            doc["user_id"] = user_id
        result = await cls._conversations().insert_one(doc)
        logger.info(
            "Saved conversation — session='%s', user='%s', doc_id='%s'",
            session_id, user_id or "anonymous", result.inserted_id,
        )
        return str(result.inserted_id)

    @classmethod
    async def get_history(cls, session_id: str) -> list[dict]:
        cursor = cls._conversations().find(
            {"session_id": session_id},
            {"_id": 0, "query": 1, "response": 1, "created_at": 1},
        ).sort("created_at", 1)
        return await cursor.to_list(length=100)

    @classmethod
    async def get_history_by_user(cls, user_id: str) -> list[dict]:
        cursor = cls._conversations().find(
            {"user_id": user_id},
            {"_id": 0, "query": 1, "response": 1, "created_at": 1, "session_id": 1},
        ).sort("created_at", -1)
        return await cursor.to_list(length=settings.mongo_history_limit)

    @classmethod
    async def get_history_by_sessions(cls, session_ids: list[str]) -> list[dict]:
        if not session_ids:
            return []
        cursor = cls._conversations().find(
            {"session_id": {"$in": session_ids}},
            {"_id": 0, "query": 1, "response": 1, "created_at": 1, "session_id": 1},
        ).sort("created_at", -1)
        return await cursor.to_list(length=settings.mongo_history_limit)

    @classmethod
    async def ensure_indexes(cls) -> None:
        db = cls.get_client()[cls.db_name()]
        await db["conversations"].create_index("created_at", expireAfterSeconds=settings.mongo_ttl_seconds)
        await db["conversations"].create_index([("user_id", 1), ("created_at", -1)])
        await db["conversations"].create_index([("session_id", 1), ("created_at", -1)])
        logger.info("MongoDB indexes ensured for conversations")

    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None

class MongoManager:
    """
    Standardized MongoDB client manager for agents.
    Handles connection pooling and basic CRUD operations.
    """
    def __init__(self, uri: str = None, db_name: str = None):
        self.uri = uri or _MONGO_URI
        self.db_name = db_name
        self._client: AsyncIOMotorClient | None = None

    async def get_client(self) -> AsyncIOMotorClient:
        if self._client is None:
            logger.info("Initializing Mongo client at %s", self.uri)
            self._client = AsyncIOMotorClient(
                self.uri,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=30000,
            )
        return self._client

    async def get_collection(self, collection_name: str, db_override: str = None) -> Any:
        client = await self.get_client()
        db_name = db_override or self.db_name or "agent_db"
        return client[db_name][collection_name]

    async def close(self):
        if self._client:
            self._client.close()
            self._client = None
