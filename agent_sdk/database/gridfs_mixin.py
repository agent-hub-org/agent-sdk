"""GridFS file storage mixin — shared by agent-financials and agent-interview-prep."""
import logging
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorGridFSBucket

logger = logging.getLogger("agent_sdk.database.gridfs_mixin")


class GridFSMixin:
    """Add GridFS file store/retrieve to any BaseMongoDatabase subclass.

    Requires the subclass to expose:
    - cls._db() -> AsyncIOMotorDatabase
    - cls._files() -> AsyncIOMotorCollection  (a 'files' metadata collection)
    """

    _gridfs: AsyncIOMotorGridFSBucket | None = None

    @classmethod
    def _gridfs_bucket(cls) -> AsyncIOMotorGridFSBucket:
        if cls._gridfs is None:
            cls._gridfs = AsyncIOMotorGridFSBucket(cls._db())
        return cls._gridfs

    @classmethod
    async def store_file(
        cls,
        file_id: str,
        filename: str,
        data: bytes,
        file_type: str,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        bucket = cls._gridfs_bucket()
        await bucket.upload_from_stream(
            file_id,
            data,
            metadata={
                "file_id": file_id,
                "original_filename": filename,
                "file_type": file_type,
                "session_id": session_id,
                "user_id": user_id,
            },
        )
        await cls._files().insert_one({
            "file_id": file_id,
            "filename": filename,
            "file_type": file_type,
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc),
        })
        logger.info("Stored file file_id='%s' filename='%s' type='%s'", file_id, filename, file_type)

    @classmethod
    async def retrieve_file(cls, file_id: str) -> tuple[bytes, dict] | None:
        bucket = cls._gridfs_bucket()
        try:
            stream = await bucket.open_download_stream_by_name(file_id)
            data = await stream.read()
            meta = await cls._files().find_one({"file_id": file_id}, {"_id": 0})
            return data, meta or {}
        except Exception:
            logger.warning("File not found in GridFS: file_id='%s'", file_id)
            return None
