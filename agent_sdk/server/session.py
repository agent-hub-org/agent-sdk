"""Session ownership verification shared across agents."""
from fastapi import HTTPException, status
from agent_sdk.database.mongo import BaseMongoDatabase


async def verify_session_ownership(
    session_id: str,
    user_id: str | None,
    mongo: type[BaseMongoDatabase],
) -> None:
    """Raise HTTP 403 if session_id belongs to a different user.

    Allows access when:
    - user_id is None (anonymous request — no ownership enforced)
    - session has no conversations at all (brand-new session)
    - session has conversations owned by this user_id
    """
    if not user_id:
        return
    owned = await mongo.get_history(session_id, user_id=user_id)
    if not owned:
        any_history = await mongo.get_history(session_id)
        if any_history:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied",
            )
