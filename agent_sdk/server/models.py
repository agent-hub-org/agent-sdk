"""Shared Pydantic request/response models used across agent API apps."""
from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    session_id: str | None = None
    response_format: str | None = None
    model_id: str | None = None


class AskResponse(BaseModel):
    session_id: str
    query: str
    response: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


class SessionsHistoryRequest(BaseModel):
    session_ids: list[str]
