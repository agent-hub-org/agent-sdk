from __future__ import annotations

from datetime import datetime, UTC
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class SubAgentInput(BaseModel):
    query: str
    entities: list[str] = []
    workspace_context: str = ""   # plain-text rendering of reads_from keys — NOT the full workspace
    user_profile: dict[str, Any] = {}


class SubAgentOutput(BaseModel):
    agent_name: str
    findings: str
    structured: dict[str, Any] = {}
    confidence: float = 1.0
    cached: bool = False
    computed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SubAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_input: SubAgentInput
    findings: str
    tool_calls_count: int
