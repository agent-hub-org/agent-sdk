import json
from dataclasses import dataclass, field
from enum import Enum


class ErrorCode(str, Enum):
    LLM_FAILURE = "llm_failure"
    TOOL_FAILURE = "tool_failure"
    MCP_UNAVAILABLE = "mcp_unavailable"
    CHECKPOINT_FAILURE = "checkpoint_failure"
    TIMEOUT = "timeout"
    INTERNAL = "internal_error"


@dataclass
class AgentError(Exception):
    error_code: ErrorCode
    message: str
    request_id: str | None = field(default=None)

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> dict:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "request_id": self.request_id,
        }

    def to_sse_error(self) -> str:
        """Serialize for embedding in __ERROR__: SSE lines."""
        return json.dumps(self.to_dict())
