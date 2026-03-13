from typing import Annotated, Sequence, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(BaseModel):

    # conversation messages (reducer handles append + dedup)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # summarized history (for long-running autonomous behavior)
    summary: Optional[str] = None

    # system prompt defined once by the agent repository
    system_prompt: Optional[str] = None

    # maximum allowed tokens before summarization
    max_context_tokens: int = 12000

    # enable automatic summarization
    enable_summarization: bool = True

    # number of recent messages to keep
    keep_last_n_messages: int = 6

    # autonomous agent configuration
    max_iterations: int = Field(
        default=20,
        description="Maximum number of reasoning/tool-use iterations before forcing a stop.",
    )
    iteration: int = Field(
        default=0,
        description="Current iteration count for the autonomous agent loop.",
    )