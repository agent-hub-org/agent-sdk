from typing import Annotated, Sequence, Literal, TypedDict
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph import add_messages


class AgentState(BaseModel):

    # LLM (bound or unbound model)
    llm: Any = Field(description="The LLM to use for the agent")

    # conversation messages
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # summarized history (for long-running autonomous behavior)
    summary: Optional[str] = None

    # tool registry available to the agent
    tools_by_name: Dict[str, Any] = Field(default_factory=dict, description="A dictionary of tools by name")

    # maximum allowed tokens before summarization
    max_context_tokens: int = 12000

    # enable automatic summarization
    enable_summarization: bool = True

    # number of recent messages to keep
    keep_last_n_messages: int = 6

    # summarization model
    summarizer_llm: Optional[Any] = None

    # autonomous agent configuration
    max_iterations: int = Field(
        default=20,
        description="Maximum number of reasoning/tool-use iterations before forcing a stop.",
    )
    iteration: int = Field(
        default=0,
        description="Current iteration count for the autonomous agent loop.",
    )