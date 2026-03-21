from typing import Annotated, Any, Sequence, Optional
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
    max_context_tokens: int = 32768

    # enable automatic summarization
    enable_summarization: bool = True

    # number of recent messages to keep
    keep_last_n_messages: int = 15

    # dynamic model override — if set, llm_call uses this model instead of agent.llm
    model_id: Optional[str] = None

    # autonomous agent configuration
    max_iterations: int = Field(
        default=20,
        description="Maximum number of reasoning/tool-use iterations before forcing a stop.",
    )
    iteration: int = Field(
        default=0,
        description="Current iteration count for the autonomous agent loop.",
    )


class FinancialAnalysisState(AgentState):
    """
    Extended state for the financial reasoning cognitive pipeline.

    Adds typed fields for structured findings from each reasoning phase.
    Each phase reads prior phases' findings and writes its own, creating
    a structured analytical thread that persists across tool calls.

    This state is ONLY used when mode="financial_analyst" is set on BaseAgent.
    Standard agents continue using AgentState unchanged.
    """

    # --- Query Classification ---
    query_classification: Optional[dict[str, Any]] = Field(
        default=None,
        description="Output of the query classifier — determines which pipeline phases to activate.",
    )

    # --- Phase Outputs (structured dicts from Pydantic schema .model_dump()) ---
    regime_context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Structured regime assessment: monetary/market regime, cycle position, key indicators.",
    )

    causal_analysis: Optional[dict[str, Any]] = Field(
        default=None,
        description="Causal chain analysis: transmission mechanisms, affected sectors/companies.",
    )

    sector_findings: Optional[dict[str, Any]] = Field(
        default=None,
        description="Sector-level analysis: valuation, growth, rotation signals.",
    )

    company_analysis: Optional[dict[str, Any]] = Field(
        default=None,
        description="Company-level fundamental analysis: valuation, quality, DCF, peer comparison.",
    )

    risk_assessment: Optional[dict[str, Any]] = Field(
        default=None,
        description="Risk assessment: scenarios, stress tests, validation warnings.",
    )

    synthesis_report: Optional[dict[str, Any]] = Field(
        default=None,
        description="Final synthesized report combining all phases.",
    )

    # --- Pipeline Control ---
    current_phase: str = Field(
        default="query_classification",
        description="Current phase in the cognitive pipeline.",
    )

    phases_to_run: list[str] = Field(
        default_factory=list,
        description="Ordered list of phases to execute, determined by query classifier.",
    )

    validation_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings from symbolic validators accumulated across phases.",
    )