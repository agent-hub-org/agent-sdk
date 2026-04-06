from typing import Annotated, Any, Sequence, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from agent_sdk.config import settings


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Simple dictionary merger for state fields."""
    return {**left, **right}


class AgentState(BaseModel):

    # conversation messages (reducer handles append + dedup)
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # summarized history (for long-running autonomous behavior)
    summary: Optional[str] = None

    # system prompt defined once by the agent repository
    system_prompt: Optional[str] = None

    # maximum allowed tokens before summarization
    max_context_tokens: int = Field(default_factory=lambda: settings.max_context_tokens)

    # enable automatic summarization
    enable_summarization: bool = True

    # number of recent messages to keep
    keep_last_n_messages: int = Field(default_factory=lambda: settings.keep_last_n_messages)

    # dynamic model override — if set, llm_call uses this model instead of agent.llm
    model_id: Optional[str] = None

    # autonomous agent configuration
    max_iterations: int = Field(
        default_factory=lambda: settings.max_iterations,
        description="Maximum number of reasoning/tool-use iterations before forcing a stop.",
    )
    iteration: int = Field(
        default=0,
        description="Current iteration count for the autonomous agent loop.",
    )
    tool_timeout: float = Field(
        default_factory=lambda: settings.tool_timeout,
        description="Maximum seconds to wait for a single batch of tool calls before timing out.",
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

    raw_fallback_count: int = Field(
        default=0,
        description="Number of phases that fell back to raw_analysis due to JSON extraction failure.",
    )

    overall_confidence: Optional[float] = Field(
        default=None,
        description="Calculated confidence score based on validation warnings and fallbacks.",
    )

    # --- Iteration Budget Control ---
    phase_iterations: Annotated[dict[str, int], merge_dicts] = Field(
        default_factory=dict,
        description="Current iteration count for each phase.",
    )

    phase_iteration_budgets: dict[str, int] = Field(
        default_factory=lambda: {
            "query_classification": 1,
            "regime_assessment": 2,
            "causal_analysis": 2,
            "sector_analysis": 2,
            "company_analysis": 8,
            "risk_assessment": 2,
            "synthesis": 3,
        },
        description="Per-phase iteration budgets for the financial cognitive pipeline. Total: 20 iterations.",
    )