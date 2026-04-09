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

    # --- User identity (for memory system) ---
    user_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable user identifier passed by the caller. Used to scope perspective memory "
            "across sessions. If None, memory writing is skipped."
        ),
    )
    session_id: str = Field(
        default="default",
        description="Session identifier — mirrors the LangGraph thread_id. Set by arun().",
    )
    perspective_context: Optional[str] = Field(
        default=None,
        description=(
            "User personality background loaded by load_user_context. "
            "Injected into llm_call for communication style adaptation ONLY — "
            "never affects analytical content or planning."
        ),
    )

    # --- Structured planning / scratchpad ---
    enable_analytical_path: bool = Field(
        default=True,
        description=(
            "When False, triager_router always routes to llm_call, bypassing the parallel planner "
            "and synthesizer. Scratchpad is still captured in tool_node. "
            "Set to False on BaseAgent(analytical_path=False) for lower-latency deployments."
        ),
    )
    query_type: Optional[str] = Field(
        default=None,
        description="'opaque' (direct answer) or 'analytical' (multi-step plan). Set by triager.",
    )
    execution_plan: Optional[list[list[dict]]] = Field(
        default=None,
        description=(
            "Ordered list of parallel batches produced by parallel_planner. "
            "Each batch is a list of {'tool': name, 'args': {...}} dicts that run concurrently."
        ),
    )
    current_batch_index: int = Field(
        default=0,
        description="Index into execution_plan pointing to the next batch to execute.",
    )
    scratchpad: Optional[str] = Field(
        default=None,
        description=(
            "Accumulated tool results. Written by stateless_executor (analytical path) and "
            "tool_node (opaque path). Read by synthesizer and memory_writer."
        ),
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

    # --- Analysis Findings ---
    findings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Map of phase name to its structured findings (e.g., {'regime_assessment': {...}}).",
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

    # --- Context Injection ---
    as_of_date: Optional[str] = Field(
        default=None,
        description="Historical reference date for the analysis (YYYY-MM-DD).",
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
            "company_analysis": 4,
            "risk_assessment": 2,
            "synthesis": 3,
        },
        description="Per-phase iteration budgets for the financial cognitive pipeline. Total: 20 iterations.",
    )

    # --- Per-phase planning / execution scratchpad ---
    phase_tool_plan: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Tool calls planned for the current financial phase (reset by each phase planner). "
            "Format: [{'tool': name, 'args': {...}}, ...]"
        ),
    )
    phase_scratchpad: Optional[str] = Field(
        default=None,
        description="Raw tool results for the current financial phase (reset by each phase planner).",
    )