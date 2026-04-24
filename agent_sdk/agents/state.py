import operator
from typing import Annotated, Any, Sequence, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

from agent_sdk.config import settings


def merge_dicts(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Simple dictionary merger for state fields.

    Note: nodes compute absolute values (e.g. phase_iterations reads the current
    count and returns count+1), so right-wins semantics is correct for counters.
    Concurrent writes to the same key will lose one update — this is acceptable
    because parallel fan-out writes to different phase keys.
    """
    if not left:
        return right
    if not right:
        return left
    return {**left, **right}


def max_int(left: int, right: int) -> int:
    """Reducer that keeps the maximum value."""
    return max(left, right)


def join_strings(left: str | None, right: str | None) -> str | None:
    """Reducer that joins strings with double newlines. Passing None clears the field.
    Passing a string prefixed with '__RESET__:' replaces the field entirely (used by
    background context compression to swap the full context for a compressed version)."""
    if right is None:
        return None  # explicit clear
    if isinstance(right, str) and right.startswith("__RESET__:"):
        return right[len("__RESET__:"):]  # full replacement, no append
    l = left or ""
    if l and right:
        return f"{l}\n\n{right}"
    return l or right or None


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
    iteration: Annotated[int, max_int] = Field(
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

    # --- Scratchpad (execution plan) + running context (work done) ---
    scratchpad: Annotated[Optional[str], join_strings] = Field(
        default=None,
        description=(
            "Execution plan written by the orchestrate node at the start of a request. "
            "Read by llm_call each ReAct loop iteration so the LLM always sees its plan. "
            "Cleared as a background task after the response is sent."
        ),
    )

    running_context: Annotated[Optional[str], join_strings] = Field(
        default=None,
        description=(
            "Structured summaries of completed work within a single request. "
            "Standard mode: tool results appended after each tool_node execution. "
            "Financial mode: each phase executor appends its findings as prose. "
            "The LLM sees this at every ReAct iteration so it never re-fetches done work. "
            "Cleared as a background task after the response is sent."
        ),
    )

    session_notepad: Annotated[Optional[Dict[str, Any]], merge_dicts] = Field(
        default=None,
        description=(
            "Structured key-value store persisted across requests within a session. "
            "Written via write_to_notepad tool. Never cleared by initialize(). "
            "Injected into llm_call so the agent remembers discoveries across follow-up messages."
        ),
    )

    # --- Response validation correction loop ---
    validation_hint: Optional[str] = Field(
        default=None,
        description=(
            "Correction instruction injected into the next llm_call when the previous "
            "response failed quality validation. Cleared after one use."
        ),
    )
    validation_retried: bool = Field(
        default=False,
        description="Ensures the correction loop fires at most once per request.",
    )


class ResearchState(AgentState):
    """
    Extended state for the research agent's enforced retrieve→download→retrieve workflow.

    The graph uses these fields to decide routing — they are NOT set by the LLM.
    """

    papers_sufficient: bool = Field(
        default=False,
        description="True when check_if_sufficient determines retrieved papers can answer the query.",
    )
    download_attempted: bool = Field(
        default=False,
        description="True after download_and_store_arxiv_papers has been called once.",
    )
    research_query: str = Field(
        default="",
        description="Cleaned academic query string extracted from the user message.",
    )
    retrieved_papers: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="Papers retrieved from the vector DB (accumulated across retrieve calls).",
    )


class FinancialAnalysisState(AgentState):
    """
    Extended state for the financial reasoning cognitive pipeline.

    Phases write prose summaries to running_context (inherited from AgentState).
    The synthesis node reads running_context to generate the final report.
    No per-phase structured dicts — all inter-phase context flows through running_context.

    This state is ONLY used when mode="financial_analyst" is set on BaseAgent.
    Standard agents continue using AgentState unchanged.
    """

    # --- Pipeline Control ---
    current_phase: str = Field(
        default="orchestrate",
        description="Current phase in the cognitive pipeline.",
    )

    phases_to_run: list[str] = Field(
        default_factory=list,
        description="Ordered list of phases to execute, determined by financial_orchestrate node.",
    )

    # Query type for synthesis routing (comparative needs different prompt)
    query_type: Optional[str] = Field(
        default=None,
        description="Query type: 'comparative', 'macro_impact', 'company_analysis', etc.",
    )

    # Entities extracted by orchestrate (used by comparative_analysis_node)
    entities: list[str] = Field(
        default_factory=list,
        description="Tickers, sectors, or other entities identified in the query.",
    )

    # --- Context Injection ---
    as_of_date: Optional[str] = Field(
        default=None,
        description="Historical reference date for the analysis (YYYY-MM-DD).",
    )

    # --- Execution trace (tool calls across all phases) ---
    tool_calls_log: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description=(
            "Flat log of every tool call made across all financial pipeline phases. "
            "Each entry: {action, phase, tool, args}. Accumulated with operator.add "
            "so each phase appends without overwriting prior phases."
        ),
    )

    entity_focus: Optional[str] = Field(
        default=None,
        description="When running comparative analysis, the single entity this branch analyzes.",
    )


class PhaseSubgraphState(BaseModel):
    """
    Internal state for a single financial phase subgraph invocation.

    This keeps phase-local buffers out of FinancialAnalysisState so parallel
    phase branches do not contend on transient keys in the parent graph.
    """

    messages: Sequence[BaseMessage] = Field(
        default_factory=list,
        description="Parent conversation messages used to seed the phase prompt.",
    )
    scratchpad: Optional[str] = Field(
        default=None,
        description="Parent execution plan used to extract the phase-specific step.",
    )
    running_context: Optional[str] = Field(
        default=None,
        description="Completed work from earlier phases, injected into the phase prompt.",
    )
    as_of_date: Optional[str] = Field(
        default=None,
        description="Historical reference date for the analysis (YYYY-MM-DD).",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional model override inherited from the parent state.",
    )
    tool_timeout: float = Field(
        default_factory=lambda: settings.tool_timeout,
        description="Timeout for executing a batch of tools within the phase.",
    )
    current_phase: str = Field(
        default="unknown",
        description="Logical phase name being executed by the subgraph.",
    )
    entity_focus: Optional[str] = Field(
        default=None,
        description="Specific entity assigned to this phase branch during comparative analysis.",
    )

    phase_messages: Annotated[Sequence[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Phase-local conversation thread excluding the dynamic system prompt.",
    )
    phase_system_text: str = Field(
        default="",
        description="Dynamic system prompt rebuilt as findings accumulate within the phase.",
    )
    phase_base_system_text: str = Field(
        default="",
        description="Original parent system prompt prefix reused when rebuilding the phase prompt.",
    )
    phase_iteration: int = Field(
        default=0,
        description="Number of LLM turns taken within the phase.",
    )
    phase_budget: int = Field(
        default=4,
        description="Maximum number of LLM turns allowed within the phase.",
    )
    phase_tools: list[str] = Field(
        default_factory=list,
        description="Tool names bound for this phase invocation.",
    )
    phase_plan: Optional[str] = Field(
        default=None,
        description="Phase-specific plan excerpt extracted from scratchpad.",
    )
    phase_findings: Annotated[Optional[str], join_strings] = Field(
        default=None,
        description="Accumulated findings gathered across tool calls and final prose.",
    )
    tool_calls_log: Annotated[list[dict], operator.add] = Field(
        default_factory=list,
        description="Tool execution log accumulated inside the phase subgraph.",
    )
