"""
Centralised runtime configuration for the agent-sdk.

All values can be overridden via environment variables prefixed with AGENT_SDK_.
For example, setting AGENT_SDK_MAX_ITERATIONS=20 overrides the default of 10.

This eliminates hardcoded magic numbers scattered across the SDK and allows
operational tuning (e.g. circuit-breaker thresholds) without code changes.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSDKSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_SDK_", case_sensitive=False)

    # ── Conversation / context limits ──────────────────────────────────────────
    max_context_tokens: int = 32768
    keep_last_n_messages: int = 15
    max_iterations: int = 10
    tool_timeout: float = 120.0

    # ── Streaming ──────────────────────────────────────────────────────────────
    stream_timeout_seconds: float = 300.0

    # ── MCP / network ─────────────────────────────────────────────────────────
    mcp_max_retries: int = 5
    mcp_connect_timeout: float = 30.0
    mcp_sse_read_timeout: float = 300.0

    # ── Circuit breaker ────────────────────────────────────────────────────────
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 60.0

    # ── Memory (Mem0) ──────────────────────────────────────────────────────────
    memory_score_threshold: float = 0.70
    memory_max_results: int = 5
    memory_truncate_chars: int = 300

    # ── MongoDB persistence ────────────────────────────────────────────────────
    mongo_history_limit: int = 200
    # TTL for conversation documents — default 90 days
    mongo_ttl_seconds: int = 7_776_000

    # ── MongoDB checkpointer ───────────────────────────────────────────────────
    checkpoint_selection_timeout_ms: int = 5000
    checkpoint_socket_timeout_ms: int = 30000

    # ── Marketplace / routing ─────────────────────────────────────────────────
    min_routing_confidence: float = 0.25
    a2a_max_retries: int = 3

    # ── LLM defaults ────────────────────────────────────────────────────────────
    llm_temperature: float = 0.7
    llm_model: str = "gpt-5-nano"
    llm_timeout: float = 120.0
    llm_max_retries: int = 3

    # ── Financial pipeline confidence scoring ─────────────────────────────────
    confidence_penalty_per_warning: float = 1.0
    confidence_penalty_per_fallback: float = 1.5

    # ── Tool result summarisation ─────────────────────────────────────────────
    large_result_threshold: int = 8000


# Singleton — import this everywhere instead of hardcoding literals.
settings = AgentSDKSettings()
