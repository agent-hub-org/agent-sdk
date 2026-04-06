# Changelog

All notable changes to `agent-sdk` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/):
- **MAJOR** — breaking changes to `AgentState`/`FinancialAnalysisState` schema, or public API removals
- **MINOR** — new features, new exports, new env-var knobs (backward-compatible)
- **PATCH** — bug fixes, internal refactors (backward-compatible)

> ⚠️ **MongoDB checkpoint compatibility**: `FinancialAnalysisState` schema changes must use `Optional` fields with defaults so existing checkpoints in `agent_financials.checkpoints` remain deserializable. Always note checkpoint-breaking changes explicitly.

---

## [Unreleased]

### Added
- `agent_sdk.agents.formatters` — `fix_flash_card_format` / `_fix_flash_card_format` centralized from all 5 agent repos
- `agent_sdk.server.streaming` — `StreamingMathFixer` (on-the-fly LaTeX delimiter fixer) and `_fix_math_delimiters` (post-processing variant) centralized from agent-research and agent-interview-prep
- `agent_sdk.a2a.factory` — `create_a2a_app(agent_card, executor_cls, mongo_db_name)` factory; all 5 agent `a2a_service/server.py` files now call this instead of repeating boilerplate
- `agent_sdk.metrics` — Prometheus metrics (histograms, gauges, counters) via isolated `CollectorRegistry`; `metrics_response()` helper for `/metrics` endpoints
- `agent_sdk.context` — `request_id_var` and `user_id_var` `ContextVar`s for threading request correlation IDs through async call stacks

### Changed
- `agent_sdk.logging.JsonFormatter` now injects `request_id` and `user_id` from context vars (when set) into every log record
- `agent_sdk.agents.nodes` — `llm_call_duration` and `tool_call_duration` histograms wired into `llm_call()` and `_execute_tool_calls()`; `raw_fallback_total` counter wired into `_build_phase_return()`
- `agent_sdk.mcp.circuit_breaker.CircuitBreaker` — `circuit_breaker_open` gauge updated on state transitions

---

## [v2.1.0] — 2026-04

### Added
- `agent_sdk.agents.nodes._build_phase_return()` — extracted shared fallback pattern from 5 financial phase nodes
- Pre-compiled module-level regexes (`_MALFORMED_TOOL_PATTERN`, `_JSON_FENCE_PATTERN`)

### Changed
- `agent_sdk.agents.nodes` — replaced all `getattr(state, field, default)` with direct Pydantic field access
- `agent_sdk.mcp.client` — `MCPSessionError` raised for session-termination errors (replaces fragile string matching)

---

## [v2.0.0] — 2026-04

### Added
- `agent_sdk.config.AgentSDKSettings` — pydantic-settings class; all 16+ hardcoded thresholds now configurable via `AGENT_SDK_*` env vars
- `agent_sdk.logging.configure_logging(service_name)` + `JsonFormatter` — centralized from all 5 agent `app.py` files
- `agent_sdk.mcp.exceptions.MCPSessionError`, `MCPToolError` — typed exceptions replacing bare `Exception` handling
- `pytest` + `pytest-asyncio` + `pytest-mock` dev dependencies; foundational unit test suite in `tests/unit/`

### Changed
- `AgentState` fields `max_context_tokens`, `keep_last_n_messages`, `max_iterations`, `tool_timeout` now default to `settings.*` values
- `CircuitBreaker.__init__` parameters default to `settings.circuit_breaker_*` values
- `mcp/client.py` `MAX_RETRIES` reads from `settings.mcp_max_retries`
- `database/mongo.py` history limit and TTL read from `settings.mongo_*`
- `database/memory.py` score threshold, max results, and truncation from `settings.memory_*`
- `checkpoint.py` MongoDB timeouts from `settings.checkpoint_*`
- `router/router_agent.py` routing confidence from `settings.min_routing_confidence`; unit vectors precomputed in `build_index()` for O(1) dot-product similarity
- `router/a2a_caller.py` max retries from `settings.a2a_max_retries`
