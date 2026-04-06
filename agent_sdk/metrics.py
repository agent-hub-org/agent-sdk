"""
Prometheus metrics for agent services.

All metrics share a single custom CollectorRegistry so they don't collide
with prometheus_client's default registry (which is used by some libraries
automatically).

Usage in app.py::

    from agent_sdk.metrics import metrics_response
    from fastapi.responses import Response

    @app.get("/metrics")
    async def metrics():
        content, content_type = metrics_response()
        return Response(content=content, media_type=content_type)

Usage in instrumented code::

    from agent_sdk.metrics import llm_call_duration
    import time

    t0 = time.monotonic()
    result = await llm.ainvoke(messages)
    llm_call_duration.labels(agent=AGENT_NAME, model=model, phase=phase).observe(time.monotonic() - t0)
"""

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)

# Isolated registry — avoids conflicts with default prometheus_client registry
REGISTRY = CollectorRegistry(auto_describe=True)

llm_call_duration = Histogram(
    "agent_llm_call_duration_seconds",
    "LLM call end-to-end latency in seconds",
    ["agent", "model", "phase"],
    registry=REGISTRY,
)

tool_call_duration = Histogram(
    "agent_tool_call_duration_seconds",
    "MCP tool call latency in seconds",
    ["agent", "tool_name"],
    registry=REGISTRY,
)

circuit_breaker_open = Gauge(
    "agent_circuit_breaker_open",
    "1 if the named circuit breaker is OPEN (blocking calls), 0 otherwise",
    ["agent", "tool_name"],
    registry=REGISTRY,
)

stream_bytes_total = Counter(
    "agent_stream_bytes_total",
    "Cumulative bytes yielded over SSE streaming responses",
    ["agent"],
    registry=REGISTRY,
)

raw_fallback_total = Counter(
    "agent_financial_raw_fallback_total",
    "Number of financial pipeline phases that fell back to raw_analysis (LLM returned unparseable JSON)",
    ["phase"],
    registry=REGISTRY,
)


def metrics_response() -> tuple[bytes, str]:
    """Return (body_bytes, content_type) ready for a /metrics HTTP response."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
