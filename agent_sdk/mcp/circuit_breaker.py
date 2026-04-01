"""
Per-tool circuit breaker for MCP tool calls.

Transitions: CLOSED → OPEN (after failure_threshold failures) → HALF_OPEN
             (after recovery_timeout seconds) → CLOSED (on success) or OPEN (on failure).

When OPEN, callers receive an immediate error ToolMessage instead of making a
live tool call, preventing cascading failures when an MCP server is down.
"""

import logging
import time
from enum import Enum, auto

logger = logging.getLogger("agent_sdk.circuit_breaker")


class _State(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Lightweight, asyncio-safe circuit breaker (no external dependencies).

    Usage::

        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

        if breaker.is_open:
            # return an error message immediately
            ...
        try:
            result = await tool.ainvoke(args)
            breaker.record_success()
        except Exception:
            breaker.record_failure(tool_name)
            raise
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = 0
        self._state = _State.CLOSED
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        """True when the circuit should block calls.

        Automatically transitions OPEN → HALF_OPEN once the recovery window
        has elapsed so a single probe call is allowed through.
        """
        if self._state == _State.OPEN:
            if self._opened_at and (time.monotonic() - self._opened_at) >= self._recovery_timeout:
                self._state = _State.HALF_OPEN
                logger.info("Circuit breaker transitioning to HALF_OPEN (recovery window elapsed)")
        return self._state == _State.OPEN

    def record_success(self) -> None:
        """Call after a successful tool invocation."""
        if self._state != _State.CLOSED:
            logger.info("Circuit breaker CLOSED (recovered)")
        self._failures = 0
        self._state = _State.CLOSED

    def record_failure(self, tool_name: str) -> None:
        """Call after a failed tool invocation."""
        self._failures += 1
        if self._state == _State.HALF_OPEN or self._failures >= self._threshold:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker OPEN for tool '%s' after %d consecutive failure(s)",
                tool_name,
                self._failures,
            )
