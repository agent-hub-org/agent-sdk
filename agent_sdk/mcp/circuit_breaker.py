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

from agent_sdk.config import settings
from agent_sdk.metrics import circuit_breaker_open

logger = logging.getLogger("agent_sdk.circuit_breaker")


class _State(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """Lightweight circuit breaker for MCP tool calls.

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

    def __init__(
        self,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
    ) -> None:
        failure_threshold = failure_threshold if failure_threshold is not None else settings.circuit_breaker_failure_threshold
        recovery_timeout = recovery_timeout if recovery_timeout is not None else settings.circuit_breaker_recovery_timeout
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._failures = 0
        self._state = _State.CLOSED
        self._opened_at: float | None = None
        self._name: str = ""  # set on first record_failure call

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
        if self._name:
            circuit_breaker_open.labels(agent="sdk", tool_name=self._name).set(0)

    def record_failure(self, tool_name: str) -> None:
        """Call after a failed tool invocation."""
        if not self._name:
            self._name = tool_name
        self._failures += 1
        if self._state == _State.HALF_OPEN or self._failures >= self._threshold:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
            logger.warning(
                "Circuit breaker OPEN for tool '%s' after %d consecutive failure(s)",
                tool_name,
                self._failures,
            )
            circuit_breaker_open.labels(agent="sdk", tool_name=tool_name).set(1)
