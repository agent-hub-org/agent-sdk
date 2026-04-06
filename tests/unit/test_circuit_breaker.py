"""Unit tests for CircuitBreaker state transitions."""
import time
import pytest
from agent_sdk.mcp.circuit_breaker import CircuitBreaker, _State


def make_breaker(threshold: int = 3, recovery_timeout: float = 60.0) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=threshold, recovery_timeout=recovery_timeout)


class TestCircuitBreakerTransitions:
    def test_starts_closed(self):
        cb = make_breaker()
        assert not cb.is_open
        assert cb._state == _State.CLOSED

    def test_opens_after_threshold_failures(self):
        cb = make_breaker(threshold=3)
        cb.record_failure("tool")
        cb.record_failure("tool")
        assert not cb.is_open  # still closed
        cb.record_failure("tool")
        assert cb.is_open  # now open

    def test_success_resets_failure_count(self):
        cb = make_breaker(threshold=3)
        cb.record_failure("tool")
        cb.record_failure("tool")
        cb.record_success()
        cb.record_failure("tool")  # count should restart from 1
        assert not cb.is_open

    def test_transitions_to_half_open_after_timeout(self):
        cb = make_breaker(threshold=1, recovery_timeout=0.01)
        cb.record_failure("tool")
        assert cb.is_open
        time.sleep(0.02)  # exceed recovery timeout
        # is_open auto-transitions to HALF_OPEN and returns False
        assert not cb.is_open
        assert cb._state == _State.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        cb = make_breaker(threshold=1, recovery_timeout=0.01)
        cb.record_failure("tool")
        time.sleep(0.02)
        cb.is_open  # trigger HALF_OPEN transition
        cb.record_success()
        assert cb._state == _State.CLOSED
        assert not cb.is_open

    def test_failure_in_half_open_reopens_circuit(self):
        cb = make_breaker(threshold=1, recovery_timeout=0.01)
        cb.record_failure("tool")
        time.sleep(0.02)
        cb.is_open  # trigger HALF_OPEN
        cb.record_failure("tool")
        assert cb._state == _State.OPEN
        assert cb.is_open
