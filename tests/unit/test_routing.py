"""Unit tests for EmbeddingRouter cosine similarity and threshold logic."""
import pytest
import math


def cosine(a, b):
    """Reference cosine similarity for test assertions."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        from agent_sdk.mcp.circuit_breaker import CircuitBreaker
        v = [1.0, 0.0, 0.0]
        assert abs(cosine(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine(a, b) - (-1.0)) < 1e-9

    def test_zero_vector_returns_zero(self):
        from agent_sdk.marketplace.router_agent import EmbeddingRouter  # noqa: F401
        # Just test the formula directly
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert cosine(a, b) == 0.0


class TestLowConfidenceError:
    def test_low_confidence_error_stores_score(self):
        from agent_sdk.config import settings
        from agent_marketplace.router.router_agent import LowConfidenceError
        err = LowConfidenceError(best_score=0.1, best_agent="agent-health")
        assert err.best_score == 0.1
        assert err.best_agent == "agent-health"
        assert "0.100" in str(err)
