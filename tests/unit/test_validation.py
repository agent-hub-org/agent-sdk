import pytest
from agent_sdk.utils.validation import SAFE_SESSION_RE


@pytest.mark.parametrize("session_id", [
    "abc123",
    "a" * 64,
    "550e8400-e29b-41d4-a716-446655440000",
    "ABC-xyz-123",
])
def test_valid_session_ids(session_id):
    assert SAFE_SESSION_RE.match(session_id), f"Expected '{session_id}' to match"


@pytest.mark.parametrize("session_id", [
    "",
    "a" * 65,
    "has space",
    "has/slash",
    "has_underscore",
    "has.dot",
    "has@symbol",
])
def test_invalid_session_ids(session_id):
    assert not SAFE_SESSION_RE.match(session_id), f"Expected '{session_id}' NOT to match"
