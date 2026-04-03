"""
Lightweight response quality validator for the standard agent mode.

Checks for common failure modes:
- Empty or trivially short responses
- Hallucination markers (LLM "I don't have real-time data" when tools were available)
- Missing citations when tool results were used

Returns a list of issue strings (empty list = pass).
"""

import re
import logging

logger = logging.getLogger("agent_sdk.response_validator")

# Phrases that suggest the LLM forgot it has tools
_HALLUCINATION_MARKERS = [
    r"as an ai (language model|assistant)",
    r"i don'?t have (real.?time|current|live|up.?to.?date)",
    r"i (cannot|can'?t) (access|browse|search|fetch|retrieve)",
    r"my (knowledge|training) (cutoff|data) is",
    r"i (have no|lack) access to (current|real|live)",
]
_HALLUCINATION_RE = re.compile("|".join(_HALLUCINATION_MARKERS), re.IGNORECASE)

# Minimum non-trivial response length for financial queries
_MIN_RESPONSE_CHARS = 80


def validate_response(
    response_text: str,
    tool_calls_made: int = 0,
    require_citations: bool = False,
) -> list[str]:
    """
    Validate response quality. Returns a list of issue descriptions.
    Empty list means all checks passed.
    """
    issues = []

    if not response_text or not response_text.strip():
        issues.append("Response is empty.")
        return issues

    stripped = response_text.strip()

    if len(stripped) < _MIN_RESPONSE_CHARS:
        issues.append(
            f"Response is very short ({len(stripped)} chars). "
            "Consider asking for elaboration."
        )

    if _HALLUCINATION_RE.search(stripped):
        if tool_calls_made > 0:
            issues.append(
                "Response contains phrases suggesting the LLM forgot it has tool access "
                f"('{_HALLUCINATION_RE.search(stripped).group()}'), despite {tool_calls_made} "
                "tool call(s) being available."
            )

    if require_citations and tool_calls_made > 0:
        citation_count = len(re.findall(r'\[\d+\]', stripped))
        if citation_count == 0:
            issues.append(
                f"No citations found ([n] markers) despite {tool_calls_made} tool call(s). "
                "Response should reference data sources."
            )

    if issues:
        logger.warning("Response quality issues: %s", "; ".join(issues))

    return issues
