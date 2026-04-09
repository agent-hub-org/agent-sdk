import json
import logging
from typing import Any, Optional

logger = logging.getLogger("agent_sdk.utils.output")

def unwrap_structured_response(content: Any, primary_key: str = "full_report") -> str:
    """
    Extracts a string from a potentially JSON-wrapped response.
    If the content is a JSON object containing the primary_key, it returns that value.
    Otherwise, it returns the original content as a string.
    """
    if not content:
        return ""

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and primary_key in parsed:
                return str(parsed[primary_key])
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(content, dict):
        return str(content.get(primary_key, content))

    return str(content)
