import logging
from typing import Any, Callable, Awaitable
from functools import wraps

logger = logging.getLogger("agent_sdk.mcp_utils")

def mcp_tool_handler(func: Callable[..., Awaitable[Any]]):
    """
    Decorator for MCP tools to standardize logging and error handling.
    Catches exceptions and returns them as a string to prevent server crash.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        tool_name = func.__name__
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error("MCP Tool '%s' failed: %s", tool_name, str(e), exc_info=True)
            return f"Error executing tool {tool_name}: {str(e)}"
    return wrapper
