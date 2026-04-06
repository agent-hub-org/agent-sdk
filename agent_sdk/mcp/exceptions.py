"""
Typed exceptions for MCP client failures.

Using typed exceptions instead of string-matching on error messages makes
error handling robust against upstream library message changes.
"""


class MCPSessionError(Exception):
    """Raised when an MCP session terminates unexpectedly.

    Replace brittle ``if "session terminated" in str(e)`` checks with::

        except MCPSessionError:
            await manager.reconnect()
    """

    def __init__(self, message: str, original: Exception | None = None) -> None:
        super().__init__(message)
        self.original = original


class MCPToolError(Exception):
    """Raised when a tool call fails after circuit-breaker checks."""
