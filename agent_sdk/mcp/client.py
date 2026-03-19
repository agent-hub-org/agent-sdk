import logging
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("agent_sdk.mcp")


class MCPConnectionManager:
    """Thin wrapper around MultiServerMCPClient that manages lifecycle and returns LangChain tools."""

    def __init__(self):
        self._client: MultiServerMCPClient | None = None

    async def connect(self, server_configs: dict[str, dict[str, Any]]) -> list[StructuredTool]:
        """
        Connect to MCP servers and return their tools as LangChain StructuredTool objects.

        Args:
            server_configs: Mapping of server name to connection config.
                Each value is passed directly to MultiServerMCPClient, e.g.:
                {
                    "web-search": {"url": "http://localhost:8010/mcp", "transport": "streamable_http"},
                    "finance-data": {"url": "http://localhost:8011/mcp", "transport": "streamable_http"},
                }
        """
        if self._client is not None:
            logger.warning("Already connected — disconnecting first")
            await self.disconnect()

        logger.info("Connecting to %d MCP server(s): %s", len(server_configs), list(server_configs.keys()))

        self._client = MultiServerMCPClient(server_configs)
        tools = await self._client.get_tools()
        logger.info("Discovered %d tool(s) from MCP servers: %s",
                     len(tools), [t.name for t in tools])
        return tools

    async def disconnect(self):
        """Cleanly close the MCP client."""
        if self._client is not None:
            try:
                logger.info("Disconnected from MCP servers")
            except Exception:
                logger.exception("Error disconnecting from MCP servers")
            finally:
                self._client = None

    @property
    def connected(self) -> bool:
        return self._client is not None
