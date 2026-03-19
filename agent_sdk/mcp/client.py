import asyncio
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger("agent_sdk.mcp")

MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


class MCPConnectionManager:
    """Thin wrapper around MultiServerMCPClient that manages lifecycle and returns LangChain tools."""

    def __init__(self):
        self._client: MultiServerMCPClient | None = None

    async def connect(self, server_configs: dict[str, dict[str, Any]]) -> list[StructuredTool]:
        """
        Connect to MCP servers and return their tools as LangChain StructuredTool objects.
        Retries on connection failure to handle service startup ordering.

        Args:
            server_configs: Mapping of server name to connection config.
                Each value is passed directly to MultiServerMCPClient, e.g.:
                {
                    "mcp-tool-servers": {"url": "http://localhost:8010/mcp", "transport": "streamable_http"},
                }
        """
        if self._client is not None:
            logger.warning("Already connected — disconnecting first")
            await self.disconnect()

        logger.info("Connecting to %d MCP server(s): %s", len(server_configs), list(server_configs.keys()))

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._client = MultiServerMCPClient(server_configs)
                tools = await self._client.get_tools()
                logger.info("Discovered %d tool(s) from MCP servers: %s",
                             len(tools), [t.name for t in tools])
                return tools
            except Exception as e:
                self._client = None
                if attempt < MAX_RETRIES:
                    logger.warning("MCP connection attempt %d/%d failed: %s — retrying in %ds",
                                   attempt, MAX_RETRIES, e, RETRY_DELAY)
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error("MCP connection failed after %d attempts: %s", MAX_RETRIES, e)
                    raise

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
