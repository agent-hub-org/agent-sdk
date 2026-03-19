import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger("agent_sdk.mcp")

MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds


class MCPConnectionManager:
    """Manages persistent MCP sessions so tools reuse connections instead of creating new ones per call."""

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._exit_stack: AsyncExitStack | None = None

    async def connect(self, server_configs: dict[str, dict[str, Any]]) -> list[BaseTool]:
        """
        Connect to MCP servers with persistent sessions and return LangChain tools.

        Unlike get_tools() which creates a new session per tool call, this opens
        long-lived sessions via client.session() and binds tools to them.
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
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.__aenter__()

                all_tools: list[BaseTool] = []
                for server_name in server_configs:
                    session = await self._exit_stack.enter_async_context(
                        self._client.session(server_name)
                    )
                    tools = await load_mcp_tools(session=session)
                    all_tools.extend(tools)
                    logger.info("Opened persistent session for '%s' — %d tool(s)",
                                server_name, len(tools))

                logger.info("Discovered %d tool(s) with persistent sessions: %s",
                             len(all_tools), [t.name for t in all_tools])
                return all_tools
            except Exception as e:
                await self._cleanup()
                if attempt < MAX_RETRIES:
                    logger.warning("MCP connection attempt %d/%d failed: %s — retrying in %ds",
                                   attempt, MAX_RETRIES, e, RETRY_DELAY)
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error("MCP connection failed after %d attempts: %s", MAX_RETRIES, e)
                    raise

    async def _cleanup(self):
        """Close all persistent sessions and reset state."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing MCP sessions")
            self._exit_stack = None
        self._client = None

    async def disconnect(self):
        """Cleanly close all persistent MCP sessions."""
        await self._cleanup()
        logger.info("Disconnected from MCP servers")

    @property
    def connected(self) -> bool:
        return self._client is not None
