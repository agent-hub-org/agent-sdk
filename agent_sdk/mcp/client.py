import asyncio
import logging
import random
from contextlib import AsyncExitStack
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_sdk.config import settings
from agent_sdk.mcp.exceptions import MCPSessionError

logger = logging.getLogger("agent_sdk.mcp")

# Keywords in exception messages that indicate a session-termination event.
# Using MCPSessionError wrapping is preferred; this set is the fallback for
# exceptions that originate inside the MCP library before we can intercept them.
_SESSION_TERMINATED_HINTS = frozenset({"session terminated", "session closed", "eof", "connection reset"})

_STREAMABLE_HTTP_TRANSPORTS = frozenset({"streamable_http", "streamable-http", "http"})


class MCPConnectionManager:
    """Manages persistent MCP sessions with automatic reconnection on failure."""

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._server_configs: dict[str, dict[str, Any]] | None = None
        self._tools: list[BaseTool] = []
        self._reconnect_lock = asyncio.Lock()

    async def connect(self, server_configs: dict[str, dict[str, Any]]) -> list[BaseTool]:
        """
        Connect to MCP servers with persistent sessions and return LangChain tools.

        Stores server_configs so sessions can be re-established on failure.
        """
        if self._client is not None:
            logger.warning("Already connected — disconnecting first")
            await self.disconnect()

        self._server_configs = server_configs
        return await self._establish_sessions()

    async def _establish_sessions(self) -> list[BaseTool]:
        """Create persistent sessions for all configured MCP servers."""
        server_configs = self._server_configs
        logger.info("Connecting to %d MCP server(s): %s", len(server_configs), list(server_configs.keys()))

        max_retries = settings.mcp_max_retries
        for attempt in range(1, max_retries + 1):
            try:
                self._client = MultiServerMCPClient(server_configs)
                self._exit_stack = AsyncExitStack()
                await self._exit_stack.__aenter__()

                all_tools: list[BaseTool] = []
                for server_name, config in server_configs.items():
                    transport = config.get("transport", "")

                    if transport in _STREAMABLE_HTTP_TRANSPORTS:
                        # Use the new streamable_http_client API directly
                        # to avoid the deprecated streamablehttp_client in
                        # langchain-mcp-adapters.
                        session = await self._create_streamable_http_session(config)
                    else:
                        # Use langchain-mcp-adapters for stdio/sse/websocket
                        session = await self._exit_stack.enter_async_context(
                            self._client.session(server_name)
                        )

                    tools = await load_mcp_tools(session=session)
                    all_tools.extend(tools)
                    logger.info("Opened persistent session for '%s' — %d tool(s)",
                                server_name, len(tools))

                logger.info("Discovered %d tool(s) with persistent sessions: %s",
                             len(all_tools), [t.name for t in all_tools])
                self._tools = all_tools
                return all_tools
            except Exception as e:
                await self._cleanup()
                if attempt < max_retries:
                    delay = min(30.0, (2 ** (attempt - 1)) + random.uniform(0, 1))
                    logger.warning("MCP connection attempt %d/%d failed: %s — retrying in %.2fs",
                                   attempt, max_retries, e, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("MCP connection failed after %d attempts: %s", max_retries, e)
                    raise

    async def _create_streamable_http_session(self, config: dict[str, Any]) -> ClientSession:
        """Create a session using the non-deprecated streamable_http_client API."""
        http_client = await self._exit_stack.enter_async_context(
            httpx.AsyncClient(
                headers=config.get("headers"),
                auth=config.get("auth"),
                timeout=httpx.Timeout(
                    config.get("timeout", 30),
                    read=config.get("sse_read_timeout", 300),
                ),
            )
        )
        read, write, _ = await self._exit_stack.enter_async_context(
            streamable_http_client(
                config["url"],
                http_client=http_client,
                terminate_on_close=config.get("terminate_on_close", True),
            )
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        return session

    async def reconnect(self) -> list[BaseTool]:
        """Re-establish MCP sessions after a failure. Thread-safe via lock."""
        async with self._reconnect_lock:
            if self._server_configs is None:
                raise RuntimeError("Cannot reconnect — no server configs stored. Call connect() first.")

            logger.warning("Reconnecting MCP sessions...")
            await self._cleanup()
            return await self._establish_sessions()

    async def _cleanup(self):
        """Close all persistent sessions and reset state."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing MCP sessions")
            self._exit_stack = None
        self._client = None

    @staticmethod
    def _is_session_error(exc: Exception) -> bool:
        """Return True if *exc* looks like an MCP session-termination error."""
        msg = str(exc).lower()
        return any(hint in msg for hint in _SESSION_TERMINATED_HINTS)

    async def disconnect(self):
        """Cleanly close all persistent MCP sessions."""
        await self._cleanup()
        self._server_configs = None
        self._tools = []
        logger.info("Disconnected from MCP servers")

    @property
    def connected(self) -> bool:
        return self._client is not None
