import asyncio
import itertools
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
    """Manages persistent MCP sessions with automatic reconnection on failure.

    Per-server independent retry: if one server fails to connect, tools from the
    other servers are still available.  The failed servers are tracked so that
    reconnect() only retries those, not already-healthy servers.
    """

    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._server_configs: dict[str, dict[str, Any]] | None = None
        self._tools: list[BaseTool] = []
        self._failed_servers: set[str] = set()
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
        """Create persistent sessions for all configured MCP servers in parallel.

        Per-server independent retry: each server is attempted independently.
        A failure on one server does not block tools from healthy servers from being
        returned.  Failed servers are recorded in self._failed_servers.
        """
        server_configs = self._server_configs
        logger.info("Connecting to %d MCP server(s) in parallel: %s", len(server_configs), list(server_configs.keys()))

        self._client = MultiServerMCPClient(server_configs)
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()
        self._failed_servers = set()

        max_retries = settings.mcp_max_retries

        async def _connect_one_server_with_retry(server_name: str, config: dict[str, Any]) -> list[BaseTool]:
            """Connect to a single MCP server with per-server retries."""
            for attempt in range(1, max_retries + 1):
                try:
                    transport = config.get("transport", "")
                    if transport in _STREAMABLE_HTTP_TRANSPORTS:
                        session = await self._create_streamable_http_session(config)
                    else:
                        session = await self._exit_stack.enter_async_context(
                            self._client.session(server_name)
                        )
                    tools = await load_mcp_tools(session=session)
                    logger.info("Opened persistent session for '%s' — %d tool(s)", server_name, len(tools))
                    return tools
                except Exception as e:
                    if attempt < max_retries:
                        delay = min(30.0, (2 ** (attempt - 1)) + random.uniform(0, 1))
                        logger.warning(
                            "MCP server '%s' attempt %d/%d failed: %s — retrying in %.2fs",
                            server_name, attempt, max_retries, e, delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "MCP server '%s' failed after %d attempts — running without it: %s",
                            server_name, max_retries, e,
                        )
                        self._failed_servers.add(server_name)
                        return []  # Partial degradation: return empty list, not exception

        results = await asyncio.gather(
            *[_connect_one_server_with_retry(name, cfg) for name, cfg in server_configs.items()],
            return_exceptions=False,  # exceptions already handled per-server above
        )
        all_tools: list[BaseTool] = list(itertools.chain.from_iterable(r for r in results if r))

        if self._failed_servers:
            logger.warning("Running in PARTIAL-DEGRADED mode — failed servers: %s", self._failed_servers)
        logger.info("Discovered %d tool(s) with persistent sessions: %s",
                     len(all_tools), [t.name for t in all_tools])
        self._tools = all_tools
        return all_tools

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
        """Re-establish MCP sessions after a failure. Thread-safe via lock.

        Reconnects ALL servers (not just failed ones), since a session error
        may have invalidated previously healthy sessions too.
        """
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
