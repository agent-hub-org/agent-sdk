import asyncio
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .graph import create_graph
from .state import AgentState
from ..llm_services.agent_llm import initialize_azure as initialize_agent_azure
from ..llm_services.summarizer_llm import initialize_azure as initialize_summarizer_azure
from ..mcp.circuit_breaker import CircuitBreaker


class _TTLDict:
    """Minimal time-based eviction dict. Items expire after ttl seconds of inactivity.

    Used for _session_notepads to prevent unbounded memory growth in long-running
    agents with many distinct sessions.
    """

    def __init__(self, ttl: float = 3600.0, maxsize: int = 500) -> None:
        self._ttl = ttl
        self._maxsize = maxsize
        self._data: dict[str, tuple[Any, float]] = {}  # key → (value, last_access)

    def _evict(self) -> None:
        now = time.monotonic()
        stale = [k for k, (_, ts) in self._data.items() if (now - ts) > self._ttl]
        for k in stale:
            del self._data[k]
        # Hard cap: evict oldest entries if still over limit
        if len(self._data) > self._maxsize:
            by_age = sorted(self._data.items(), key=lambda x: x[1][1])
            for k, _ in by_age[:len(self._data) - self._maxsize]:
                del self._data[k]

    def get(self, key: str, default: Any = None) -> Any:
        entry = self._data.get(key)
        if entry is None:
            return default
        value, _ = entry
        self._data[key] = (value, time.monotonic())
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getitem__(self, key: str) -> Any:
        _MISSING = object()
        entry = self._data.get(key, _MISSING)
        if entry is _MISSING:
            raise KeyError(key)
        value, _ = entry
        self._data[key] = (value, time.monotonic())
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = (value, time.monotonic())
        if len(self._data) > self._maxsize * 1.2:
            self._evict()

    def setdefault(self, key: str, default: Any) -> Any:
        if key not in self._data:
            self[key] = default
        return self.get(key)

    def __delitem__(self, key: str) -> None:
        del self._data[key]

logger = logging.getLogger("agent_sdk.agent")

# Nodes that produce user-facing LLM output and should be streamed to the client.
DEFAULT_STREAMING_NODES = frozenset({
    "llm_call",   # standard mode ReAct loop
    "synthesis",  # financial_analyst mode — final user-facing output only
})

# Human-readable labels shown as progress markers while nodes run silently.
# Phase labels are derived from PHASE_REGISTRY so they stay in sync automatically.
def _build_phase_progress_labels() -> dict[str, str]:
    from agent_sdk.financial.phase_registry import PHASE_REGISTRY
    labels: dict[str, str] = {
        "load_user_context":     "Loading user context...",
        "memory_writer":         "Saving memory...",
        "orchestrate":           "Planning approach...",
        "financial_orchestrate": "🔎 Classifying query & building plan...",
        "comparative_analysis":  "⚖️ Running comparative analysis...",
    }
    for phase_def in PHASE_REGISTRY.values():
        if phase_def.progress_label:
            labels[phase_def.name] = phase_def.progress_label
    return labels


_PHASE_PROGRESS_LABELS: dict[str, str] = _build_phase_progress_labels()


class BaseAgent:
    """
    High-level autonomous agent wrapper.

    - Uses a LangGraph checkpointer for persistence (defaults to InMemorySaver,
      but accepts any LangGraph-compatible checkpointer such as AsyncMongoDBSaver).
    - Relies on `thread_id` to provide short-term conversational memory
      across multiple `run` calls for the same session.
    - Optionally connects to MCP servers to discover remote tools.
    - `mode` parameter selects the graph topology:
        - "standard" (default): flat autonomous loop — works for any agent.
        - "financial_analyst": multi-step cognitive pipeline for structured
          financial reasoning (regime → causal → sector → company → risk → synthesis).
    """

    VALID_MODES = ("standard", "financial_analyst", "research")

    def __init__(self, tools=None, system_prompt=None, provider: str = "azure",
                 mcp_servers: dict | None = None, checkpointer=None,
                 mode: str = "standard", streaming_nodes: set[str] | frozenset[str] | None = None,
                 memory_manager=None, semantic_memory=None,
                 allowed_tools: list[str] | None = None):

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}")

        tools = tools or []
        self.mode = mode
        self.streaming_nodes = streaming_nodes or DEFAULT_STREAMING_NODES

        self.llm = initialize_agent_azure()
        self.summarizer = initialize_summarizer_azure()

        # 3-tier memory system (optional — None disables memory entirely)
        self.memory_manager = memory_manager
        if memory_manager is not None and getattr(memory_manager, "llm", None) is None:
            # Provide the agent's LLM to the memory manager if it doesn't have one
            memory_manager.llm = self.llm
        # Semantic memory tier (optional — None disables semantic memory)
        self.semantic_memory = semantic_memory

        self.tools = list(tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Persistent checkpointer — callers can pass a MongoDB/Redis/Postgres
        # checkpointer for durability; defaults to in-memory for dev/test.
        self.memory = checkpointer or InMemorySaver()
        self.system_prompt = system_prompt or (
            "You are an autonomous assistant. "
            "You may call tools to achieve the user's goal, "
            "or respond directly when tools are not needed."
        )

        self._mcp_servers = mcp_servers
        self._allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self._mcp_manager = None
        self._initialized = False
        self._degraded = False
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._init_lock = None
        self._tool_catalog_cache = {}
        self._bound_llm_cache = {}
        # Cache for financial phase tool lists — populated after MCP init, invalidated on reconnect
        self._phase_tools_cache: dict[str, list] = {}
        # Per-session notepad with TTL eviction (prevents unbounded growth)
        self._session_notepads: _TTLDict = _TTLDict(ttl=3600.0, maxsize=500)
        self._notepad_tools = self._build_notepad_tools()
        # Background running_context compression tasks: session_id → asyncio.Task
        self._pending_ctx_compressions: dict[str, asyncio.Task] = {}
        # Tracked background tasks for graceful shutdown
        self._background_tasks: set[asyncio.Task] = set()

        if mcp_servers:
            # Defer graph creation until MCP tools are discovered
            self.graph = None
            logger.info("BaseAgent created with %d local tool(s) + %d MCP server(s) — graph deferred (mode=%s)",
                        len(self.tools), len(mcp_servers), mode)
            
            # Eager MCP init via background task if an event loop is running
            try:
                asyncio.create_task(self._ensure_initialized())
            except RuntimeError:
                pass
        else:
            # No MCP — build graph immediately (backward compatible)
            self.graph = self._build_graph()
            self._initialized = True
            logger.info("BaseAgent initialized with %d tool(s) (mode=%s): %s",
                        len(self.tools), mode, list(self.tools_by_name.keys()))

    @property
    def _lock(self):
        if self._init_lock is None:
            import asyncio
            self._init_lock = asyncio.Lock()
        return self._init_lock

    def get_tool_catalog(self) -> str:
        key = frozenset(t.name for t in self.tools)
        if key not in self._tool_catalog_cache:
            from .nodes import _format_tool_catalog
            self._tool_catalog_cache[key] = _format_tool_catalog(self.tools)
        return self._tool_catalog_cache[key]

    def get_bound_llm(self, llm_instance: Any, tools: list) -> Any:
        key = frozenset(t.name for t in tools)
        cache_key = (id(llm_instance), key)
        if cache_key not in self._bound_llm_cache:
            self._bound_llm_cache[cache_key] = llm_instance.bind_tools(tools, parallel_tool_calls=True) if tools else llm_instance
        return self._bound_llm_cache[cache_key]

    def _get_breaker(self, name: str) -> CircuitBreaker:
        """Return the per-tool circuit breaker, creating one on first access."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker()
        return self._circuit_breakers[name]

    def get_available_tools(self, phase_tools: list = None) -> list:
        """Return tools that are not currently blocked by an OPEN circuit breaker, plus notepad tools."""
        subset = phase_tools if phase_tools is not None else self.tools
        available = [t for t in subset if not self._get_breaker(t.name).is_open]
        # Always append notepad tools (always available, no circuit breaker needed)
        notepad_names = {t.name for t in self._notepad_tools}
        available = [t for t in available if t.name not in notepad_names] + self._notepad_tools
        return available

    def _build_notepad_tools(self) -> list:
        """Create write_to_notepad and read_notepad tools as closures over this agent instance."""
        from langchain_core.tools import tool as lc_tool
        agent_ref = self

        @lc_tool
        def write_to_notepad(key: str, value: str) -> str:
            """Persist an important discovery to the session notepad for use in follow-up messages.
            Use for: user preferences, confirmed entities, completed sub-task results,
            risk profile, budget constraints. Key should be concise (e.g., 'user_risk_profile',
            'target_ticker', 'completed_dcf')."""
            from agent_sdk.agents.nodes import _current_session_id
            session_id = _current_session_id.get()
            agent_ref._session_notepads.setdefault(session_id, {})[key] = value
            return f"Noted: {key} = {value}"

        @lc_tool
        def read_notepad() -> str:
            """Read all entries saved to the session notepad from earlier in this conversation."""
            from agent_sdk.agents.nodes import _current_session_id
            session_id = _current_session_id.get()
            notepad = agent_ref._session_notepads.get(session_id, {})
            if not notepad:
                return "Session notepad is empty."
            return "\n".join(f"• {k}: {v}" for k, v in notepad.items())

        return [write_to_notepad, read_notepad]

    def _tracked_task(self, coro) -> asyncio.Task:
        """Schedule a background coroutine and track it for graceful shutdown."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _wait_background_tasks(self, timeout: float = 10.0) -> None:
        """Await all pending background tasks before shutdown (best-effort)."""
        if not self._background_tasks:
            return
        logger.info("Waiting for %d background task(s) to complete (timeout=%.0fs)…",
                    len(self._background_tasks), timeout)
        pending = list(self._background_tasks)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Background tasks did not finish within %.0fs — proceeding with shutdown", timeout)

    def _build_graph(self):
        """Build the appropriate graph based on mode."""
        if self.mode == "financial_analyst":
            from .graph import create_financial_reasoning_graph
            return create_financial_reasoning_graph(agent=self, checkpointer=self.memory)
        if self.mode == "research":
            from .research_graph import create_research_graph
            return create_research_graph(agent=self, checkpointer=self.memory)
        return create_graph(agent=self, checkpointer=self.memory)

    async def _ensure_initialized(self):
        """Connect to MCP servers (if configured) and build the graph on first use.
        Includes retries to handle transient MCP server unavailability.
        """
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            if self._mcp_servers:
                from ..mcp.client import MCPConnectionManager
                self._mcp_manager = MCPConnectionManager()

                max_retries = 3

                async def _connect_with_retries():
                    for attempt in range(max_retries):
                        try:
                            return await self._mcp_manager.connect(self._mcp_servers)
                        except Exception as e:
                            if attempt == max_retries - 1:
                                logger.error("MCP initialization failed after %d attempts: %s", max_retries, e)
                                raise
                            backoff = 2 ** attempt
                            logger.warning("MCP connection attempt %d failed: %s — retrying in %ds", attempt + 1, e, backoff)
                            await asyncio.sleep(backoff)

                try:
                    mcp_tools = await asyncio.wait_for(_connect_with_retries(), timeout=15.0)
                    if self._allowed_tools is not None:
                        before = len(mcp_tools)
                        mcp_tools = [t for t in mcp_tools if t.name in self._allowed_tools]
                        logger.info("Tool filter applied: %d → %d tools", before, len(mcp_tools))
                    # Merge MCP tools with any local tools
                    self.tools.extend(mcp_tools)
                    for t in mcp_tools:
                        self.tools_by_name[t.name] = t
                    logger.info("Merged tools — total: %d (%s)", len(self.tools), list(self.tools_by_name.keys()))
                except (asyncio.TimeoutError, Exception) as e:
                    logger.error(
                        "MCP initialization failed — agent running in DEGRADED mode (no MCP tools): %s", e
                    )
                    self._degraded = True

        self.graph = self._build_graph()
        self._initialized = True
        logger.info("BaseAgent graph built with %d tool(s) (mode=%s)", len(self.tools), self.mode)

    async def _disconnect_mcp(self):
        """Cleanly shut down MCP connections."""
        if self._mcp_manager is not None:
            await self._mcp_manager.disconnect()
            self._mcp_manager = None

    async def arun(self, query: str, session_id: str = "default",
                   system_prompt: str | None = None, model_id: str | None = None,
                   **kwargs) -> dict:
        await self._ensure_initialized()

        logger.info("Agent run started — session='%s', query='%s', model_id='%s', extra_fields=%s",
                    session_id, query[:100], model_id or "default", list(kwargs.keys()))

        invoke_input: dict[str, Any] = {
            "messages": [HumanMessage(content=query)],
            "system_prompt": system_prompt or self.system_prompt,
            "iteration": 0,
            "session_id": session_id,
            "scratchpad": None,
        }
        if model_id:
            invoke_input["model_id"] = model_id

        # Merge any extra state fields (e.g. as_of_date, user_id)
        invoke_input.update(kwargs)

        result = await self.graph.ainvoke(
            invoke_input,
            config={"recursion_limit": 100, "configurable": {"thread_id": session_id}},
        )

        raw = result["messages"][-1].content

        if isinstance(raw, list):
            response = "".join(
                block["text"] for block in raw if block.get("type") == "text"
            )
        else:
            response = raw

        # SDK-level unwrapping of structured synthesis
        from agent_sdk.utils.output import unwrap_structured_response
        response = unwrap_structured_response(response)

        # Build execution trace from message history
        steps = []

        from langchain_core.messages import AIMessage, ToolMessage
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                for tc in tool_calls:
                    steps.append({
                        "action": "tool_call",
                        "tool": tc["name"],
                        "args": tc.get("args", {}),
                    })
            elif isinstance(msg, ToolMessage):
                content = msg.content
                steps.append({
                    "action": "tool_result",
                    "tool_call_id": msg.tool_call_id,
                    "result_length": len(content),
                    "result_preview": content[:500] if len(content) > 500 else content,
                })

        # Financial mode: tool calls live in tool_calls_log (not state.messages)
        steps.extend(result.get("tool_calls_log", []))

        logger.info("Agent run completed — session='%s', response length: %d chars, steps: %d",
                    session_id, len(response), len(steps))
        return {"response": response, "steps": steps, "plan": result.get("scratchpad")}

    def astream(self, query: str, session_id: str = "default",
                system_prompt: str | None = None, model_id: str | None = None,
                **kwargs):
        """Return a StreamResult that yields text chunks and tracks tool calls.

        Usage:
            stream = agent.astream(query, session_id=session_id)
            async for chunk in stream:
                # send chunk to client
            steps = stream.steps  # available after iteration completes
        """
        return StreamResult(self, query, session_id, system_prompt or self.system_prompt, model_id, **kwargs)


class StreamResult:
    """Async iterator that streams text chunks and collects execution steps."""

    def __init__(self, agent: "BaseAgent", query: str, session_id: str,
                 system_prompt: str, model_id: str | None = None, **kwargs):
        self._agent = agent
        self._query = query
        self._session_id = session_id
        self._system_prompt = system_prompt
        self._model_id = model_id
        self._extra_fields = kwargs
        self.steps: list[dict] = []
        self.plan: str | None = None

    def __aiter__(self):
        return self._stream()

    async def _stream(self):
        await self._agent._ensure_initialized()

        logger.info("Agent stream started — session='%s', query='%s'", self._session_id, self._query[:100])

        import json as _json

        chunks_yielded = False
        last_full_response = ""

        stream_input: dict[str, Any] = {
            "messages": [HumanMessage(content=self._query)],
            "system_prompt": self._system_prompt,
            "iteration": 0,
            "session_id": self._session_id,
            "scratchpad": None,
        }
        if self._model_id:
            stream_input["model_id"] = self._model_id

        # Merge any extra state fields (e.g. as_of_date, user_id)
        stream_input.update(self._extra_fields)

        _phases_announced: set[str] = set()

        async for event in self._agent.graph.astream_events(
            stream_input,
            config={"recursion_limit": 100, "configurable": {"thread_id": self._session_id}},
            version="v2",
        ):
            # Emit a progress marker when a financial pipeline phase node starts.
            # Shown for all phase nodes so users see pipeline progress even when
            # intermediate phases are excluded from _STREAMING_NODES.
            if event["event"] == "on_chain_start":
                node = event.get("metadata", {}).get("langgraph_node") or event.get("name", "")
                label = _PHASE_PROGRESS_LABELS.get(node)
                if label and node not in _phases_announced:
                    _phases_announced.add(node)
                    # Prefixed so callers (e.g. the SSE event_stream) can route
                    # progress lines to a separate event type and exclude them
                    # from the saved conversation response.
                    yield f"__PROGRESS__:{label}"

            elif event["event"] == "on_tool_start":
                tool_name = event.get("name", "unknown")
                # Clean up tool names for better UX (e.g. resolve_indian_ticker -> Resolving ticker)
                label = f"🛠️ Executing {tool_name.replace('_', ' ')}..."
                yield f"__PROGRESS__:{label}"

            elif event["event"] == "on_tool_end":
                tool_name = event.get("name", "unknown")
                label = f"✅ Completed {tool_name.replace('_', ' ')}"
                yield f"__PROGRESS__:{label}"

            if event["event"] == "on_chat_model_stream":
                # Only stream user-facing LLM nodes, not the summarizer.
                # Nested subgraph nodes may report as "parent:llm_call".
                node = event.get("metadata", {}).get("langgraph_node")
                node_parts = set(node.split(":")) if isinstance(node, str) else set()
                is_streaming = any(
                    node == n or n in node_parts
                    for n in self._agent.streaming_nodes
                )
                if not is_streaming:
                    continue
                chunk = event["data"]["chunk"]
                content = chunk.content

                if isinstance(content, str) and content:
                    chunks_yielded = True
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                            chunks_yielded = True
                            yield block["text"]

            elif event["event"] == "on_chat_model_end":
                # Only track from the main LLM, not the summarizer.
                # Nested subgraph nodes may report as "parent:summarize_conversation".
                node = event.get("metadata", {}).get("langgraph_node")
                node_parts = set(node.split(":")) if isinstance(node, str) else set()
                if "summarize_conversation" in node_parts:
                    continue

                # Track tool calls from LLM responses
                output = event["data"].get("output")
                if output:
                    tool_calls = getattr(output, "tool_calls", None) or []
                    for tc in tool_calls:
                        self.steps.append({
                            "action": "tool_call",
                            "tool": tc["name"],
                            "args": tc.get("args", {}),
                        })
                    # Capture full response as fallback
                    content = getattr(output, "content", None)
                    if isinstance(content, str) and content:
                        last_full_response = content
                    elif isinstance(content, list):
                        parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                parts.append(block["text"])
                        if parts:
                            last_full_response = "".join(parts)

            elif event["event"] == "on_tool_end":
                # Track tool results
                output = event["data"].get("output")
                if output:
                    content = getattr(output, "content", "") or str(output)
                    self.steps.append({
                        "action": "tool_result",
                        "tool": event.get("name", "unknown"),
                        "result_length": len(content),
                        "result_preview": content[:500] if len(content) > 500 else content,
                    })

        # Read final graph state from checkpoint to capture:
        # - tool_calls_log: financial phase tool calls (not in messages)
        # - scratchpad: the orchestrator's execution plan
        try:
            snapshot = await self._agent.graph.aget_state(
                {"configurable": {"thread_id": self._session_id}}
            )
            if snapshot:
                log = snapshot.values.get("tool_calls_log") or []
                if log:
                    # State-tracked log is authoritative for financial mode
                    self.steps = log
                self.plan = snapshot.values.get("scratchpad")
        except Exception:
            pass  # best-effort; existing event-derived steps are still valid

        # Fallback: if no streaming chunks were yielded, emit the full response
        if not chunks_yielded and last_full_response:
            logger.warning(
                "No streaming chunks received — falling back to full response (%d chars)",
                len(last_full_response),
            )
            # Also unwrap JSON on the fallback path
            clean_fallback = last_full_response
            try:
                from agent_sdk.utils.output import unwrap_structured_response
                clean_fallback = unwrap_structured_response(last_full_response)
            except Exception:
                pass
            yield clean_fallback


    def run(self, query: str, session_id: str = "default") -> dict:
        """
        Synchronous convenience wrapper around `arun`.
        """

        # Lazy import to avoid forcing any particular async runner
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If already in an event loop, schedule the coroutine.
            # Caller is responsible for awaiting `arun` directly in async code.
            raise RuntimeError(
                "BaseAgent.run() cannot be called from an async context. "
                "Use `await BaseAgent.arun(...)` instead."
            )

        return asyncio.run(self.arun(query, session_id=session_id))
