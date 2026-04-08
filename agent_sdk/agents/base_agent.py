import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .graph import create_graph
from .state import AgentState
from ..llm_services.agent_llm import initialize_azure as initialize_agent_azure
from ..llm_services.summarizer_llm import initialize_azure as initialize_summarizer_azure
from ..mcp.circuit_breaker import CircuitBreaker

logger = logging.getLogger("agent_sdk.agent")

# Nodes that produce user-facing LLM output and should be streamed to the client.
# Covers both the standard graph ("llm_call") and the financial reasoning graph.
DEFAULT_STREAMING_NODES = frozenset({
    "llm_call",
    "regime_assessment", "causal_analysis",
    "sector_analysis", "company_analysis",
    "risk_assessment", "synthesis",
})

# Human-readable labels for financial pipeline phases — shown as progress markers
# while intermediate phases run silently (i.e. when streaming_nodes is restricted).
_PHASE_PROGRESS_LABELS: dict[str, str] = {
    "classify_query":        "🔎 Classifying query...",
    "regime_assessment":     "🌐 Analyzing macro regime & market environment...",
    "causal_analysis":       "🔗 Mapping causal transmission chains...",
    "sector_analysis":       "📊 Evaluating sector positioning...",
    "company_analysis":      "🏢 Running company fundamental analysis...",
    "comparative_analysis":  "⚖️ Running comparative analysis...",
    "risk_assessment":       "⚠️ Stress-testing scenarios & risks...",
    "synthesis":             "✍️ Synthesizing final report...",
}


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

    VALID_MODES = ("standard", "financial_analyst")

    def __init__(self, tools=None, system_prompt=None, provider: str = "azure",
                 mcp_servers: dict | None = None, checkpointer=None,
                 mode: str = "standard", streaming_nodes: set[str] | frozenset[str] | None = None):

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}")

        tools = tools or []
        self.mode = mode
        self.streaming_nodes = streaming_nodes or DEFAULT_STREAMING_NODES

        self.llm = initialize_agent_azure()
        self.summarizer = initialize_summarizer_azure()

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
        self._mcp_manager = None
        self._initialized = False
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

        if mcp_servers:
            # Defer graph creation until MCP tools are discovered
            self.graph = None
            logger.info("BaseAgent created with %d local tool(s) + %d MCP server(s) — graph deferred (mode=%s)",
                        len(self.tools), len(mcp_servers), mode)
        else:
            # No MCP — build graph immediately (backward compatible)
            self.graph = self._build_graph()
            self._initialized = True
            logger.info("BaseAgent initialized with %d tool(s) (mode=%s): %s",
                        len(self.tools), mode, list(self.tools_by_name.keys()))

    def _get_breaker(self, name: str) -> CircuitBreaker:
        """Return the per-tool circuit breaker, creating one on first access."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker()
        return self._circuit_breakers[name]

    def _build_graph(self):
        """Build the appropriate graph based on mode."""
        if self.mode == "financial_analyst":
            from .graph import create_financial_reasoning_graph
            return create_financial_reasoning_graph(agent=self, checkpointer=self.memory)
        return create_graph(agent=self, checkpointer=self.memory)

    async def _ensure_initialized(self):
        """Connect to MCP servers (if configured) and build the graph on first use."""
        if self._initialized:
            return

        if self._mcp_servers:
            from ..mcp.client import MCPConnectionManager
            self._mcp_manager = MCPConnectionManager()
            mcp_tools = await self._mcp_manager.connect(self._mcp_servers)

            # Merge MCP tools with any local tools
            self.tools.extend(mcp_tools)
            for t in mcp_tools:
                self.tools_by_name[t.name] = t

            logger.info("Merged tools — total: %d (%s)", len(self.tools), list(self.tools_by_name.keys()))

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
        }
        if model_id:
            invoke_input["model_id"] = model_id
        
        # Merge any extra state fields (e.g. as_of_date)
        invoke_input.update(kwargs)
        
        # For financial_analyst mode, set up per-phase iteration budgets
        if self.mode == "financial_analyst":
            invoke_input["phase_iteration_budgets"] = {
                "query_classification": 1,
                "regime_assessment": 2,
                "causal_analysis": 2,
                "sector_analysis": 2,
                "company_analysis": 4,
                "risk_assessment": 2,
                "synthesis": 3,
            }

        result = await self.graph.ainvoke(
            invoke_input,
            config={"recursion_limit": 100, "configurable": {"thread_id": session_id}},
        )

        # For financial_analyst mode, prefer the structured synthesis report
        if self.mode == "financial_analyst":
            synthesis = result.get("synthesis_report") or {}
            raw = synthesis.get("full_report", "")
        else:
            raw = ""

        # Fallback to last message content
        if not raw:
            raw = result["messages"][-1].content

        if isinstance(raw, list):
            response = "".join(
                block["text"] for block in raw if block.get("type") == "text"
            )
        else:
            response = raw

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

        logger.info("Agent run completed — session='%s', response length: %d chars, steps: %d",
                    session_id, len(response), len(steps))
        # Include structured synthesis (if available) so callers can render rich UI.
        # Only exposed for financial_analyst mode to avoid unexpected payload bloat elsewhere.
        structured = result.get("synthesis_report") if self.mode == "financial_analyst" else None
        return {"response": response, "steps": steps, "synthesis_report": structured}

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

    def __aiter__(self):
        return self._stream()

    async def _stream(self):
        await self._agent._ensure_initialized()

        logger.info("Agent stream started — session='%s', query='%s'", self._session_id, self._query[:100])

        import json as _json

        chunks_yielded = False
        last_full_response = ""
        # Synthesis outputs JSON-wrapped content {"full_report": "..."}. Buffer its tokens
        # and unwrap on model-end so the client receives clean markdown, not raw JSON.
        _synthesis_buffer: list[str] = []

        stream_input: dict[str, Any] = {
            "messages": [HumanMessage(content=self._query)],
            "system_prompt": self._system_prompt,
            "iteration": 0,
        }
        if self._model_id:
            stream_input["model_id"] = self._model_id
        
        # Merge any extra state fields (e.g. as_of_date)
        stream_input.update(self._extra_fields)

        # For financial_analyst mode, set up per-phase iteration budgets
        if self._agent.mode == "financial_analyst":
            stream_input["phase_iteration_budgets"] = {
                "query_classification": 1,
                "regime_assessment": 2,
                "causal_analysis": 2,
                "sector_analysis": 2,
                "company_analysis": 4,
                "risk_assessment": 2,
                "synthesis": 3,
            }

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

            if event["event"] == "on_chat_model_stream":
                # Only stream user-facing LLM nodes, not the summarizer
                node = event.get("metadata", {}).get("langgraph_node")
                if node not in self._agent.streaming_nodes:
                    continue
                chunk = event["data"]["chunk"]
                content = chunk.content

                if node == "synthesis":
                    # Buffer instead of yielding — synthesis outputs JSON that must be unwrapped first
                    if isinstance(content, str) and content:
                        _synthesis_buffer.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                _synthesis_buffer.append(block["text"])
                else:
                    if isinstance(content, str) and content:
                        chunks_yielded = True
                        yield content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                                chunks_yielded = True
                                yield block["text"]

            elif event["event"] == "on_chat_model_end":
                # Only track from the main LLM, not the summarizer
                node = event.get("metadata", {}).get("langgraph_node")
                if node == "summarize_conversation":
                    continue

                # Flush the synthesis buffer with JSON unwrapping
                if node == "synthesis" and _synthesis_buffer:
                    raw = "".join(_synthesis_buffer)
                    _synthesis_buffer.clear()
                    clean = raw
                    try:
                        parsed = _json.loads(raw)
                        if isinstance(parsed, dict) and "full_report" in parsed:
                            clean = parsed["full_report"]
                            logger.debug(
                                "Synthesis: unwrapped JSON full_report (%d → %d chars)",
                                len(raw), len(clean),
                            )
                    except Exception:
                        pass  # Not valid JSON — stream as-is
                    if clean:
                        chunks_yielded = True
                        yield clean

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

        # Fallback: if no streaming chunks were yielded, emit the full response
        if not chunks_yielded and last_full_response:
            logger.warning(
                "No streaming chunks received — falling back to full response (%d chars)",
                len(last_full_response),
            )
            # Also unwrap JSON on the fallback path
            clean_fallback = last_full_response
            try:
                parsed = _json.loads(last_full_response)
                if isinstance(parsed, dict) and "full_report" in parsed:
                    clean_fallback = parsed["full_report"]
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
