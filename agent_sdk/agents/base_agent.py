import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .graph import create_graph
from .state import AgentState
from ..llm_services.agent_llm import initialize_azure as initialize_agent_azure
from ..llm_services.summarizer_llm import initialize_azure as initialize_summarizer_azure

logger = logging.getLogger("agent_sdk.agent")

# Nodes that produce user-facing LLM output and should be streamed to the client.
# Covers both the standard graph ("llm_call") and the financial reasoning graph.
_STREAMING_NODES = {
    "llm_call",
    "regime_assessment", "causal_analysis",
    "sector_analysis", "company_analysis",
    "risk_assessment", "synthesis",
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
                 mode: str = "standard"):

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}")

        tools = tools or []
        self.mode = mode

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
                   system_prompt: str | None = None, model_id: str | None = None) -> dict:
        await self._ensure_initialized()

        logger.info("Agent run started — session='%s', query='%s', model_id='%s'",
                    session_id, query[:100], model_id or "default")

        invoke_input = {
            "messages": [HumanMessage(content=query)],
            "system_prompt": system_prompt or self.system_prompt,
            "iteration": 0,
        }
        if model_id:
            invoke_input["model_id"] = model_id

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
        return {"response": response, "steps": steps}

    def astream(self, query: str, session_id: str = "default",
                system_prompt: str | None = None, model_id: str | None = None):
        """Return a StreamResult that yields text chunks and tracks tool calls.

        Usage:
            stream = agent.astream(query, session_id=session_id)
            async for chunk in stream:
                # send chunk to client
            steps = stream.steps  # available after iteration completes
        """
        return StreamResult(self, query, session_id, system_prompt or self.system_prompt, model_id)


class StreamResult:
    """Async iterator that streams text chunks and collects execution steps."""

    def __init__(self, agent: "BaseAgent", query: str, session_id: str,
                 system_prompt: str, model_id: str | None = None):
        self._agent = agent
        self._query = query
        self._session_id = session_id
        self._system_prompt = system_prompt
        self._model_id = model_id
        self.steps: list[dict] = []

    def __aiter__(self):
        return self._stream()

    async def _stream(self):
        await self._agent._ensure_initialized()

        logger.info("Agent stream started — session='%s', query='%s'", self._session_id, self._query[:100])

        chunks_yielded = False
        last_full_response = ""

        stream_input = {
            "messages": [HumanMessage(content=self._query)],
            "system_prompt": self._system_prompt,
            "iteration": 0,
        }
        if self._model_id:
            stream_input["model_id"] = self._model_id

        async for event in self._agent.graph.astream_events(
            stream_input,
            config={"recursion_limit": 100, "configurable": {"thread_id": self._session_id}},
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                # Only stream user-facing LLM nodes, not the summarizer
                node = event.get("metadata", {}).get("langgraph_node")
                if node not in _STREAMING_NODES:
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
                # Only track from the main LLM, not the summarizer
                node = event.get("metadata", {}).get("langgraph_node")
                if node == "summarize_conversation":
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

        # Fallback: if no streaming chunks were yielded, emit the full response
        if not chunks_yielded and last_full_response:
            logger.warning("No streaming chunks received — falling back to full response (%d chars)",
                          len(last_full_response))
            yield last_full_response

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
