import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .graph import create_graph
from .state import AgentState
from ..llm_services.agent_llm import initialize_gemini as initialize_agent_gemini
from ..llm_services.agent_llm import initialize_groq as initialize_agent_groq
from ..llm_services.summarizer_llm import initialize_gemini as initialize_summarizer_gemini
from ..llm_services.summarizer_llm import initialize_groq as initialize_summarizer_groq

logger = logging.getLogger("agent_sdk.agent")


class BaseAgent:
    """
    High-level autonomous agent wrapper.

    - Uses a LangGraph checkpointer (`InMemorySaver`) for persistence.
    - Relies on `thread_id` to provide short-term conversational memory
      across multiple `run` calls for the same session.
    """

    def __init__(self, tools=None, system_prompt=None, provider: str = "groq"):
        tools = tools or []

        if provider == "gemini":
            self.llm = initialize_agent_gemini()
            self.summarizer = initialize_summarizer_gemini()
        else:
            self.llm = initialize_agent_groq()
            self.summarizer = initialize_summarizer_groq()

        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}

        # Persistent, per-thread conversational memory
        self.memory = InMemorySaver()
        self.system_prompt = system_prompt or (
            "You are an autonomous assistant. "
            "You may call tools to achieve the user's goal, "
            "or respond directly when tools are not needed."
        )

        # Autonomous LangGraph agent with persistence
        self.graph = create_graph(agent=self, checkpointer=self.memory)
        logger.info("BaseAgent initialized with %d tool(s): %s",
                    len(self.tools), list(self.tools_by_name.keys()))

    async def arun(self, query: str, session_id: str = "default", system_prompt: str | None = None) -> dict:
        logger.info("Agent run started — session='%s', query='%s'", session_id, query[:100])

        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=query)],
                "system_prompt": system_prompt or self.system_prompt,
            },
            config={"configurable": {"thread_id": session_id}},
        )

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