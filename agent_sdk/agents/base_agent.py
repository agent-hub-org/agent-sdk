from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .graph import create_graph
from .state import AgentState

try:  # Local import; ignore if llm package is not present at type-check time
    from ..llm.factory import create_llm, create_summarizer  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - defensive fallback
    create_llm = create_summarizer = None  # type: ignore[assignment]


class BaseAgent:
    """
    High-level autonomous agent wrapper.

    - Uses a LangGraph checkpointer (`InMemorySaver`) for persistence.
    - Relies on `thread_id` to provide short-term conversational memory
      across multiple `run` calls for the same session.
    """

    def __init__(self, tools=None):
        tools = tools or []

        self.llm = create_llm()
        self.summarizer = create_summarizer()

        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}

        # Persistent, per-thread conversational memory
        self.memory = InMemorySaver()

        # Autonomous LangGraph agent with persistence
        self.graph = create_graph(checkpointer=self.memory)

    async def arun(self, query: str, session_id: str = "default") -> str:
        """
        Async entrypoint for low-latency execution.

        Uses the graph's `ainvoke` method so that async LLMs/tools
        (e.g. Groq-backed models) can be awaited end-to-end.
        """

        state = AgentState(
            messages=[HumanMessage(content=query)],
            tools_by_name=self.tools_by_name,
            llm=self.llm,
            summarizer_llm=self.summarizer,
        )

        result = await self.graph.ainvoke(
            state,
            config={"configurable": {"thread_id": session_id}},
        )

        return result["messages"][-1].content

    def run(self, query: str, session_id: str = "default") -> str:
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