from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .graph import create_graph
from .state import AgentState
from ..llm_services.agent_llm import initialize_llm as initialize_agent_llm
from ..llm_services.summarizer_llm import initialize_llm as initialize_summarizer_llm


class BaseAgent:
    """
    High-level autonomous agent wrapper.

    - Uses a LangGraph checkpointer (`InMemorySaver`) for persistence.
    - Relies on `thread_id` to provide short-term conversational memory
      across multiple `run` calls for the same session.
    """

    def __init__(self, tools=None, system_prompt=None):
        tools = tools or []

        self.llm = initialize_agent_llm()
        self.summarizer = initialize_summarizer_llm()

        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}

        # Persistent, per-thread conversational memory
        self.memory = InMemorySaver()
        system_prompt = system_prompt or (
            "You are an autonomous assistant. "
            "You may call tools to achieve the user's goal, "
            "or respond directly when tools are not needed."
        )

        # Autonomous LangGraph agent with persistence
        self.graph = create_graph(agent=self, checkpointer=self.memory)

    async def arun(self, query: str, session_id: str = "default") -> str:
        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=query)],
                "system_prompt": self.system_prompt,
            },
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