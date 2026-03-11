from __future__ import annotations

from typing import Any, Optional

from langgraph.graph import StateGraph, START, END

from agent_sdk.agents.state import AgentState
from agent_sdk.agents.nodes import llm_call, tool_node, should_continue


def create_graph(checkpointer: Optional[Any] = None):
    """
    Build a LangGraph-based autonomous agent graph with optional persistence.

    Persistence & short-term memory
    --------------------------------
    - Short-term conversational memory is handled by:
      - `AgentState.messages` being annotated with `add_messages` so messages
        accumulate across steps and invocations.
      - A LangGraph checkpointer (e.g. `InMemorySaver`) passed as `checkpointer`
        and a `thread_id` in the `config` when invoking the graph.
    """

    graph = StateGraph(AgentState)

    graph.add_node("llm_call", llm_call)
    graph.add_node("tool_node", tool_node)

    graph.add_edge(START, "llm_call")
    graph.add_conditional_edges("llm_call", should_continue)
    graph.add_edge("tool_node", "llm_call")

    graph.add_edge("llm_call", END)

    return graph.compile(checkpointer=checkpointer)