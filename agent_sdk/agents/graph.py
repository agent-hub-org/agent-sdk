from __future__ import annotations

from functools import partial
from typing import Any, Optional

from langgraph.graph import StateGraph, START, END

from agent_sdk.agents.state import AgentState
from agent_sdk.agents.nodes import (
    initialize,
    llm_call,
    tool_node,
    summarize_conversation,
    should_continue,
)


def create_graph(agent, checkpointer: Optional[Any] = None):
    """
    Build a LangGraph-based autonomous agent graph with optional persistence.

    Dependencies (`llm`, `tools_by_name`, `summarizer`) live on the `agent`
    instance and are bound to node functions via `functools.partial`.
    This keeps AgentState free of non-serializable objects.

    Persistence & short-term memory
    --------------------------------
    - Short-term conversational memory is handled by:
      - `AgentState.messages` being annotated with `add_messages` so messages
        accumulate across steps and invocations.
      - A LangGraph checkpointer (e.g. `InMemorySaver`) passed as `checkpointer`
        and a `thread_id` in the `config` when invoking the graph.
    - `summary` is preserved across invocations because only
      `{"messages": [...]}` is passed to `ainvoke`, so the checkpointed
      summary is never overwritten.

    Graph flow
    ----------
    START → initialize → llm_call → should_continue → tool_node → llm_call (loop)
                                                     → summarize_conversation → END
                                                     → END
    """

    graph = StateGraph(AgentState)

    # initialize and should_continue don't need agent — register directly
    graph.add_node("initialize", initialize)
    graph.add_node("llm_call", partial(llm_call, agent))
    graph.add_node("tool_node", partial(tool_node, agent))
    graph.add_node("summarize_conversation", partial(summarize_conversation, agent))

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "llm_call")
    graph.add_conditional_edges("llm_call", should_continue)
    graph.add_edge("tool_node", "llm_call")
    graph.add_edge("summarize_conversation", END)

    return graph.compile(checkpointer=checkpointer)