"""
Reusable ReAct loop subgraph.

Extracts the standard llm_call -> tool_node cycle into a compiled subgraph
that can be nested inside any parent graph (standard mode, financial phases,
research synthesis, entity analysis).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from langgraph.graph import StateGraph, START, END

from agent_sdk.agents.state import AgentState
from agent_sdk.agents.nodes import (
    llm_call,
    tool_node,
    summarize_conversation,
    should_continue,
    post_tool_router,
    pre_llm_router,
)

logger = logging.getLogger("agent_sdk.subgraphs.react")


def create_react_subgraph(agent, state_schema=AgentState):
    """
    Build a reusable ReAct loop subgraph.

    Compatible with AgentState or any subclass (FinancialAnalysisState,
    ResearchState) because the inner nodes read/write fields present on all
    of them (messages, iteration, summary, etc.).

    The compiled subgraph is attached to the parent graph as a single node.
    LangGraph handles nested checkpointing automatically.
    """
    graph = StateGraph(state_schema)

    graph.add_node("llm_call", partial(llm_call, agent))
    graph.add_node("tool_node", partial(tool_node, agent))
    graph.add_node("summarize_conversation", partial(summarize_conversation, agent))

    # Entry router — summarize first if context is large
    graph.add_conditional_edges(START, pre_llm_router, {
        "llm_call": "llm_call",
        "summarize_conversation": "summarize_conversation",
    })

    # Core loop
    graph.add_conditional_edges("llm_call", should_continue, {
        "tool_node": "tool_node",
        "llm_call": "llm_call",
        "summarize_conversation": "summarize_conversation",
        END: END,
    })
    graph.add_conditional_edges("tool_node", post_tool_router)
    graph.add_edge("summarize_conversation", "llm_call")

    return graph
