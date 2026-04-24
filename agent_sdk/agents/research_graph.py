"""
Research agent LangGraph subgraph.

Enforces the mandatory retrieve -> check -> [download -> retrieve] -> synthesize
workflow at the graph level rather than relying on the system prompt.

Graph topology:
    START -> research_initialize -> retrieve -> check_if_sufficient
        -> research_router -> synthesize -> memory_writer -> END
                          -> download_and_retrieve -> synthesize
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from agent_sdk.agents.state import ResearchState
from agent_sdk.agents.subgraphs import create_react_subgraph
from agent_sdk.agents.nodes import (
    initialize,
    memory_writer,
    tool_node,
    _execute_tool_calls,
    _invoke_with_retry,
)

logger = logging.getLogger("agent_sdk.research_graph")

# Keywords that signal a math/theory question requiring hybrid_retrieve_papers
_THEORY_KEYWORDS = frozenset({
    "derive", "derivation", "proof", "prove", "theorem", "lemma",
    "algorithm", "mathematical", "equation", "formula", "convergence",
    "gradient", "backpropagation", "optimisation", "optimization",
    "complexity", "loss function", "objective", "regularization",
})


def _extract_clean_query(state: ResearchState) -> str:
    """Extract the most recent human message as the research query."""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # Strip the [CONTEXT]...[/CONTEXT] prefix added by the research agent
            marker = "[/CONTEXT]"
            if marker in content:
                content = content[content.find(marker) + len(marker):].strip()
            return content.strip()
    return ""


def _needs_theory_search(query: str) -> bool:
    """Return True if the query likely requires hybrid (mathematical) paper retrieval."""
    lower = query.lower()
    return any(kw in lower for kw in _THEORY_KEYWORDS)


async def research_initialize(agent, state: ResearchState) -> dict:
    """Extract and normalize the research query, then run the standard initialize node."""
    base = await initialize(state)
    clean_query = _extract_clean_query(state)
    logger.info("research_initialize — query='%s'", clean_query[:80])
    return {**base, "research_query": clean_query}


async def retrieve(agent, state: ResearchState) -> dict:
    """
    Call retrieve_papers (and optionally hybrid_retrieve_papers for math queries).
    Stores results in state.retrieved_papers.
    """
    query = state.research_query or _extract_clean_query(state)
    logger.info("retrieve — query='%s'", query[:80])

    tool_calls: list[dict] = []

    retrieve_tool = agent.tools_by_name.get("retrieve_papers")
    hybrid_tool = agent.tools_by_name.get("hybrid_retrieve_papers")

    if retrieve_tool:
        tool_calls.append({
            "name": "retrieve_papers",
            "args": {"query": query, "top_k": 5},
            "id": "retrieve_papers_0",
        })

    if hybrid_tool and _needs_theory_search(query):
        tool_calls.append({
            "name": "hybrid_retrieve_papers",
            "args": {"query": query, "top_k": 5},
            "id": "hybrid_retrieve_papers_0",
        })

    if not tool_calls:
        logger.warning("retrieve — no retrieve_papers tool available, skipping")
        return {"papers_sufficient": False}

    messages = await _execute_tool_calls(
        agent, tool_calls, timeout=60.0
    )

    papers: list[dict] = []
    for msg in messages:
        if msg.content and not msg.content.startswith("Error"):
            papers.append({"source": msg.tool_call_id, "content": msg.content})

    logger.info("retrieve — got %d results", len(papers))
    # Append retrieved_papers (uses operator.add reducer)
    return {"retrieved_papers": papers, "messages": messages}


async def check_if_sufficient(agent, state: ResearchState) -> dict:
    """
    Fast LLM check: are the retrieved papers sufficient to answer the query?
    Uses the summarizer model (smaller/faster) to decide.
    """
    query = state.research_query
    papers = state.retrieved_papers

    if not papers:
        logger.info("check_if_sufficient — no papers, marking insufficient")
        return {"papers_sufficient": False}

    summarizer = getattr(agent, "summarizer", None) or agent.llm
    papers_text = "\n---\n".join(p.get("content", "")[:500] for p in papers[-5:])

    prompt = [
        SystemMessage(content=(
            "You are a research relevance checker. "
            "Given a user query and retrieved paper excerpts, answer ONLY 'YES' or 'NO': "
            "Do the retrieved papers contain enough information to meaningfully answer the query? "
            "Answer YES if the papers are on-topic and substantive. "
            "Answer NO if the papers are off-topic, empty, or very sparse."
        )),
        HumanMessage(content=f"Query: {query}\n\nRetrieved papers:\n{papers_text}"),
    ]

    try:
        response = await _invoke_with_retry(summarizer, prompt)
        answer = (response.content or "").strip().upper()
        sufficient = answer.startswith("YES")
        logger.info("check_if_sufficient — answer='%s', sufficient=%s", answer[:10], sufficient)
    except Exception as e:
        logger.warning("check_if_sufficient failed: %s — treating as insufficient", e)
        sufficient = False

    return {"papers_sufficient": sufficient}


async def download_and_retrieve(agent, state: ResearchState) -> dict:
    """
    Download fresh papers from arXiv then retrieve again.
    Sets download_attempted=True so the router doesn't loop.
    """
    query = state.research_query
    logger.info("download_and_retrieve — query='%s'", query[:80])

    tool_calls: list[dict] = []

    if "download_and_store_arxiv_papers" in agent.tools_by_name:
        tool_calls.append({
            "name": "download_and_store_arxiv_papers",
            "args": {"query": query, "max_results": 5},
            "id": "download_0",
        })

    if tool_calls:
        await _execute_tool_calls(agent, tool_calls, timeout=120.0)

    # Retrieve again after download
    if "retrieve_papers" in agent.tools_by_name:
        retrieve_calls = [{
            "name": "retrieve_papers",
            "args": {"query": query, "top_k": 5},
            "id": "retrieve_after_download_0",
        }]
        messages = await _execute_tool_calls(agent, retrieve_calls, timeout=60.0)

        papers: list[dict] = []
        for msg in messages:
            if msg.content and not msg.content.startswith("Error"):
                papers.append({"source": msg.tool_call_id, "content": msg.content})

        logger.info("download_and_retrieve — retrieved %d papers after download", len(papers))
        return {
            "download_attempted": True,
            "retrieved_papers": papers,
            "messages": messages,
        }

    return {"download_attempted": True}


def research_router(state: ResearchState) -> str:
    """Route to synthesize if papers are sufficient or download was attempted; else download."""
    if state.papers_sufficient or state.download_attempted:
        logger.info("research_router -> synthesize (sufficient=%s, downloaded=%s)",
                    state.papers_sufficient, state.download_attempted)
        return "synthesize"
    logger.info("research_router -> download_and_retrieve")
    return "download_and_retrieve"


def create_research_graph(agent, checkpointer: Optional[Any] = None):
    """
    Build the research agent graph with enforced retrieve->download->retrieve workflow.

    Graph flow:
        START -> research_initialize -> retrieve -> check_if_sufficient
            -> research_router -> synthesize -> memory_writer -> END
                              -> download_and_retrieve -> synthesize
    """
    graph = StateGraph(ResearchState)

    # Reuse the standard ReAct subgraph for synthesis so it gets per-iteration checkpointing
    react = create_react_subgraph(agent, ResearchState).compile()

    graph.add_node("research_initialize", partial(research_initialize, agent))
    graph.add_node("retrieve", partial(retrieve, agent))
    graph.add_node("check_if_sufficient", partial(check_if_sufficient, agent))
    graph.add_node("download_and_retrieve", partial(download_and_retrieve, agent))
    graph.add_node("synthesize", react)
    graph.add_node("memory_writer", partial(memory_writer, agent))

    graph.add_edge(START, "research_initialize")
    graph.add_edge("research_initialize", "retrieve")
    graph.add_edge("retrieve", "check_if_sufficient")
    graph.add_conditional_edges("check_if_sufficient", research_router, {
        "synthesize": "synthesize",
        "download_and_retrieve": "download_and_retrieve",
    })
    graph.add_edge("download_and_retrieve", "synthesize")
    graph.add_edge("synthesize", "memory_writer")
    graph.add_edge("memory_writer", END)

    return graph.compile(checkpointer=checkpointer)
