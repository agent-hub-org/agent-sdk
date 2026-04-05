# agent-sdk

Shared Python library providing a high-level autonomous agent framework built on LangGraph.

## Features

- **BaseAgent** — autonomous agent wrapper with LangGraph, InMemorySaver checkpointer for session persistence
- **LLM Providers** — Groq (llama-3.3-70b-versatile), Gemini (gemini-2.5-flash), NVIDIA
- **MCP Client** — `MCPConnectionManager` connects to remote FastMCP servers, converts tools to LangChain `StructuredTool` objects
- **Lazy Initialization** — when `mcp_servers` is provided, graph creation is deferred until first `arun()` call; without MCP, graph builds immediately (backward compatible)
- **Conversation Summarization** — automatic context management with configurable token limits and message retention
- **Configurable** — iteration limits, context token limits, `keep_last_n_messages`

## Architecture

LangGraph flow:

```
START → initialize → llm_call → should_continue → tool_node → post_tool_router → llm_call
                                                 → summarize_conversation → llm_call
                                                 → END
```

## Structure

```
agent_sdk/
├── agents/
│   ├── base_agent.py       # BaseAgent class
│   ├── graph.py            # LangGraph StateGraph construction
│   ├── nodes.py            # Node functions (initialize, llm_call, tool_node, summarize, routing)
│   └── state.py            # AgentState pydantic model
├── mcp/
│   ├── __init__.py
│   └── client.py           # MCPConnectionManager
└── llm_services/
    ├── agent_llm.py        # LLM initialization per provider
    └── summarizer_llm.py   # Summarizer LLM initialization
```

## Usage

```python
from agent_sdk.agents import BaseAgent

# Local tools only
agent = BaseAgent(tools=[my_tool], system_prompt="...", provider="groq")
result = await agent.arun("Hello", session_id="abcd")

# With remote MCP servers
agent = BaseAgent(
    tools=[],
    mcp_servers={
        "web-search": {"url": "http://localhost:8010/mcp", "transport": "streamable_http"},
    },
    provider="nvidia",
)
result = await agent.arun("Search for X", session_id="abc")
await agent._disconnect_mcp()
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | If using Groq provider | Groq API key |
| `GOOGLE_API_KEY` | If using Gemini provider | Google AI API key |
| `NVIDIA_API_KEY` | If using NVIDIA provider | NVIDIA API key |

## Dependencies

`langchain`, `langgraph`, `langchain-groq`, `langchain-google-genai`, `langchain-nvidia-ai-endpoints`, `langchain-mcp-adapters`
