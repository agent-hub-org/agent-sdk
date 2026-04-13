import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def mock_llm():
    """A mock LLM that returns a pre-configured AIMessage."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"result": "mocked"}'))
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def mock_tool():
    """A mock tool that returns a fixed string."""
    tool = AsyncMock()
    tool.name = "mock_tool"
    tool.ainvoke = AsyncMock(return_value="mocked tool result")
    return tool


@pytest.fixture
def base_agent_state():
    """A minimal AgentState for testing."""
    from agent_sdk.agents.state import AgentState
    return AgentState(
        messages=[HumanMessage(content="test query")],
    )


@pytest.fixture
def financial_state():
    """A minimal FinancialAnalysisState for testing."""
    from agent_sdk.agents.state import FinancialAnalysisState
    return FinancialAnalysisState(
        messages=[HumanMessage(content="Analyse RELIANCE.NS")],
    )
