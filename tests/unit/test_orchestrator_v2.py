import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.messages import AIMessage, HumanMessage


async def test_financial_orchestrate_returns_template_name():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="""
```json
{"template_name": "buy_decision", "entities": ["TCS.NS"], "as_of_date": null}
```
    """))

    from agent_sdk.financial.orchestrator import financial_orchestrate
    from agent_sdk.agents.state import FinancialAnalysisState

    state = FinancialAnalysisState(
        messages=[HumanMessage(content="Should I buy TCS?")]
    )

    with patch("agent_sdk.financial.orchestrator._get_orchestrator_llm", return_value=mock_llm):
        result = await financial_orchestrate(None, state)

    assert result["current_template"] == "buy_decision"
    assert "TCS.NS" in result["entities"]
    assert result["workspace_id"] != ""

    # Verify the orchestrator system prompt was sent to the LLM
    call_args = mock_llm.ainvoke.call_args
    messages_sent = call_args[0][0]  # first positional arg
    assert any("financial query router" in getattr(m, "content", "") for m in messages_sent)


async def test_financial_orchestrate_educational_query():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="""
{"template_name": "educational", "entities": [], "as_of_date": null}
    """))

    from agent_sdk.financial.orchestrator import financial_orchestrate
    from agent_sdk.agents.state import FinancialAnalysisState

    state = FinancialAnalysisState(
        messages=[HumanMessage(content="What is P/E ratio?")]
    )

    with patch("agent_sdk.financial.orchestrator._get_orchestrator_llm", return_value=mock_llm):
        result = await financial_orchestrate(None, state)

    assert result["current_template"] == "educational"
    assert result["entities"] == []


async def test_financial_orchestrate_invalid_template_falls_back_to_general_analysis():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="""
{"template_name": "unknown_template", "entities": ["RELIANCE.NS"], "as_of_date": null}
    """))

    from agent_sdk.financial.orchestrator import financial_orchestrate
    from agent_sdk.agents.state import FinancialAnalysisState

    state = FinancialAnalysisState(
        messages=[HumanMessage(content="something unclear")]
    )

    with patch("agent_sdk.financial.orchestrator._get_orchestrator_llm", return_value=mock_llm):
        result = await financial_orchestrate(None, state)

    assert result["current_template"] == "general_analysis"


async def test_financial_orchestrate_workspace_id_is_unique():
    mock_llm = AsyncMock()
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"template_name": "macro_query", "entities": [], "as_of_date": null}'))

    from agent_sdk.financial.orchestrator import financial_orchestrate
    from agent_sdk.agents.state import FinancialAnalysisState

    state = FinancialAnalysisState(messages=[HumanMessage(content="macro outlook")])

    with patch("agent_sdk.financial.orchestrator._get_orchestrator_llm", return_value=mock_llm):
        r1 = await financial_orchestrate(None, state)
        r2 = await financial_orchestrate(None, state)

    assert r1["workspace_id"] != r2["workspace_id"]
