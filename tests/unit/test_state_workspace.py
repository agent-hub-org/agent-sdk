from langchain_core.messages import HumanMessage


def test_financial_state_has_workspace_id():
    from agent_sdk.agents.state import FinancialAnalysisState
    s = FinancialAnalysisState(messages=[HumanMessage(content="test")])
    assert s.workspace_id == ""
    assert s.workspace_populated == set()
    assert s.agents_completed == set()


def test_workspace_populated_reducer_union():
    """workspace_populated uses set-union reducer so parallel fan-outs don't overwrite."""
    from agent_sdk.agents.state import _union_sets
    assert _union_sets({"macro"}, {"company_profiling"}) == {"macro", "company_profiling"}
    assert _union_sets(None, {"macro"}) == {"macro"}   # LangGraph initial-state case
    assert _union_sets({"macro"}, None) == {"macro"}


def test_agents_completed_reducer_union():
    from agent_sdk.agents.state import _union_sets
    assert _union_sets({"macro"}, {"sector"}) == {"macro", "sector"}
    assert _union_sets(None, None) == set()
