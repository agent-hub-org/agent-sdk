"""Unit tests for _build_phase_return helper in agent_sdk.agents.nodes."""
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage


def make_state(raw_fallback_count=0, validation_warnings=None, iteration=0):
    state = MagicMock()
    state.raw_fallback_count = raw_fallback_count
    state.validation_warnings = validation_warnings or []
    state.iteration = iteration
    state.phase_iterations = {}
    return state


def test_build_phase_return_with_data():
    from agent_sdk.agents.nodes import _build_phase_return
    response = AIMessage(content='{"key": "value"}')
    state = make_state()
    data = {"key": "value"}
    result = _build_phase_return(response, "causal_analysis", data, state, "causal_analysis")

    assert result["causal_analysis"] == data
    assert result["raw_fallback_count"] == 0
    assert result["iteration"] == 1
    assert result["phase_iterations"] == {"causal_analysis": 1}
    assert response in result["messages"]


def test_build_phase_return_without_data_increments_fallback():
    from agent_sdk.agents.nodes import _build_phase_return
    response = AIMessage(content="some text that is not json")
    state = make_state(raw_fallback_count=1)
    result = _build_phase_return(response, "sector_findings", {}, state, "sector_analysis")

    assert result["sector_findings"] == {"raw_analysis": "some text that is not json"}
    assert result["raw_fallback_count"] == 2  # was 1, incremented by 1


def test_build_phase_return_extra_fields_merged():
    from agent_sdk.agents.nodes import _build_phase_return
    response = AIMessage(content='{"k": 1}')
    state = make_state()
    data = {"k": 1}
    extra = {"validation_warnings": ["warning1"]}
    result = _build_phase_return(response, "risk_assessment", data, state, "risk_assessment", extra=extra)

    assert result["validation_warnings"] == ["warning1"]
    assert result["risk_assessment"] == data


def test_build_phase_return_phase_iterations_accumulate():
    from agent_sdk.agents.nodes import _build_phase_return
    response = AIMessage(content='{}')
    state = make_state(iteration=2)
    state.phase_iterations = {"company_analysis": 3}
    result = _build_phase_return(response, "company_analysis", {"x": 1}, state, "company_analysis")

    assert result["phase_iterations"]["company_analysis"] == 4  # 3 + 1
    assert result["iteration"] == 3  # 2 + 1
