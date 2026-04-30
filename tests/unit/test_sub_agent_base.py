import pytest
from datetime import datetime, UTC
from agent_sdk.sub_agents.base import SubAgentInput, SubAgentOutput


def test_sub_agent_input_defaults():
    inp = SubAgentInput(query="What does TCS do?", entities=["TCS.NS"])
    assert inp.workspace_context == ""
    assert inp.user_profile == {}


def test_sub_agent_output_defaults():
    out = SubAgentOutput(agent_name="macro", findings="Market is bullish")
    assert out.structured == {}
    assert out.confidence == 1.0
    assert out.cached is False
    assert isinstance(out.computed_at, datetime)


def test_sub_agent_output_round_trip():
    out = SubAgentOutput(
        agent_name="fundamental",
        findings="Revenue grew 15%",
        structured={"revenue_growth": 15.0},
        confidence=0.9,
    )
    d = out.model_dump()
    restored = SubAgentOutput.model_validate(d)
    assert restored.agent_name == "fundamental"
    assert restored.structured["revenue_growth"] == 15.0
