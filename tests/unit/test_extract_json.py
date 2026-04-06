"""Unit tests for _extract_json in agent_sdk.agents.nodes."""
import pytest
from agent_sdk.agents.nodes import _extract_json


def test_raw_json_object():
    assert _extract_json('{"key": "value"}') == {"key": "value"}


def test_json_in_markdown_fence():
    text = '```json\n{"key": "value"}\n```'
    assert _extract_json(text) == {"key": "value"}


def test_json_in_plain_fence():
    text = '```\n{"key": "value"}\n```'
    assert _extract_json(text) == {"key": "value"}


def test_json_embedded_in_prose():
    text = 'Here is the analysis: {"market_regime": "bull", "confidence": 0.8} as shown.'
    result = _extract_json(text)
    assert result == {"market_regime": "bull", "confidence": 0.8}


def test_multiple_objects_returns_largest():
    # Should return the dict with more keys
    text = 'Small: {"a": 1}. Large: {"x": 1, "y": 2, "z": 3}.'
    result = _extract_json(text)
    assert result == {"x": 1, "y": 2, "z": 3}


def test_malformed_json_returns_none():
    assert _extract_json("this is not json at all") is None


def test_empty_string_returns_none():
    assert _extract_json("") is None


def test_nested_json():
    text = '{"outer": {"inner": "value"}, "count": 5}'
    result = _extract_json(text)
    assert result == {"outer": {"inner": "value"}, "count": 5}
