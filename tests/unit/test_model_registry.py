import os
import importlib
from unittest.mock import MagicMock, patch


def test_get_llm_uses_settings_timeout(monkeypatch):
    monkeypatch.setenv("AZURE_AI_FOUNDRY_ENDPOINT", "http://fake")
    monkeypatch.setenv("AZURE_AI_FOUNDRY_API_KEY", "fake-key")
    monkeypatch.setenv("AGENT_SDK_LLM_TIMEOUT", "42.0")

    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    # Reload config and model_registry to pick up new env vars
    import agent_sdk.config as config_mod
    importlib.reload(config_mod)

    from agent_sdk.llm_services import model_registry
    importlib.reload(model_registry)
    model_registry._LLM_CACHE.clear()

    with patch("langchain_openai.ChatOpenAI", side_effect=fake_chat_openai):
        model_registry.get_llm("azure/gpt-5-nano")

    assert captured, "ChatOpenAI was never called — mock did not intercept the constructor"
    assert captured["timeout"] == 42.0, f"Expected 42.0, got {captured['timeout']}"


def test_get_llm_uses_settings_max_retries(monkeypatch):
    monkeypatch.setenv("AZURE_AI_FOUNDRY_ENDPOINT", "http://fake")
    monkeypatch.setenv("AZURE_AI_FOUNDRY_API_KEY", "fake-key")
    monkeypatch.setenv("AGENT_SDK_LLM_MAX_RETRIES", "7")

    captured = {}

    def fake_chat_openai(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    # Reload config and model_registry to pick up new env vars
    import agent_sdk.config as config_mod
    importlib.reload(config_mod)

    from agent_sdk.llm_services import model_registry
    importlib.reload(model_registry)
    model_registry._LLM_CACHE.clear()

    with patch("langchain_openai.ChatOpenAI", side_effect=fake_chat_openai):
        model_registry.get_llm("azure/gpt-5-nano")

    assert captured, "ChatOpenAI was never called — mock did not intercept the constructor"
    assert captured["max_retries"] == 7, f"Expected 7, got {captured['max_retries']}"


def test_list_models_excludes_hidden():
    from agent_sdk.llm_services.model_registry import list_models
    models = list_models()
    ids = [m["id"] for m in models]
    assert "azure/financial-synthesis" not in ids
    assert "azure/financial-risk" not in ids


def test_list_models_includes_warning_when_set():
    from agent_sdk.llm_services.model_registry import list_models
    models = {m["id"]: m for m in list_models()}
    assert "azure/gpt-oss-120b" in models
    assert "warning" in models["azure/gpt-oss-120b"]
    assert "Tool calls" in models["azure/gpt-oss-120b"]["warning"]


def test_list_models_no_spurious_warning():
    from agent_sdk.llm_services.model_registry import list_models
    models = {m["id"]: m for m in list_models()}
    assert "warning" not in models.get("azure/gpt-5-nano", {})


def test_gpt_5_4_mini_in_catalog():
    from agent_sdk.llm_services.model_registry import MODEL_CATALOG
    assert "azure/gpt-5.4-mini" in MODEL_CATALOG
    assert MODEL_CATALOG["azure/gpt-5.4-mini"]["model"] == "gpt-5.4-mini"
