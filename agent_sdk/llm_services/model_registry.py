"""
Centralized model catalog and factory.

Each entry maps a stable model_id to its provider, model name, and display label.
The `get_llm()` factory creates a LangChain chat model for any cataloged ID.
"""
import os
import logging

logger = logging.getLogger("agent_sdk.model_registry")

MODEL_CATALOG = {
    # Groq-hosted models
    "groq/gpt-oss-120b": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "label": "GPT-OSS 120B (Groq)",
    },
    "groq/llama-4-scout": {
        "provider": "groq",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "label": "Llama 4 Scout (Groq)",
    },
    "groq/llama-4-maverick": {
        "provider": "groq",
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "label": "Llama 4 Maverick (Groq)",
    },
    # Google Gemini models
    "gemini/gemini-2.5-flash": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "label": "Gemini 2.5 Flash",
    },
    "gemini/gemini-2.5-pro": {
        "provider": "gemini",
        "model": "gemini-2.5-pro-preview-05-06",
        "label": "Gemini 2.5 Pro",
    },
    # NVIDIA NIM models
    "nvidia/nemotron-120b": {
        "provider": "nvidia",
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "label": "Nemotron 120B (NVIDIA)",
    },
}


def get_llm(model_id: str, temperature: float = 0.7):
    """Create a LangChain chat model from a model_id in the catalog."""
    config = MODEL_CATALOG.get(model_id)
    if not config:
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_CATALOG.keys())}")

    provider = config["provider"]
    model = config["model"]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("GROQ_API_KEY"),
            model_kwargs={"parallel_tool_calls": True},
        )
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
        )
    elif provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA, ChatNVIDIADynamo
        return ChatNVIDIADynamo(
            model=model,
            api_key=os.environ.get("NVIDIA_API_KEY"),
            # Dynamo-specific hints
            osl=4096,              # I expect ~1000 tokens of financial analysis
            iat=100,               # I want tokens every 100ms
            priority=1,            # Give this agent high priority
            latency_sensitivity=1.0,
            max_completion_tokens=65536 # Minimize time-to-first-token
        )
    else:
        raise ValueError(f"Unknown provider '{provider}' for model_id '{model_id}'")


def list_models() -> list[dict]:
    """Return the catalog as a list suitable for API responses."""
    return [
        {"id": model_id, "label": config["label"], "provider": config["provider"]}
        for model_id, config in MODEL_CATALOG.items()
    ]
