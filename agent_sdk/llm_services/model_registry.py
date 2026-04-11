"""
Centralized model catalog and factory.

Each entry maps a stable model_id to its provider, model name, and display label.
The `get_llm()` factory creates a LangChain chat model for any cataloged ID.
Set `hidden: True` on entries that are internal pipeline slots and must not appear in the UI.
"""
import os
import logging
import threading
import httpx

logger = logging.getLogger("agent_sdk.model_registry")

_LLM_CACHE = {}
_LLM_CACHE_LOCK = threading.Lock()

MODEL_CATALOG = {
    # Azure AI Foundry — primary model
    "azure/llama-4-maverick": {
        "provider": "Azure AI Foundry",
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "label": "Llama 4 Maverick",
    },

    # Azure AI Foundry — GPT-4o mini (tool calling + prompt caching)
    "azure/gpt-5-nano": {
        "provider": "Azure AI Foundry",
        "model": "gpt-5-nano",
        "label": "GPT-5 nano",
    },

    # Azure AI Foundry — GPT-4.1 mini (latest GPT-4.1 series, strong reasoning + tool calling)
    "azure/gpt-5.4-nano": {
        "provider": "Azure AI Foundry",
        "model": "gpt-5.4-nano",
        "label": "GPT-5.4 nano",
    },

    # Azure AI Foundry — GPT-OSS-120B (Apache 2.0, o4-mini-level claims)
    # NOTE: Tool calling may 400 due to Harmony format incompatibility with LangChain
    # (GitHub issues #32425, #32885, #34751). Works for non-tool conversational turns.
    "azure/gpt-oss-120b": {
        "provider": "Azure AI Foundry",
        "model": "gpt-oss-120b",
        "label": "GPT-OSS 120B",
    },

    # Fine-tuned model slots for financial reasoning pipeline phases.
    # These slots are ready for fine-tuned model weights once training is complete.
    # The cognitive pipeline can use model_id override per-phase via state.model_id.
    # hidden=True keeps them out of the UI model picker.
    "azure/financial-synthesis": {
        "provider": "Azure AI Foundry",
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",  # placeholder — swap with fine-tuned model
        "label": "Financial Synthesis (Fine-tuned)",
        "hidden": True,
    },
    "azure/financial-risk": {
        "provider": "Azure AI Foundry",
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",  # placeholder — swap with fine-tuned model
        "label": "Financial Risk Assessment (Fine-tuned)",
        "hidden": True,
    },
}

_DEFAULT_MODEL_ID = "azure/gpt-5-nano"


def get_llm(model_id: str, temperature: float = 0.7):
    """Create a LangChain chat model from a model_id in the catalog.

    Falls back to the default model with a warning if model_id is unknown,
    so stale client-side model IDs never crash the stream.
    All models share the same Azure AI Foundry endpoint.
    """
    config = MODEL_CATALOG.get(model_id)
    if not config:
        logger.warning(
            "Unknown model_id '%s' — falling back to %s. Available: %s",
            model_id, _DEFAULT_MODEL_ID, list(MODEL_CATALOG.keys()),
        )
        config = MODEL_CATALOG[_DEFAULT_MODEL_ID]

    cache_key = (model_id, temperature)
    with _LLM_CACHE_LOCK:
        if cache_key not in _LLM_CACHE:
            from langchain_openai import ChatOpenAI
            
            # Using custom http_async_client to prevent connection pool exhaustion across sessions
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=50, max_connections=100)
            )

            _LLM_CACHE[cache_key] = ChatOpenAI(
                base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
                api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
                model=config["model"],
                temperature=temperature,
                timeout=float(os.getenv("AGENT_LLM_TIMEOUT", "120.0")),
                max_retries=int(os.getenv("AGENT_LLM_MAX_RETRIES", "3")),
                http_async_client=http_client,
            )

        return _LLM_CACHE[cache_key]


def list_models() -> list[dict]:
    """Return the catalog as a list suitable for API responses.

    Entries with hidden=True are internal pipeline slots and excluded from the list.
    """
    return [
        {"id": model_id, "label": config["label"], "provider": config["provider"]}
        for model_id, config in MODEL_CATALOG.items()
        if not config.get("hidden")
    ]
