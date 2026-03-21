"""
Centralized model catalog and factory.

Each entry maps a stable model_id to its provider, model name, and display label.
The `get_llm()` factory creates a LangChain chat model for any cataloged ID.
Set `hidden: True` on entries that are internal pipeline slots and must not appear in the UI.
"""
import os
import logging

logger = logging.getLogger("agent_sdk.model_registry")

MODEL_CATALOG = {
    # Azure AI Foundry — primary model
    "azure/llama-4-maverick": {
        "provider": "Azure AI Foundry",
        "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "label": "Llama 4 Maverick",
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

_DEFAULT_MODEL_ID = "azure/llama-4-maverick"


def get_llm(model_id: str, temperature: float = 0.7):
    """Create a LangChain chat model from a model_id in the catalog.

    Falls back to the default model with a warning if model_id is unknown,
    so stale client-side model IDs never crash the stream.
    """
    config = MODEL_CATALOG.get(model_id)
    if not config:
        logger.warning(
            "Unknown model_id '%s' — falling back to %s. Available: %s",
            model_id, _DEFAULT_MODEL_ID, list(MODEL_CATALOG.keys()),
        )
        config = MODEL_CATALOG[_DEFAULT_MODEL_ID]

    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
    return AzureAIOpenAIApiChatModel(
        endpoint=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        credential=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
        model=config["model"],
        temperature=temperature,
    )


def list_models() -> list[dict]:
    """Return the catalog as a list suitable for API responses.

    Entries with hidden=True are internal pipeline slots and excluded from the list.
    """
    return [
        {"id": model_id, "label": config["label"], "provider": config["provider"]}
        for model_id, config in MODEL_CATALOG.items()
        if not config.get("hidden")
    ]
