import os

from langchain_openai import ChatOpenAI

from agent_sdk.config import settings


def initialize_azure() -> ChatOpenAI:
    """Initialize and return an Azure AI Foundry LLM for use as the agent's primary model."""
    return ChatOpenAI(
        base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )