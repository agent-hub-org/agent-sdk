import os

from langchain_openai import ChatOpenAI


def initialize_azure() -> ChatOpenAI:
    """Initialize and return an Azure AI Foundry LLM for use as the agent's primary model."""
    temperature = float(os.getenv("AGENT_LLM_TEMPERATURE", "0.7"))
    model = os.getenv("AGENT_LLM_MODEL", "Llama-4-Maverick-17B-128E-Instruct-FP8")
    timeout = float(os.getenv("AGENT_LLM_TIMEOUT", "120.0"))
    max_retries = int(os.getenv("AGENT_LLM_MAX_RETRIES", "3"))

    return ChatOpenAI(
        base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
