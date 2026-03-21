import os

from langchain_openai import ChatOpenAI


def initialize_azure(temperature: float = 0.3) -> ChatOpenAI:
    """Initialize and return an Azure AI Foundry LLM tuned for summarization tasks."""
    return ChatOpenAI(
        base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature=temperature,
    )
