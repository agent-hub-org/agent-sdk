import os

from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel


def initialize_azure(temperature: float = 0.7) -> AzureAIOpenAIApiChatModel:
    """Initialize and return an Azure AI Foundry LLM for use as the agent's primary model."""
    return AzureAIOpenAIApiChatModel(
        endpoint=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
        credential=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
        model="Llama-4-Maverick-17B-128E-Instruct-FP8",
        temperature=temperature,
    )
