import os
from langchain_google_genai import ChatGoogleGenerativeAI


def initialize_llm(model: str = "gemini-2.0-flash", temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Initialize and return a Gemini LLM for use as the agent's primary model."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )
