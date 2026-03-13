import os
from langchain_google_genai import ChatGoogleGenerativeAI


def initialize_llm(model: str = "gemini-2.0-flash", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Initialize and return a Gemini LLM tuned for summarization tasks."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )
