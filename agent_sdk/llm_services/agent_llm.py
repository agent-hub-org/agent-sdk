import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def initialize_gemini(model: str = "gemini-2.5-flash", temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    """Initialize and return a Gemini LLM for use as the agent's primary model."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    )


def initialize_groq(model: str = "openai/gpt-oss-120b", temperature: float = 0.7) -> ChatGroq:
    """Initialize and return a Groq-hosted LLM for use as the agent's primary model."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY"),
        model_kwargs={"parallel_tool_calls": True}
    )
