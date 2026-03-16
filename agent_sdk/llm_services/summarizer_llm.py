import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def initialize_gemini(model: str = "gemini-2.5-flash", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Initialize and return a Gemini LLM tuned for summarization tasks."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    )


def initialize_groq(model: str = "llama-3.3-70b-versatile", temperature: float = 0.3) -> ChatGroq:
    """Initialize and return a Groq-hosted Llama LLM tuned for summarization tasks."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY"),
    )
