import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def initialize_gemini(model: str = "gemini-2.5-flash", temperature: float = 0.3) -> ChatGoogleGenerativeAI:
    """Initialize and return a Gemini LLM tuned for summarization tasks."""
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=os.environ.get("GEMINI_API_KEY"),
    )


def initialize_groq(model: str = "openai/gpt-oss-20b", temperature: float = 0.3) -> ChatGroq:
    """Initialize and return a Groq-hosted LLM tuned for summarization tasks."""
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.environ.get("GROQ_API_KEY"),
    )


def initialize_nvidia(model: str = "nvidia/nemotron-3-super-120b-a12b", temperature: float = 0.3) -> ChatNVIDIA:
    """Initialize and return an NVIDIA NIM-hosted LLM tuned for summarization tasks."""
    return ChatNVIDIA(
        model=model,
        temperature=temperature,
        api_key=os.environ.get("NVIDIA_API_KEY"),
    )
