import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA, ChatNVIDIADynamo


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


def initialize_nvidia(model: str = "nvidia/nemotron-3-super-120b-a12b", temperature: float = 0.7) -> ChatNVIDIA:
    """Initialize and return an NVIDIA NIM-hosted LLM for use as the agent's primary model."""
    return ChatNVIDIADynamo(
        model=model,
        api_key=os.environ.get("NVIDIA_API_KEY"),
        # Dynamo-specific hints
        osl=4096,              # I expect ~1000 tokens of financial analysis
        iat=100,               # I want tokens every 100ms
        priority=1,            # Give this agent high priority
        latency_sensitivity=1.0,
        max_completion_tokens=32768 # Minimize time-to-first-token
    )
