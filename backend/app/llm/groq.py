from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import settings


@lru_cache(maxsize=1)
def get_groq_llm() -> ChatGroq:
    """Return a cached ChatGroq client preserving the configured Groq model."""
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        api_key=settings.GROQ_API_KEY,
        timeout=settings.GROQ_TIMEOUT_SECONDS,
        max_retries=2,
    )
