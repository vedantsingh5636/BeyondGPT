from langchain_groq import ChatGroq
from app.config import settings


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=settings.GROQ_API_KEY
)


def test_groq_connection():
    response = llm.invoke(
        "Say hello and introduce yourself in one sentence."
    )

    return response.content