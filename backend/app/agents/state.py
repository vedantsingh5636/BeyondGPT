from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    query: str
    standalone_query: str | None
    chat_history: list[dict[str, str]]
    documents: list[Document]
    retrieved_documents: list[Document]
    answer: str | None
    source: str | None
    next_agent: str | None
