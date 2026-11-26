import logging

from app.agents.prompt_utils import contextualize_query
from app.agents.state import AgentState
from app.retrieval.retrieval_pipeline import retrieve_documents

logger = logging.getLogger(__name__)


def retrieval_agent(state: AgentState) -> dict:
    """Contextualize follow-up queries and retrieve relevant document chunks."""
    query = state["query"]
    chat_history = state.get("chat_history") or []
    search_query = contextualize_query(query, chat_history)

    try:
        retrieved_documents = retrieve_documents(
            query=search_query,
            top_k=5,
        )
    except ValueError as exc:
        if "No documents available for search" in str(exc) or "empty" in str(exc).lower():
            logger.info("Retrieval returned no documents: %s", exc)
            return {
                "standalone_query": search_query,
                "retrieved_documents": [],
            }
        raise

    logger.debug("Retrieved document count: %d", len(retrieved_documents))
    for i, doc in enumerate(retrieved_documents):
        logger.debug(
            "Chunk %d | Source=%s | Page=%s",
            i + 1,
            doc.metadata.get("source"),
            doc.metadata.get("page"),
        )

    return {
        "standalone_query": search_query,
        "retrieved_documents": retrieved_documents,
    }
