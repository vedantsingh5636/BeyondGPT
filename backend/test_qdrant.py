from app.retrieval.vector_store import create_vector_store
from langchain_core.documents import Document


def test_qdrant_insertion():
    vector_store = create_vector_store()

    documents = [
        Document(
            page_content="Agentic RAG combines retrieval, AI agents, and language models.",
            metadata={"source": "test_document"}
        )
    ]

    vector_store.add_documents(documents)

    print("Document Inserted Successfully!")


if __name__ == "__main__":
    test_qdrant_insertion()