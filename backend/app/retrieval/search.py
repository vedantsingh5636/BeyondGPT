from app.retrieval.vector_store import create_vector_store


def search_documents(query: str, k: int = 5):
    vector_store = create_vector_store()

    results = vector_store.similarity_search(
        query=query,
        k=k
    )

    return results