from pathlib import Path

from qdrant_client.models import FieldCondition, Filter, MatchAny

from app.retrieval.bm25_search import create_bm25_retriever
from app.retrieval.embeddings import embeddings_lease
from app.retrieval.vector_store import create_vector_store, get_all_documents

_RRF_K = 60


def _chunk_dedup_key(document) -> str:
    """Build a normalized deduplication key for a retrieved chunk."""
    normalized_text = " ".join(document.page_content.split())
    if normalized_text:
        return normalized_text
    metadata = document.metadata or {}
    return f"{metadata.get('document_id') or metadata.get('source')}:{metadata.get('chunk_index')}"


def _extract_target_sources(query: str, documents: list) -> list[str]:
    """Identify uploaded filenames explicitly referenced in the user query."""
    lowered_query = query.lower()
    known_sources: set[str] = set()

    for doc in documents:
        metadata = doc.metadata or {}
        for key in ("filename", "source"):
            val = metadata.get(key)
            if isinstance(val, str) and val.strip():
                known_sources.add(val.strip())

    matched: list[str] = []
    for src in sorted(known_sources):
        name = Path(src).name.lower()
        stem = Path(src).stem.lower()
        if (len(name) >= 4 and name in lowered_query) or (
            len(stem) >= 5 and stem in lowered_query
        ):
            matched.append(src)

    return matched


def _matches_target_sources(document, target_sources: set[str]) -> bool:
    metadata = document.metadata or {}
    return (
        metadata.get("filename") in target_sources
        or metadata.get("source") in target_sources
    )


def _dense_search_with_optional_filter(
    vector_store,
    query: str,
    k: int,
    target_sources: list[str],
) -> list:
    if not target_sources:
        return vector_store.similarity_search(query=query, k=k)

    qdrant_filter = Filter(
        should=[
            FieldCondition(key="metadata.filename", match=MatchAny(any=target_sources)),
            FieldCondition(key="metadata.source", match=MatchAny(any=target_sources)),
        ]
    )
    try:
        return vector_store.similarity_search(query=query, k=k, filter=qdrant_filter)
    except TypeError:
        candidates = vector_store.similarity_search(query=query, k=max(k * 2, 20))
        target_set = set(target_sources)
        return [doc for doc in candidates if _matches_target_sources(doc, target_set)][:k]


def hybrid_search(query: str, k: int = 5):
    """Retrieve and fuse dense and BM25 candidates across all documents without duplicates."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    # Get documents from Qdrant (cached when unchanged)
    documents = get_all_documents()

    if not documents:
        raise ValueError("No documents available for search.")

    target_sources = _extract_target_sources(query, documents)
    if target_sources:
        target_set = set(target_sources)
        scoped_documents = [
            doc for doc in documents if _matches_target_sources(doc, target_set)
        ]
        if not scoped_documents:
            scoped_documents = documents
            target_sources = []
    else:
        scoped_documents = documents

    candidate_k = min(len(scoped_documents), max(k * 2, 12))

    # Dense semantic search
    with embeddings_lease():
        vector_store = create_vector_store()
        dense_results = _dense_search_with_optional_filter(
            vector_store,
            query=query,
            k=candidate_k,
            target_sources=target_sources,
        )

    # BM25 keyword search (reuses cached index per corpus)
    bm25_retriever = create_bm25_retriever(
        scoped_documents,
        k=candidate_k,
    )
    keyword_results = bm25_retriever.invoke(query)

    # Fuse dense + BM25 rankings using Reciprocal Rank Fusion and deduplicate
    scores: dict[str, float] = {}
    unique_docs: dict[str, object] = {}

    for results in (dense_results, keyword_results):
        seen_in_list: set[str] = set()
        for rank, document in enumerate(results, start=1):
            key = _chunk_dedup_key(document)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            if key not in unique_docs:
                unique_docs[key] = document

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [unique_docs[key] for key in ranked_keys[:k]]
