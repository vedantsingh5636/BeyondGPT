import re

from app.retrieval.hybrid_search import (
    _chunk_dedup_key,
    _extract_target_sources,
    _matches_target_sources,
    hybrid_search,
)
from app.retrieval.reranker import rerank_documents
from app.retrieval.vector_store import get_all_documents

_SUMMARY_QUERY_RE = re.compile(
    r"\b("
    r"summarize|summarise|summary|overview|tl;?dr|highlights|"
    r"main\s+(points|topics|ideas)|key\s+(points|takeaways|findings)|"
    r"what\s+is\s+(this|the)\s+(document|pdf|docx|file|report|paper)\s+about"
    r")\b",
    re.IGNORECASE,
)


def classify_query_intent(query: str) -> str:
    """Classify a retrieval query as 'summary' or 'specific'."""
    if _SUMMARY_QUERY_RE.search(query):
        return "summary"
    return "specific"


def _select_representative_chunks(documents: list, max_chunks: int) -> list:
    """Select spread-out chunks across documents and pages for summary queries."""
    if not documents or max_chunks <= 0:
        return []

    by_doc: dict[str, list] = {}
    for doc in documents:
        metadata = doc.metadata or {}
        doc_key = str(
            metadata.get("document_id")
            or metadata.get("filename")
            or metadata.get("source")
            or "default"
        )
        by_doc.setdefault(doc_key, []).append(doc)

    # Sort each document's chunks by (page, chunk_index) to cover intro -> body -> end.
    per_doc_sampled: list[list] = []
    per_doc_quota = max(2, max_chunks // len(by_doc) + 1)

    for doc_chunks in by_doc.values():
        ordered = sorted(
            doc_chunks,
            key=lambda d: (
                int((d.metadata or {}).get("page"))
                if isinstance((d.metadata or {}).get("page"), int)
                else 0,
                int((d.metadata or {}).get("chunk_index"))
                if isinstance((d.metadata or {}).get("chunk_index"), int)
                else 0,
            ),
        )
        if len(ordered) <= per_doc_quota:
            per_doc_sampled.append(ordered)
        else:
            step = (len(ordered) - 1) / (per_doc_quota - 1)
            indices = sorted({round(i * step) for i in range(per_doc_quota)})
            per_doc_sampled.append([ordered[idx] for idx in indices])

    # Interleave across documents so multi-doc summaries stay balanced.
    selected: list = []
    max_len = max(len(group) for group in per_doc_sampled)
    for pos in range(max_len):
        for group in per_doc_sampled:
            if pos < len(group):
                selected.append(group[pos])
                if len(selected) >= max_chunks:
                    return selected
    return selected


def retrieve_documents(query: str, top_k: int = 5):
    """Retrieve and rerank relevant chunks for both specific and summary queries."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    intent = classify_query_intent(query)

    if intent == "summary":
        summary_top_k = max(top_k, 8)
        search_results = hybrid_search(query=query, k=max(summary_top_k * 2, 16))

        all_docs = get_all_documents()
        target_sources = _extract_target_sources(query, all_docs)
        scoped_docs = (
            [d for d in all_docs if _matches_target_sources(d, set(target_sources))]
            if target_sources
            else all_docs
        )
        representative = _select_representative_chunks(
            scoped_docs or all_docs,
            max_chunks=summary_top_k,
        )

        # Merge representative multi-page chunks with hybrid search hits without duplicates.
        merged: list = []
        seen: set[str] = set()
        for doc in representative + search_results:
            key = _chunk_dedup_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return rerank_documents(
            query=query,
            documents=merged,
            top_k=summary_top_k,
            prune_irrelevant_tail=False,
        )

    # Specific query: fetch wider candidate pool then rerank with safe tail pruning.
    candidate_k = max(top_k * 3, 15)
    search_results = hybrid_search(
        query=query,
        k=candidate_k,
    )

    return rerank_documents(
        query=query,
        documents=search_results,
        top_k=top_k,
        prune_irrelevant_tail=True,
    )
