import re
from threading import Lock

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SPLIT_SUBTOKEN_RE = re.compile(r"[._-]+")

_bm25_lock = Lock()
_bm25_cache: dict[tuple, BM25Retriever] = {}
_MAX_BM25_CACHE_ENTRIES = 3


def preprocess_bm25_text(text: str) -> list[str]:
    """Tokenize text for BM25 preserving technical terms, numbers, and sub-tokens."""
    lowered = text.lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(lowered):
        tokens.append(match)
        if any(sep in match for sep in (".", "_", "-")):
            for sub in _SPLIT_SUBTOKEN_RE.split(match):
                if sub and sub != match:
                    tokens.append(sub)
    return tokens


def _corpus_fingerprint(documents: list[Document]) -> tuple:
    """Build a lightweight fingerprint so cached BM25 stays fresh across corpus edits."""
    return (
        len(documents),
        tuple(
            (
                (doc.metadata or {}).get("document_id") or (doc.metadata or {}).get("source"),
                (doc.metadata or {}).get("chunk_index"),
                hash(doc.page_content),
            )
            for doc in documents
        ),
    )


def invalidate_bm25_cache() -> None:
    """Clear the cached BM25 retriever index."""
    with _bm25_lock:
        _bm25_cache.clear()


def create_bm25_retriever(documents: list[Document], k: int = 5) -> BM25Retriever:
    """Return a cached or newly built BM25Retriever for the provided documents."""
    if not documents:
        raise ValueError("No documents available for BM25 search.")

    key = _corpus_fingerprint(documents)
    with _bm25_lock:
        cached = _bm25_cache.get(key)
        if cached is None:
            cached = BM25Retriever.from_documents(
                documents,
                preprocess_func=preprocess_bm25_text,
            )
            # Keep bounded cache size to avoid unbounded memory growth on filtered subsets.
            while len(_bm25_cache) >= _MAX_BM25_CACHE_ENTRIES:
                oldest_key = next(iter(_bm25_cache))
                _bm25_cache.pop(oldest_key, None)
            _bm25_cache[key] = cached

    return cached.model_copy(update={"k": min(k, len(documents))})


def bm25_search(retriever: BM25Retriever, query: str):
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    return retriever.invoke(query)
