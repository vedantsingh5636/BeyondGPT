import logging
from functools import lru_cache
from threading import Lock

import httpx
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)

from app.config import settings
from app.retrieval.bm25_search import invalidate_bm25_cache
from app.retrieval.embeddings import get_embeddings
from app.utils.file_utils import generate_chunk_id

logger = logging.getLogger(__name__)
COLLECTION_NAME = settings.QDRANT_COLLECTION_NAME

_cache_lock = Lock()
_initialized_client_ids: set[int] = set()
_doc_cache: dict[str, object] = {
    "count": None,
    "documents": [],
}


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Return a cached persistent Qdrant client using configured connection settings."""
    if settings.QDRANT_API_KEY:
        return QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=settings.QDRANT_TIMEOUT_SECONDS,
            check_compatibility=False,
        )
    return QdrantClient(
        url=settings.QDRANT_URL,
        timeout=settings.QDRANT_TIMEOUT_SECONDS,
        check_compatibility=False,
    )


def invalidate_document_cache() -> None:
    """Invalidate cached scrolled documents and BM25 index after ingestion."""
    with _cache_lock:
        _doc_cache["count"] = None
        _doc_cache["documents"] = []
    invalidate_bm25_cache()


def check_qdrant_health(client: QdrantClient | None = None) -> dict[str, object]:
    """Check Qdrant connectivity and collection availability without exposing secrets."""
    try:
        qdrant = client or get_qdrant_client()
        exists = bool(qdrant.collection_exists(COLLECTION_NAME))
        points_count = None
        if exists:
            points_count = qdrant.count(collection_name=COLLECTION_NAME, exact=False).count
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "collection_exists": exists,
            "points_count": points_count,
        }
    except (ApiException, httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return {
            "status": "unavailable",
            "collection": COLLECTION_NAME,
            "collection_exists": False,
            "points_count": None,
        }


_PAYLOAD_INDEX_FIELDS = (
    "metadata.file_hash",
    "metadata.document_id",
    "metadata.filename",
    "metadata.source",
)


def _ensure_payload_indexes(client: QdrantClient) -> None:
    """Create missing metadata keyword payload indexes without overwriting existing ones."""
    for field_name in _PAYLOAD_INDEX_FIELDS:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except (ApiException, ValueError, RuntimeError) as exc:
            logger.debug("Skipping payload index creation for %s: %s", field_name, exc)


def ensure_collection_exists(client: QdrantClient) -> None:
    """Create the Qdrant collection and metadata payload indexes if missing.

    Never deletes or overwrites an existing collection or its stored documents.
    """
    client_id = id(client)
    with _cache_lock:
        if client_id in _initialized_client_ids:
            return

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE,
            ),
        )

    _ensure_payload_indexes(client)

    with _cache_lock:
        _initialized_client_ids.add(client_id)


def create_vector_store() -> QdrantVectorStore:
    client = get_qdrant_client()
    ensure_collection_exists(client)

    embeddings = get_embeddings()

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def add_documents_batched(
    vector_store,
    documents: list[Document],
    *,
    ids: list[str] | None = None,
    batch_size: int = 64,
) -> list[str]:
    """Upsert documents in bounded batches aligned with low-memory FastEmbed settings."""
    try:
        return vector_store.add_documents(documents, ids=ids, batch_size=batch_size)
    except TypeError:
        return vector_store.add_documents(documents, ids=ids)


def find_document_by_hash(
    file_hash: str,
    client: QdrantClient | None = None,
) -> dict | None:
    """Look up an already-ingested document in Qdrant by its SHA-256 file hash."""
    qdrant = client or get_qdrant_client()
    if not qdrant.collection_exists(COLLECTION_NAME):
        return None

    ensure_collection_exists(qdrant)

    records, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.file_hash",
                    match=MatchValue(value=file_hash),
                )
            ]
        ),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None

    payload = records[0].payload or {}
    return payload.get("metadata", {})


def enrich_chunks_metadata(
    chunks: list[Document],
    *,
    filename: str,
    file_hash: str,
    document_id: str,
    saved_path: str | None = None,
) -> tuple[list[Document], list[str]]:
    """Attach stable document metadata and deterministic point IDs to every chunk."""
    enriched: list[Document] = []
    chunk_ids: list[str] = []

    for idx, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        metadata["source"] = filename
        metadata["filename"] = filename
        metadata["document_id"] = document_id
        metadata["file_hash"] = file_hash
        metadata["chunk_index"] = idx
        if saved_path is not None:
            metadata["saved_path"] = saved_path

        enriched.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )
        chunk_ids.append(generate_chunk_id(document_id, idx))

    return enriched, chunk_ids


def _scroll_all_documents(qdrant: QdrantClient) -> list[Document]:
    documents: list[Document] = []
    next_page = None

    while True:
        records, next_page = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,
            offset=next_page,
            with_payload=True,
            with_vectors=False,
        )

        for record in records:
            payload = record.payload or {}
            documents.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
            )

        if next_page is None:
            break

    return documents


def get_all_documents(client: QdrantClient | None = None) -> list[Document]:
    """Return all stored document chunks from Qdrant, using count-validated caching."""
    qdrant = client or get_qdrant_client()
    if not qdrant.collection_exists(COLLECTION_NAME):
        return []

    # Bypass shared cache when an explicit client double is passed in tests.
    if client is not None:
        return _scroll_all_documents(qdrant)

    try:
        current_count = qdrant.count(collection_name=COLLECTION_NAME, exact=False).count
    except (ApiException, ValueError, RuntimeError):
        current_count = None

    with _cache_lock:
        if current_count is not None and _doc_cache["count"] == current_count:
            return list(_doc_cache["documents"])  # type: ignore[arg-type]

    documents = _scroll_all_documents(qdrant)

    with _cache_lock:
        _doc_cache["count"] = current_count if current_count is not None else len(documents)
        _doc_cache["documents"] = list(documents)

    return documents
