from contextlib import AbstractContextManager

from fastembed import TextEmbedding
from langchain_community.embeddings.fastembed import (
    FastEmbedEmbeddings as _LangChainFastEmbedEmbeddings,
)

from app.retrieval.model_lifecycle import clear_model, get_or_load_model, model_lease

FastEmbedEmbeddings = _LangChainFastEmbedEmbeddings


def _build_embeddings() -> _LangChainFastEmbedEmbeddings:
    if FastEmbedEmbeddings is _LangChainFastEmbedEmbeddings:
        model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            threads=1,
            enable_cpu_mem_arena=False,
        )
        return _LangChainFastEmbedEmbeddings.model_construct(
            model_name="BAAI/bge-small-en-v1.5",
            max_length=512,
            cache_dir=None,
            threads=1,
            doc_embed_type="default",
            batch_size=4,
            parallel=None,
            providers=None,
            model=model,
        )
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=4,
        threads=1,
    )


def get_embeddings() -> FastEmbedEmbeddings:
    """Return a cached FastEmbed embeddings instance configured for low-memory batching."""
    return get_or_load_model("embeddings", _build_embeddings, evict_other="reranker")


def embeddings_lease() -> AbstractContextManager[None]:
    """Hold an active-use lease on the embeddings model during vector operations."""
    return model_lease("embeddings")


get_embeddings.cache_clear = lambda: clear_model("embeddings", force=True)  # type: ignore[attr-defined]
