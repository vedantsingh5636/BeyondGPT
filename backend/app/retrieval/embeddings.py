from functools import lru_cache

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


@lru_cache(maxsize=1)
def get_embeddings() -> FastEmbedEmbeddings:
    """Return a cached FastEmbed embeddings instance configured for low-memory batching."""
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=64,
    )
