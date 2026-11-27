import logging
from contextlib import AbstractContextManager
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.retrieval.model_lifecycle import clear_model, get_or_load_model, model_lease

logger = logging.getLogger(__name__)

ONNX_RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
PYTORCH_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# MS-MARCO MiniLM-L-6-v2 outputs unnormalized logits where relevant matches can
# legitimately be negative (e.g. -1 to -5). Only prune tail items that lag far
# behind the best hit AND fall below a very low logit floor.
_MAX_RELATIVE_SCORE_DROP = 6.5
_LOW_SCORE_PRUNE_FLOOR = -6.0


def _build_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(
        model_name=ONNX_RERANKER_MODEL,
        threads=1,
        enable_cpu_mem_arena=False,
    )


def get_reranker() -> TextCrossEncoder:
    """Return a cached single-threaded FastEmbed ONNX TextCrossEncoder instance."""
    return get_or_load_model("reranker", _build_reranker, evict_other="embeddings")


def reranker_lease() -> AbstractContextManager[None]:
    """Hold an active-use lease on the reranker model during scoring."""
    return model_lease("reranker")


get_reranker.cache_clear = lambda: clear_model("reranker", force=True)  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def get_pytorch_reranker():
    """Lazily load and cache the legacy PyTorch CrossEncoder for parity testing only."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(PYTORCH_RERANKER_MODEL)


def _rank_and_prune_documents(
    documents: list,
    scores: list[float],
    top_k: int,
    prune_irrelevant_tail: bool,
) -> list:
    ranked_documents = sorted(
        zip(documents, scores, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )

    best_score = ranked_documents[0][1]
    selected: list = []

    for rank, (document, score) in enumerate(ranked_documents):
        logger.debug(
            "RERANKER SCORE=%.4f SOURCE=%s CONTENT=%s",
            score,
            document.metadata.get("source"),
            document.page_content[:120],
        )

        # Always retain the highest-scoring candidate; only prune clearly
        # off-topic tail candidates when they fall far below both best_score and floor.
        if (
            prune_irrelevant_tail
            and rank > 0
            and score < _LOW_SCORE_PRUNE_FLOOR
            and (best_score - score) > _MAX_RELATIVE_SCORE_DROP
        ):
            continue

        selected.append(document)
        if len(selected) >= top_k:
            break

    return selected


def _score_documents(reranker, query: str, documents: list) -> list[float]:
    texts = [document.page_content for document in documents]
    if hasattr(reranker, "rerank"):
        # FastEmbed TextCrossEncoder.rerank yields a generator, so materialize explicitly.
        raw_scores = list(reranker.rerank(query, texts))
    else:
        pairs = [(query, text) for text in texts]
        raw_scores = list(reranker.predict(pairs))
    return [float(score) for score in raw_scores]


def rerank_documents(
    query: str,
    documents: list,
    top_k: int = 5,
    prune_irrelevant_tail: bool = True,
) -> list:
    """Rerank candidate documents with cached FastEmbed ONNX TextCrossEncoder."""
    if not documents or top_k <= 0:
        return []

    with reranker_lease():
        reranker = get_reranker()
        scores = _score_documents(reranker, query, documents)
    return _rank_and_prune_documents(documents, scores, top_k, prune_irrelevant_tail)


def rerank_documents_pytorch(
    query: str,
    documents: list,
    top_k: int = 5,
    prune_irrelevant_tail: bool = True,
) -> list:
    """Rerank documents using the legacy PyTorch CrossEncoder for side-by-side parity tests."""
    if not documents or top_k <= 0:
        return []

    reranker = get_pytorch_reranker()
    scores = _score_documents(reranker, query, documents)
    return _rank_and_prune_documents(documents, scores, top_k, prune_irrelevant_tail)
