from types import SimpleNamespace

from langchain_core.documents import Document

from app.retrieval import bm25_search as bm25_module
from app.retrieval import embeddings as embeddings_module
from app.retrieval import hybrid_search as hybrid_module
from app.retrieval import model_lifecycle as lifecycle_module
from app.retrieval import reranker as reranker_module
from app.retrieval import retrieval_pipeline as pipeline_module
from app.retrieval import vector_store as vs_module


def test_expensive_models_and_client_are_cached(monkeypatch):
    emb_calls = 0
    rerank_calls = 0
    qdrant_calls = 0

    def fake_fastembed(**kwargs):
        nonlocal emb_calls
        emb_calls += 1
        return SimpleNamespace(**kwargs)

    def fake_text_cross_encoder(**kwargs):
        nonlocal rerank_calls
        rerank_calls += 1
        return SimpleNamespace(**kwargs)

    def fake_qdrant(**kwargs):
        nonlocal qdrant_calls
        qdrant_calls += 1
        return SimpleNamespace(url=kwargs["url"])

    embeddings_module.get_embeddings.cache_clear()
    reranker_module.get_reranker.cache_clear()
    vs_module.get_qdrant_client.cache_clear()

    try:
        monkeypatch.setattr(embeddings_module, "FastEmbedEmbeddings", fake_fastembed)
        monkeypatch.setattr(reranker_module, "TextCrossEncoder", fake_text_cross_encoder)
        monkeypatch.setattr(vs_module, "QdrantClient", fake_qdrant)

        e1 = embeddings_module.get_embeddings()
        e2 = embeddings_module.get_embeddings()
        assert e1 is e2
        assert e1.batch_size == 4
        assert e1.threads == 1
        assert emb_calls == 1

        r1 = reranker_module.get_reranker()
        r2 = reranker_module.get_reranker()
        assert r1 is r2
        assert r1.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"
        assert r1.threads == 1
        assert rerank_calls == 1

        q1 = vs_module.get_qdrant_client()
        q2 = vs_module.get_qdrant_client()
        assert q1 is q2
        assert qdrant_calls == 1
    finally:
        embeddings_module.get_embeddings.cache_clear()
        reranker_module.get_reranker.cache_clear()
        vs_module.get_qdrant_client.cache_clear()


def test_bm25_preprocessing_caching_and_invalidation():
    tokens = bm25_module.preprocess_bm25_text("Error ERR-4092 in FastAPI, v1.5 by Dr. Rao (2026).")
    assert "err-4092" in tokens
    assert "4092" in tokens
    assert "fastapi" in tokens
    assert "v1.5" in tokens
    assert "2026" in tokens
    assert "rao" in tokens

    docs = [
        Document(
            page_content="Service returned ERR-4092 during SHA-256 verification.",
            metadata={"document_id": "d1", "chunk_index": 0, "source": "spec.pdf"},
        ),
        Document(
            page_content="Lead architect Dr. Kavita Rao approved the 2026 roadmap.",
            metadata={"document_id": "d2", "chunk_index": 0, "source": "roadmap.docx"},
        ),
        Document(
            page_content="General onboarding checklist for new hires.",
            metadata={"document_id": "d3", "chunk_index": 0, "source": "onboarding.pdf"},
        ),
    ]

    bm25_module.invalidate_bm25_cache()
    r1 = bm25_module.create_bm25_retriever(docs, k=2)
    r2 = bm25_module.create_bm25_retriever(docs, k=1)
    assert r1.vectorizer is r2.vectorizer
    assert r1.k == 2
    assert r2.k == 1

    hits = bm25_module.bm25_search(r2, "ERR-4092")
    assert len(hits) == 1
    assert hits[0].metadata["source"] == "spec.pdf"

    bm25_module.invalidate_bm25_cache()
    r3 = bm25_module.create_bm25_retriever(docs, k=2)
    assert r3.vectorizer is not r1.vectorizer


def test_get_all_documents_caches_until_count_changes_or_invalidated(monkeypatch):
    scroll_calls = 0
    current_records = [
        SimpleNamespace(
            id="1",
            payload={"page_content": "First chunk", "metadata": {"source": "a.pdf", "document_id": "d1"}},
        )
    ]

    class _CountingClient:
        def collection_exists(self, name):
            return True

        def count(self, collection_name, exact=False):
            return SimpleNamespace(count=len(current_records))

        def scroll(self, **kwargs):
            nonlocal scroll_calls
            scroll_calls += 1
            return list(current_records), None

    fake_client = _CountingClient()
    vs_module.invalidate_document_cache()
    monkeypatch.setattr(vs_module, "get_qdrant_client", lambda: fake_client)

    docs_1 = vs_module.get_all_documents()
    docs_2 = vs_module.get_all_documents()
    assert len(docs_1) == 1
    assert len(docs_2) == 1
    assert scroll_calls == 1

    # Adding a record updates Qdrant count and triggers refresh
    current_records.append(
        SimpleNamespace(
            id="2",
            payload={"page_content": "Second chunk", "metadata": {"source": "b.pdf", "document_id": "d2"}},
        )
    )
    docs_3 = vs_module.get_all_documents()
    assert len(docs_3) == 2
    assert scroll_calls == 2

    # Explicit invalidation forces fresh scroll even if count is unchanged
    vs_module.invalidate_document_cache()
    docs_4 = vs_module.get_all_documents()
    assert len(docs_4) == 2
    assert scroll_calls == 3


def test_reranker_handles_negative_logits_without_false_drops(monkeypatch):
    docs = [
        Document(page_content="Chunk A", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Chunk B", metadata={"source": "b.pdf", "page": 2}),
        Document(page_content="Chunk C", metadata={"source": "c.pdf", "page": 3}),
    ]

    # Case 1: All negative logits within normal relevant spread via generator -> all retained
    monkeypatch.setattr(
        reranker_module,
        "get_reranker",
        lambda: SimpleNamespace(
            rerank=lambda query, texts: (score for score in (-2.8, -1.4, -3.5))
        ),
    )
    ranked_neg = reranker_module.rerank_documents("query", docs, top_k=3)
    assert [d.page_content for d in ranked_neg] == ["Chunk B", "Chunk A", "Chunk C"]
    assert ranked_neg[0].metadata == {"source": "b.pdf", "page": 2}

    # Case 2: Clear off-topic tail (-10.2) pruned when lagging far behind +3.1
    monkeypatch.setattr(
        reranker_module,
        "get_reranker",
        lambda: SimpleNamespace(
            rerank=lambda query, texts: (score for score in (3.1, -1.9, -10.2))
        ),
    )
    ranked_pruned = reranker_module.rerank_documents("query", docs, top_k=3)
    assert [d.page_content for d in ranked_pruned] == ["Chunk A", "Chunk B"]

    # Case 3: Empty document list is handled safely
    assert reranker_module.rerank_documents("query", [], top_k=5) == []


def test_multi_document_exact_keyword_and_filename_scoping(monkeypatch):
    doc_fin_1 = Document(
        page_content="Q3 revenue reached 42.5 million USD across EMEA.",
        metadata={"source": "finance_2026.pdf", "filename": "finance_2026.pdf", "document_id": "doc-fin", "page": 1, "chunk_index": 0},
    )
    doc_eng_1 = Document(
        page_content="Microservice gateway uses error code ERR-7712 for rate limiting.",
        metadata={"source": "engineering_spec.pdf", "filename": "engineering_spec.pdf", "document_id": "doc-eng", "page": 4, "chunk_index": 0},
    )
    doc_hr_1 = Document(
        page_content="Dr. Kavita Rao serves as VP of People Operations.",
        metadata={"source": "hr_policy.docx", "filename": "hr_policy.docx", "document_id": "doc-hr", "page": "N/A", "chunk_index": 0},
    )
    all_docs = [doc_fin_1, doc_eng_1, doc_hr_1]

    captured_filters: list = []

    class _FakeVectorStore:
        def similarity_search(self, query: str, k: int = 5, filter=None):
            captured_filters.append(filter)
            if filter is not None:
                return [doc_fin_1]
            # Simulate dense search ranking finance_2026.pdf highest even for an exact-code query
            return [doc_fin_1, doc_hr_1, doc_eng_1][:k]

    monkeypatch.setattr(hybrid_module, "get_all_documents", lambda: all_docs)
    monkeypatch.setattr(hybrid_module, "create_vector_store", lambda: _FakeVectorStore())
    monkeypatch.setattr(pipeline_module, "get_all_documents", lambda: all_docs)
    monkeypatch.setattr(
        reranker_module,
        "get_reranker",
        lambda: SimpleNamespace(
            rerank=lambda query, texts: (
                4.0
                if (
                    ("ERR-7712" in query and "ERR-7712" in text)
                    or ("revenue" in query and "revenue" in text)
                )
                else -8.5
                for text in texts
            )
        ),
    )

    # Exact keyword query retrieves engineering_spec.pdf and prunes unrelated docs
    kw_results = pipeline_module.retrieve_documents("What is ERR-7712?", top_k=3)
    assert len(kw_results) == 1
    assert kw_results[0].metadata["source"] == "engineering_spec.pdf"
    assert kw_results[0].metadata["page"] == 4
    assert kw_results[0].metadata["document_id"] == "doc-eng"

    # Filename-targeted query passes filter and isolates finance_2026.pdf
    scoped_results = hybrid_module.hybrid_search("What does finance_2026.pdf say?", k=3)
    assert captured_filters[-1] is not None
    assert all(doc.metadata["filename"] == "finance_2026.pdf" for doc in scoped_results)


def test_summary_query_retrieves_multi_page_and_multi_doc_coverage(monkeypatch):
    assert pipeline_module.classify_query_intent("Summarize the uploaded PDFs") == "summary"
    assert pipeline_module.classify_query_intent("Give me an overview of report.pdf") == "summary"
    assert pipeline_module.classify_query_intent("What is the Q3 margin?") == "specific"

    docs = [
        Document(
            page_content=f"Doc A page {page} details.",
            metadata={"source": "doc_a.pdf", "filename": "doc_a.pdf", "document_id": "doc-a", "page": page, "chunk_index": page - 1},
        )
        for page in range(1, 7)
    ] + [
        Document(
            page_content=f"Doc B section {idx} details.",
            metadata={"source": "doc_b.docx", "filename": "doc_b.docx", "document_id": "doc-b", "page": "N/A", "chunk_index": idx},
        )
        for idx in range(4)
    ]

    class _FakeVectorStore:
        def similarity_search(self, query: str, k: int = 5, filter=None):
            return docs[:2]

    monkeypatch.setattr(hybrid_module, "get_all_documents", lambda: docs)
    monkeypatch.setattr(hybrid_module, "create_vector_store", lambda: _FakeVectorStore())
    monkeypatch.setattr(pipeline_module, "get_all_documents", lambda: docs)
    monkeypatch.setattr(
        reranker_module,
        "get_reranker",
        lambda: SimpleNamespace(rerank=lambda query, texts: (-2.0 for _ in texts)),
    )

    summary_results = pipeline_module.retrieve_documents("Summarize all uploaded documents", top_k=5)

    # Summary expands beyond 5 chunks, covers both documents, and avoids duplicates
    assert len(summary_results) >= 6
    contents = [doc.page_content for doc in summary_results]
    assert len(contents) == len(set(contents))
    sources = {doc.metadata["source"] for doc in summary_results}
    assert sources == {"doc_a.pdf", "doc_b.docx"}


def test_model_lifecycle_evicts_idle_sibling_but_protects_active_leases(monkeypatch):
    embeddings_module.get_embeddings.cache_clear()
    reranker_module.get_reranker.cache_clear()

    try:
        monkeypatch.setattr(embeddings_module, "FastEmbedEmbeddings", lambda **kw: SimpleNamespace(kind="emb", **kw))
        monkeypatch.setattr(reranker_module, "TextCrossEncoder", lambda **kw: SimpleNamespace(kind="rerank", **kw))

        # Idle sibling is evicted when loading the other model
        e1 = embeddings_module.get_embeddings()
        assert lifecycle_module.get_model_state("embeddings") == (True, 0)
        r1 = reranker_module.get_reranker()
        assert lifecycle_module.get_model_state("embeddings") == (False, 0)
        assert lifecycle_module.get_model_state("reranker") == (True, 0)
        assert r1.kind == "rerank"

        # Active lease protects reranker from eviction during concurrent embeddings load
        with reranker_module.reranker_lease():
            assert lifecycle_module.clear_model("reranker") is False
            e2 = embeddings_module.get_embeddings()
            assert e2 is not e1
            assert lifecycle_module.get_model_state("reranker") == (True, 1)
            assert lifecycle_module.get_model_state("embeddings") == (True, 0)

        # Active lease protects embeddings from eviction during concurrent reranker load
        with embeddings_module.embeddings_lease():
            assert lifecycle_module.clear_model("embeddings") is False
            r2 = reranker_module.get_reranker()
            assert r2 is r1
            assert lifecycle_module.get_model_state("embeddings") == (True, 1)

        # Once lease exits, idle sibling can be evicted normally
        reranker_module.get_reranker.cache_clear()
        _ = reranker_module.get_reranker()
        assert lifecycle_module.get_model_state("embeddings") == (False, 0)
    finally:
        embeddings_module.get_embeddings.cache_clear()
        reranker_module.get_reranker.cache_clear()


def test_bm25_cache_is_bounded_to_three_entries():
    bm25_module.invalidate_bm25_cache()
    try:
        for i in range(5):
            docs = [
                Document(
                    page_content=f"Unique BM25 corpus entry {i}",
                    metadata={"document_id": f"doc-{i}", "chunk_index": 0},
                )
            ]
            bm25_module.create_bm25_retriever(docs, k=1)
        assert len(bm25_module._bm25_cache) == 3
    finally:
        bm25_module.invalidate_bm25_cache()


def test_low_memory_onnx_session_options_trim_and_post_ingest_unload(monkeypatch):
    from pathlib import Path

    import onnxruntime as ort

    from app.api import upload as upload_api

    captured: dict[str, object] = {}
    tokenizer_called = False

    def fake_inference_session(model_path, providers=None, sess_options=None):
        captured["model_path"] = model_path
        captured["providers"] = providers
        captured["sess_options"] = sess_options
        return SimpleNamespace(kind="ort_session")

    def fake_ensure_tokenizer():
        nonlocal tokenizer_called
        tokenizer_called = True

    inner = SimpleNamespace(
        _model_dir=Path("."),
        model_description=SimpleNamespace(model_file="model_optimized.onnx", additional_files=[]),
        model=None,
        _ensure_tokenizer=fake_ensure_tokenizer,
    )

    monkeypatch.setattr(lifecycle_module.ort, "InferenceSession", fake_inference_session)
    lifecycle_module.configure_low_memory_onnx_session(inner)

    opts = captured["sess_options"]
    assert captured["providers"] == ["CPUExecutionProvider"]
    assert opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    assert opts.intra_op_num_threads == 1
    assert opts.inter_op_num_threads == 1
    assert opts.enable_cpu_mem_arena is False
    assert opts.enable_mem_pattern is False
    assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert tokenizer_called is True
    assert isinstance(lifecycle_module.trim_process_memory(), bool)

    # Verify _ingest_and_store_file unloads idle embeddings, but preserves active concurrent lease
    embeddings_module.get_embeddings.cache_clear()
    try:
        monkeypatch.setattr(embeddings_module, "FastEmbedEmbeddings", lambda **kw: SimpleNamespace(kind="emb", **kw))
        monkeypatch.setattr(
            upload_api,
            "ingest_document",
            lambda *a, **kw: [Document(page_content="Hello", metadata={"source": "a.pdf", "page": 1})],
        )

        class _FakeStore:
            def add_documents(self, docs, ids=None, batch_size=4):
                _ = embeddings_module.get_embeddings()
                return ids or []

        monkeypatch.setattr(upload_api, "create_vector_store", lambda: _FakeStore())

        upload_api._ingest_and_store_file("dummy.pdf", "a.pdf", "h" * 64, "doc-1")
        assert lifecycle_module.get_model_state("embeddings") == (False, 0)

        with embeddings_module.embeddings_lease():
            upload_api._ingest_and_store_file("dummy.pdf", "a.pdf", "h" * 64, "doc-1")
            assert lifecycle_module.get_model_state("embeddings") == (True, 1)
    finally:
        embeddings_module.get_embeddings.cache_clear()
