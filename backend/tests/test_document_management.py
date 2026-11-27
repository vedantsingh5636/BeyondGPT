import io
from pathlib import Path
from types import SimpleNamespace

import pypdf
from docx import Document as WordDocument
from langchain_core.documents import Document
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.api import upload as upload_api
from app.retrieval import hybrid_search as hybrid_module
from app.retrieval import vector_store as vs_module
from app.utils.file_utils import (
    compute_file_hash,
    generate_chunk_id,
    generate_document_id,
)


class _FakeUploadFile:
    """Async upload double avoiding Starlette's anyio threadpool during unit tests."""

    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


def _run_coro(coro):
    """Drive a coroutine whose awaits resolve immediately without an event loop."""
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise AssertionError("Coroutine suspended unexpectedly during unit test.")


def _build_pdf_bytes(text: str) -> bytes:
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 740 Td ({text}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _build_docx_bytes(heading: str, body: str) -> bytes:
    word_doc = WordDocument()
    word_doc.add_heading(heading, level=1)
    word_doc.add_paragraph(body)
    buf = io.BytesIO()
    word_doc.save(buf)
    return buf.getvalue()


class _FakeQdrantClient:
    """In-memory Qdrant double that preserves existing records across calls."""

    def __init__(self, initial_records: list[dict] | None = None, collection_exists: bool = True):
        self._exists = collection_exists
        self.created_collections: list[str] = []
        self.deleted_collections: list[str] = []
        self.payload_indexes: list[str] = []
        self.records: dict[str, dict] = {}
        for idx, rec in enumerate(initial_records or []):
            rec_id = str(rec.get("id", f"legacy-{idx}"))
            self.records[rec_id] = {
                "id": rec_id,
                "payload": {
                    "page_content": rec["page_content"],
                    "metadata": dict(rec.get("metadata", {})),
                },
            }

    def collection_exists(self, collection_name: str) -> bool:
        return self._exists

    def create_collection(self, collection_name: str, vectors_config=None) -> None:
        self._exists = True
        self.created_collections.append(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        self.deleted_collections.append(collection_name)
        self.records.clear()

    def create_payload_index(self, collection_name: str, field_name: str, field_schema=None) -> None:
        self.payload_indexes.append(field_name)

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> list[str]:
        assigned_ids: list[str] = []
        for idx, doc in enumerate(documents):
            point_id = ids[idx] if ids else f"auto-{len(self.records)}"
            self.records[point_id] = {
                "id": point_id,
                "payload": {
                    "page_content": doc.page_content,
                    "metadata": dict(doc.metadata),
                },
            }
            assigned_ids.append(point_id)
        return assigned_ids

    def scroll(
        self,
        collection_name: str,
        scroll_filter=None,
        limit: int = 1000,
        offset=None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ):
        items = list(self.records.values())
        if scroll_filter and getattr(scroll_filter, "must", None):
            cond = scroll_filter.must[0]
            target_hash = cond.match.value
            items = [
                item
                for item in items
                if item["payload"].get("metadata", {}).get("file_hash") == target_hash
            ]

        start = int(offset or 0)
        batch = [SimpleNamespace(id=item["id"], payload=item["payload"]) for item in items[start : start + limit]]
        next_offset = start + limit if start + limit < len(items) else None
        return batch, next_offset


def test_sha256_hash_and_stable_ids_are_deterministic():
    content_a = b"beyondgpt-sample-pdf-bytes"
    content_b = b"beyondgpt-other-pdf-bytes"

    hash_a1 = compute_file_hash(content_a)
    hash_a2 = compute_file_hash(content_a)
    hash_b = compute_file_hash(content_b)

    assert len(hash_a1) == 64
    assert hash_a1 == hash_a2
    assert hash_a1 != hash_b

    doc_id_1 = generate_document_id(hash_a1)
    doc_id_2 = generate_document_id(hash_a2)
    assert doc_id_1 == doc_id_2
    assert doc_id_1 != generate_document_id(hash_b)
    assert generate_chunk_id(doc_id_1, 0) == generate_chunk_id(doc_id_2, 0)
    assert generate_chunk_id(doc_id_1, 0) != generate_chunk_id(doc_id_1, 1)


def test_ensure_collection_preserves_existing_user_documents():
    legacy_record = {
        "id": "legacy-point-1",
        "page_content": "Legacy user document content.",
        "metadata": {"source": "uploads/legacy.pdf", "page": 1},
    }
    fake_client = _FakeQdrantClient(initial_records=[legacy_record], collection_exists=True)

    vs_module.ensure_collection_exists(fake_client)

    assert fake_client.created_collections == []
    assert fake_client.deleted_collections == []
    assert "legacy-point-1" in fake_client.records

    docs = vs_module.get_all_documents(client=fake_client)
    assert len(docs) == 1
    assert docs[0].page_content == "Legacy user document content."


def test_get_all_documents_paginates_and_handles_missing_collection(monkeypatch):
    missing_client = _FakeQdrantClient(collection_exists=False)
    assert vs_module.get_all_documents(client=missing_client) == []

    # Simulate 3 records scrolled in pages of 2
    records = [
        {"id": f"p{i}", "page_content": f"Chunk {i}", "metadata": {"source": f"doc{i}.pdf"}}
        for i in range(3)
    ]
    fake_client = _FakeQdrantClient(initial_records=records, collection_exists=True)

    original_scroll = fake_client.scroll
    monkeypatch.setattr(
        fake_client,
        "scroll",
        lambda **kwargs: original_scroll(**{**kwargs, "limit": 2}),
    )

    docs = vs_module.get_all_documents(client=fake_client)
    assert [d.page_content for d in docs] == ["Chunk 0", "Chunk 1", "Chunk 2"]


def test_duplicate_upload_is_skipped_and_metadata_is_preserved(tmp_path: Path, monkeypatch):
    legacy_record = {
        "id": "legacy-1",
        "page_content": "Pre-existing Qdrant document chunk.",
        "metadata": {"source": "existing_notes.pdf", "page": 1},
    }
    fake_client = _FakeQdrantClient(initial_records=[legacy_record], collection_exists=True)

    saved_calls: list[str] = []

    def fake_save(filename: str, content: bytes) -> str:
        saved_calls.append(filename)
        target = tmp_path / f"stored_{len(saved_calls)}_{Path(filename).name}"
        target.write_bytes(content)
        return str(target)

    async def _sync_run(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(upload_api, "_run_blocking", _sync_run)
    monkeypatch.setattr(upload_api, "save_uploaded_file", fake_save)
    monkeypatch.setattr(upload_api, "find_document_by_hash", lambda h: vs_module.find_document_by_hash(h, client=fake_client))
    monkeypatch.setattr(upload_api, "create_vector_store", lambda: fake_client)

    pdf_bytes = _build_pdf_bytes("Quarterly revenue grew by 25 percent.")
    expected_hash = compute_file_hash(pdf_bytes)
    expected_doc_id = generate_document_id(expected_hash)

    # First upload succeeds and persists metadata
    first_file = _FakeUploadFile(filename="quarterly_2026.pdf", content=pdf_bytes)
    res1 = _run_coro(upload_api.upload_document(first_file))

    assert res1["filename"] == "quarterly_2026.pdf"
    assert res1["duplicate"] is False
    assert res1["chunks_created"] == 1
    assert res1["document_id"] == expected_doc_id
    assert res1["file_hash"] == expected_hash
    assert len(saved_calls) == 1

    # Duplicate upload with identical bytes (even under a new filename) is skipped
    duplicate_file = _FakeUploadFile(filename="quarterly_copy.pdf", content=pdf_bytes)
    res2 = _run_coro(upload_api.upload_document(duplicate_file))

    assert res2["duplicate"] is True
    assert res2["chunks_created"] == 0
    assert res2["document_id"] == expected_doc_id
    assert res2["file_hash"] == expected_hash
    assert len(saved_calls) == 1  # No duplicate file written to disk
    assert len(fake_client.records) == 2  # Legacy + 1 new chunk only

    # Second distinct document (DOCX) is stored alongside legacy & PDF documents
    docx_bytes = _build_docx_bytes("Architecture", "BeyondGPT uses Qdrant and Groq.")
    docx_file = _FakeUploadFile(filename="architecture.docx", content=docx_bytes)
    res3 = _run_coro(upload_api.upload_document(docx_file))

    assert res3["duplicate"] is False
    assert res3["chunks_created"] == 1
    assert res3["document_id"] != expected_doc_id
    assert len(fake_client.records) == 3

    # Existing legacy document was never deleted and new chunks keep original metadata
    all_docs = vs_module.get_all_documents(client=fake_client)
    assert any(d.page_content == "Pre-existing Qdrant document chunk." for d in all_docs)

    pdf_chunks = [d for d in all_docs if d.metadata.get("file_hash") == expected_hash]
    assert len(pdf_chunks) == 1
    assert pdf_chunks[0].metadata["source"] == "quarterly_2026.pdf"
    assert pdf_chunks[0].metadata["filename"] == "quarterly_2026.pdf"
    assert pdf_chunks[0].metadata["document_id"] == expected_doc_id
    assert pdf_chunks[0].metadata["page"] == 1
    assert pdf_chunks[0].metadata["chunk_index"] == 0


def test_hybrid_search_retrieves_across_multiple_docs_without_duplicates(monkeypatch):
    doc_a_chunk1 = Document(
        page_content="Project Alpha uses FastAPI and PostgreSQL for transactional storage.",
        metadata={"source": "alpha.pdf", "filename": "alpha.pdf", "document_id": "doc-a", "file_hash": "hash-a", "chunk_index": 0, "page": 1},
    )
    doc_a_legacy_dup = Document(
        page_content="Project Alpha uses FastAPI and PostgreSQL for transactional storage.",
        metadata={"source": "uploads/old_uuid.pdf", "page": 1},
    )
    doc_b_chunk1 = Document(
        page_content="Project Beta relies on Qdrant vector search and BM25 keyword token XQ99.",
        metadata={"source": "beta.docx", "filename": "beta.docx", "document_id": "doc-b", "file_hash": "hash-b", "chunk_index": 0, "page": "N/A"},
    )
    doc_c_chunk1 = Document(
        page_content="Annual leave policy allows 20 paid vacation days per calendar year.",
        metadata={"source": "hr.pdf", "filename": "hr.pdf", "document_id": "doc-c", "file_hash": "hash-c", "chunk_index": 0, "page": 2},
    )

    all_docs = [doc_a_chunk1, doc_a_legacy_dup, doc_b_chunk1, doc_c_chunk1]

    class _FakeVectorStore:
        def similarity_search(self, query: str, k: int = 5):
            # Dense search returns duplicate chunks from doc A first
            return [doc_a_chunk1, doc_a_legacy_dup, doc_c_chunk1][:k]

    monkeypatch.setattr(hybrid_module, "get_all_documents", lambda: all_docs)
    monkeypatch.setattr(hybrid_module, "create_vector_store", lambda: _FakeVectorStore())

    results = hybrid_module.hybrid_search("XQ99 keyword", k=3)

    contents = [doc.page_content for doc in results]
    # Zero duplicate chunks despite dense returning identical content twice
    assert len(contents) == len(set(contents))
    # BM25 exact match from second document (beta.docx) is preserved via RRF
    assert any("XQ99" in content for content in contents)
    # Multi-document metadata remains intact
    sources = {doc.metadata.get("source") for doc in results}
    assert "beta.docx" in sources
    assert "alpha.pdf" in sources


def test_find_document_by_hash_ordering_preserves_collection_and_avoids_duplicates():
    missing_client = _FakeQdrantClient(collection_exists=False)
    assert vs_module.find_document_by_hash("abc123", client=missing_client) is None
    assert missing_client.created_collections == []
    assert missing_client.payload_indexes == []

    existing_record = {
        "id": "existing-1",
        "page_content": "Existing content",
        "metadata": {"source": "existing.pdf", "file_hash": "hash-existing"},
    }
    ordered_events: list[str] = []

    class _OrderedFakeQdrantClient(_FakeQdrantClient):
        def create_payload_index(self, collection_name: str, field_name: str, field_schema=None) -> None:
            ordered_events.append(f"index:{field_name}")
            super().create_payload_index(collection_name, field_name, field_schema)

        def scroll(self, *args, **kwargs):
            ordered_events.append("scroll")
            return super().scroll(*args, **kwargs)

    existing_client = _OrderedFakeQdrantClient(
        initial_records=[existing_record],
        collection_exists=True,
    )
    vs_module._initialized_client_ids.discard(id(existing_client))

    found = vs_module.find_document_by_hash("hash-existing", client=existing_client)
    assert found is not None
    assert found["source"] == "existing.pdf"
    assert existing_client.created_collections == []
    assert existing_client.deleted_collections == []
    assert "existing-1" in existing_client.records
    assert ordered_events[:4] == [
        "index:metadata.file_hash",
        "index:metadata.document_id",
        "index:metadata.filename",
        "index:metadata.source",
    ]
    assert ordered_events[4] == "scroll"

    # Subsequent lookup and ensure_collection_exists reuse initialization cache without duplicate indexes
    assert vs_module.find_document_by_hash("hash-missing", client=existing_client) is None
    vs_module.ensure_collection_exists(existing_client)
    assert len(existing_client.payload_indexes) == 4


def test_add_documents_batched_uses_64_and_falls_back_for_simple_doubles():
    captured: list[tuple] = []

    class _BatchedStore:
        def add_documents(self, docs, ids=None, batch_size=64):
            captured.append((len(docs), ids, batch_size))
            return ids or []

    class _LegacyStore:
        def add_documents(self, docs, ids=None):
            captured.append((len(docs), ids, None))
            return ids or []

    docs = [Document(page_content="Hello", metadata={"source": "a.pdf"})]
    vs_module.add_documents_batched(_BatchedStore(), docs, ids=["id-1"])
    vs_module.add_documents_batched(_LegacyStore(), docs, ids=["id-2"])

    assert captured == [(1, ["id-1"], 64), (1, ["id-2"], None)]
