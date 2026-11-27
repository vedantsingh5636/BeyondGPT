import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pypdf
import pytest
from fastapi import HTTPException
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)
from langchain_core.documents import Document
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from qdrant_client.http.exceptions import ResponseHandlingException

import main as main_module
from app.agents import prompt_utils
from app.api import chat as chat_api
from app.api import upload as upload_api
from app.config import settings
from app.ingestion import github_loader, web_loader
from app.retrieval import vector_store as vs_module
from app.utils import file_utils, rate_limiter


class _FakeUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str | None = None):
        self.filename = filename
        self._content = content
        self.content_type = content_type
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            data = self._content[self._offset :]
            self._offset = len(self._content)
            return data
        data = self._content[self._offset : self._offset + size]
        self._offset += len(data)
        return data


def _run_coro(coro):
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    raise AssertionError("Coroutine suspended unexpectedly during unit test.")


async def _sync_run(func, *args, **kwargs):
    return func(*args, **kwargs)


def _valid_pdf_bytes(text: str = "Secure upload test content.") -> bytes:
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


def test_upload_security_rejects_oversized_spoofed_and_unsafe_files(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(upload_api, "_run_blocking", _sync_run)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "persistent_uploads"))
    monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_MB", 1)

    # 1. Extension check
    with pytest.raises(HTTPException) as exc_ext:
        _run_coro(upload_api.upload_document(_FakeUploadFile("payload.exe", b"MZ...", "application/octet-stream")))
    assert exc_ext.value.status_code == 400

    # 2. MIME mismatch check
    with pytest.raises(HTTPException) as exc_mime:
        _run_coro(upload_api.upload_document(_FakeUploadFile("report.pdf", b"%PDF-1.4", "text/html")))
    assert exc_mime.value.status_code == 400

    # 3. File size limit check (413)
    oversized = b"%PDF-1.4\n" + (b"A" * (1024 * 1024 + 10))
    with pytest.raises(HTTPException) as exc_size:
        _run_coro(upload_api.upload_document(_FakeUploadFile("huge.pdf", oversized, "application/pdf")))
    assert exc_size.value.status_code == 413

    # 4. Magic-byte signature check
    with pytest.raises(HTTPException) as exc_sig:
        _run_coro(upload_api.upload_document(_FakeUploadFile("fake.pdf", b"MZ-not-a-pdf", "application/pdf")))
    assert exc_sig.value.status_code == 400

    # 5. Path traversal filename is sanitized and persisted inside configured UPLOAD_DIR
    saved_paths: list[str] = []
    monkeypatch.setattr(upload_api, "find_document_by_hash", lambda h: None)
    monkeypatch.setattr(
        upload_api,
        "create_vector_store",
        lambda: SimpleNamespace(add_documents=lambda docs, ids=None: ids),
    )
    orig_save = file_utils.save_uploaded_file

    def tracked_save(filename: str, content: bytes) -> str:
        path = orig_save(filename, content)
        saved_paths.append(path)
        return path

    monkeypatch.setattr(upload_api, "save_uploaded_file", tracked_save)

    pdf_bytes = _valid_pdf_bytes()
    res = _run_coro(
        upload_api.upload_document(
            _FakeUploadFile("../../secret/quarterly.pdf", pdf_bytes, "application/pdf")
        )
    )
    assert res["filename"] == "quarterly.pdf"
    assert Path(saved_paths[0]).exists()
    assert Path(saved_paths[0]).parent == Path(settings.UPLOAD_DIR).resolve()


def test_upload_cleans_up_temp_file_on_qdrant_outage(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(upload_api, "_run_blocking", _sync_run)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(upload_api, "find_document_by_hash", lambda h: None)

    def broken_store():
        raise httpx.ConnectError("Qdrant connection refused")

    monkeypatch.setattr(upload_api, "create_vector_store", broken_store)

    pdf_bytes = _valid_pdf_bytes()
    with pytest.raises(HTTPException) as exc_qdrant:
        _run_coro(upload_api.upload_document(_FakeUploadFile("doc.pdf", pdf_bytes, "application/pdf")))

    assert exc_qdrant.value.status_code == 503
    assert "unavailable" in exc_qdrant.value.detail.lower()
    # Uploaded file must be cleaned up when Qdrant insertion fails
    assert list((tmp_path / "uploads").glob("*")) == []


def test_upload_handles_file_save_error_with_cors_and_qdrant_timeout_default(tmp_path: Path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.config import Settings

    assert Settings.model_fields["QDRANT_TIMEOUT_SECONDS"].default == 15.0

    monkeypatch.setattr(upload_api, "_run_blocking", _sync_run)
    monkeypatch.setattr(upload_api, "find_document_by_hash", lambda h: None)

    def failing_save(filename: str, content: bytes) -> str:
        raise PermissionError("Read-only upload directory")

    monkeypatch.setattr(upload_api, "save_uploaded_file", failing_save)

    client = TestClient(main_module.app)
    resp = client.post(
        "/upload",
        headers={"Origin": "http://localhost:5173"},
        files={"file": ("doc.pdf", _valid_pdf_bytes(), "application/pdf")},
    )

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to save uploaded file."
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_ssrf_protection_blocks_internal_targets_and_allows_public_urls(monkeypatch):
    for bad_url in (
        "file:///etc/passwd",
        "http://localhost:8000/health",
        "http://127.0.0.1:6333/collections",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/admin",
        "http://192.168.1.10",
        "https://user:pass@example.com",
        "http://metadata.google.internal/computeMetadata/v1/",
    ):
        with pytest.raises(ValueError):
            web_loader.validate_public_url(bad_url)

    # Hostname resolving to loopback/private IP is blocked
    monkeypatch.setattr(
        web_loader.socket,
        "getaddrinfo",
        lambda host, port, type=None: [(None, None, None, None, ("127.0.0.1", 80))],
    )
    with pytest.raises(ValueError, match="private or internal"):
        web_loader.validate_public_url("https://rebind.example.com/data")

    # Public IP resolution succeeds
    monkeypatch.setattr(
        web_loader.socket,
        "getaddrinfo",
        lambda host, port, type=None: [(None, None, None, None, ("93.184.216.34", 443))],
    )
    assert web_loader.validate_public_url("https://example.com/docs") == "https://example.com/docs"

    # GitHub loader blocks non-github hosts and option-injection branch names
    with pytest.raises(ValueError, match="github.com"):
        github_loader.load_github_repo("https://example.com/repo.git")
    with pytest.raises(ValueError, match="Invalid branch"):
        github_loader.load_github_repo("https://github.com/org/repo", branch="--upload-pack=evil")


def test_rate_limiter_and_chat_error_redaction(monkeypatch):
    rate_limiter.reset_rate_limits()
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)
    req = SimpleNamespace(client=SimpleNamespace(host="10.1.2.3"))

    rate_limiter.enforce_rate_limit(req)
    rate_limiter.enforce_rate_limit(req)
    with pytest.raises(HTTPException) as exc_rate:
        rate_limiter.enforce_rate_limit(req)
    assert exc_rate.value.status_code == 429
    rate_limiter.reset_rate_limits()

    # Oversized chat message rejected with 400
    monkeypatch.setattr(settings, "MAX_CHAT_MESSAGE_LENGTH", 20)
    with pytest.raises(HTTPException) as exc_len:
        chat_api.chat(chat_api.ChatRequest(message="x" * 25))
    assert exc_len.value.status_code == 400

    # Groq timeout returns 504 without exposing secrets
    monkeypatch.setattr(
        chat_api,
        "build_graph",
        lambda: SimpleNamespace(
            invoke=lambda state: (_ for _ in ()).throw(
                httpx.ReadTimeout("timeout with key gsk_secret_12345")
            )
        ),
    )
    with pytest.raises(HTTPException) as exc_timeout:
        chat_api.chat(chat_api.ChatRequest(message="hello"))
    assert exc_timeout.value.status_code == 504
    assert "gsk_" not in exc_timeout.value.detail

    # Qdrant failure in stream_chat_events yields safe error event without leaking secrets
    def failing_retrieval(state):
        raise ResponseHandlingException(RuntimeError(f"Qdrant error {settings.GROQ_API_KEY}"))

    monkeypatch.setattr(chat_api, "retrieval_agent", failing_retrieval)
    events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in chat_api.stream_chat_events("hello", [])
    ]
    assert events[0]["type"] == "error"
    assert settings.GROQ_API_KEY not in events[0]["detail"]
    assert "gsk_" not in events[0]["detail"]


def test_health_endpoint_and_env_example(monkeypatch):
    healthy_client = SimpleNamespace(
        collection_exists=lambda name: True,
        count=lambda collection_name, exact=False: SimpleNamespace(count=12),
    )
    monkeypatch.setattr(
        main_module,
        "check_qdrant_health",
        lambda: vs_module.check_qdrant_health(client=healthy_client),
    )
    res_ok = main_module.health()
    assert res_ok["status"] == "healthy"
    assert res_ok["backend"] == "ok"
    assert res_ok["qdrant"]["status"] == "ok"
    assert res_ok["qdrant"]["points_count"] == 12
    assert res_ok["groq_configured"] is True
    assert settings.GROQ_API_KEY not in json.dumps(res_ok)

    broken_client = SimpleNamespace(
        collection_exists=lambda name: (_ for _ in ()).throw(httpx.ConnectError("down")),
    )
    monkeypatch.setattr(
        main_module,
        "check_qdrant_health",
        lambda: vs_module.check_qdrant_health(client=broken_client),
    )
    res_deg = main_module.health()
    assert res_deg["status"] == "degraded"
    assert res_deg["qdrant"]["status"] == "unavailable"

    env_example = Path(__file__).resolve().parent.parent / ".env.example"
    assert env_example.exists()
    content = env_example.read_text(encoding="utf-8")
    assert "GROQ_API_KEY=your_groq_api_key_here" in content
    assert settings.GROQ_API_KEY not in content


def test_qdrant_client_disables_compatibility_check_and_fails_fast(tmp_path: Path, monkeypatch):
    captured_kwargs: list[dict] = []
    vs_module.get_qdrant_client.cache_clear()
    try:
        def fake_qdrant_cls(**kwargs):
            captured_kwargs.append(kwargs)
            return SimpleNamespace(**kwargs)

        monkeypatch.setattr(vs_module, "QdrantClient", fake_qdrant_cls)
        vs_module.get_qdrant_client()
        assert captured_kwargs[0]["check_compatibility"] is False
        assert captured_kwargs[0]["timeout"] == settings.QDRANT_TIMEOUT_SECONDS
    finally:
        vs_module.get_qdrant_client.cache_clear()

    # Upload returns 503 during duplicate-check outage without writing file to disk
    monkeypatch.setattr(upload_api, "_run_blocking", _sync_run)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(
        upload_api,
        "find_document_by_hash",
        lambda h: (_ for _ in ()).throw(ResponseHandlingException(RuntimeError("refused"))),
    )
    with pytest.raises(HTTPException) as exc_up:
        _run_coro(upload_api.upload_document(_FakeUploadFile("doc.pdf", _valid_pdf_bytes(), "application/pdf")))
    assert exc_up.value.status_code == 503
    assert not (tmp_path / "uploads").exists()

    # /chat returns 503 when Qdrant raises ResponseHandlingException
    monkeypatch.setattr(
        chat_api,
        "build_graph",
        lambda: SimpleNamespace(
            invoke=lambda state: (_ for _ in ()).throw(ResponseHandlingException(RuntimeError("refused")))
        ),
    )
    with pytest.raises(HTTPException) as exc_chat:
        chat_api.chat(chat_api.ChatRequest(message="hello"))
    assert exc_chat.value.status_code == 503
    assert "unavailable" in exc_chat.value.detail.lower()


def test_groq_sdk_errors_are_handled_and_never_leak_keys(monkeypatch):
    fake_req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    fake_resp_401 = httpx.Response(401, request=fake_req)
    fake_resp_429 = httpx.Response(429, request=fake_req)

    cases = [
        (
            AuthenticationError(f"Invalid key {settings.GROQ_API_KEY}", response=fake_resp_401, body=None),
            502,
            "authentication failed",
        ),
        (
            RateLimitError("Rate limit reached", response=fake_resp_429, body=None),
            429,
            "rate-limited",
        ),
        (
            APITimeoutError(request=fake_req),
            504,
            "timed out",
        ),
        (
            APIConnectionError(message="Connection failed gsk_leak_test", request=fake_req),
            503,
            "unavailable",
        ),
    ]

    docs = [Document(page_content="Test context", metadata={"source": "doc.pdf", "page": 1})]
    monkeypatch.setattr(
        chat_api,
        "retrieval_agent",
        lambda state: {"standalone_query": state["query"], "retrieved_documents": docs},
    )

    for exc_obj, expected_status, expected_phrase in cases:
        # Test /chat
        monkeypatch.setattr(
            chat_api,
            "build_graph",
            lambda e=exc_obj: SimpleNamespace(invoke=lambda state: (_ for _ in ()).throw(e)),
        )
        with pytest.raises(HTTPException) as exc_info:
            chat_api.chat(chat_api.ChatRequest(message="Test query"))
        assert exc_info.value.status_code == expected_status
        assert expected_phrase in exc_info.value.detail.lower()
        assert settings.GROQ_API_KEY not in exc_info.value.detail
        assert "gsk_" not in exc_info.value.detail

        # Test /chat/stream
        def broken_stream(_msgs, e=exc_obj):
            raise e

        monkeypatch.setattr(
            chat_api,
            "get_groq_llm",
            lambda fn=broken_stream: SimpleNamespace(stream=fn),
        )
        events = [
            json.loads(line.removeprefix("data: ").strip())
            for line in chat_api.stream_chat_events("Test query", [])
        ]
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert expected_phrase in events[0]["detail"].lower()
        assert settings.GROQ_API_KEY not in events[0]["detail"]
        assert "gsk_" not in events[0]["detail"]

    # Test contextualize_query fallback on GroqError
    monkeypatch.setattr(
        prompt_utils,
        "get_groq_llm",
        lambda: SimpleNamespace(
            invoke=lambda msgs: (_ for _ in ()).throw(APIConnectionError(message="offline", request=fake_req))
        ),
    )
    rewritten = prompt_utils.contextualize_query(
        "What about its margin?",
        [{"role": "user", "content": "finance_2026.pdf revenue"}],
    )
    assert rewritten == "finance_2026.pdf revenue What about its margin?"


def test_upload_limits_reject_10mb_100pages_and_500chunks_without_truncation(tmp_path: Path, monkeypatch):
    assert settings.MAX_UPLOAD_SIZE_MB == 10
    assert settings.MAX_PDF_PAGES == 100
    assert settings.MAX_DOCUMENT_CHUNKS == 500

    monkeypatch.setattr(upload_api, "_run_blocking", _sync_run)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(upload_api, "find_document_by_hash", lambda h: None)

    # 1. File > 10 MB rejected with 413 before hashing/saving/processing
    too_large_bytes = b"%PDF-1.4\n" + (b"0" * (10 * 1024 * 1024 + 1))
    with pytest.raises(HTTPException) as exc_size:
        _run_coro(upload_api.upload_document(_FakeUploadFile("large_11mb.pdf", too_large_bytes, "application/pdf")))
    assert exc_size.value.status_code == 413
    assert "10 MB" in exc_size.value.detail
    assert not (tmp_path / "uploads").exists()

    # 2. PDF with > 100 pages rejected with 413 before extracting page content, and cleaned up
    writer = pypdf.PdfWriter()
    for _ in range(101):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_101_bytes = buf.getvalue()

    with pytest.raises(HTTPException) as exc_pages:
        _run_coro(upload_api.upload_document(_FakeUploadFile("long_101_pages.pdf", pdf_101_bytes, "application/pdf")))
    assert exc_pages.value.status_code == 413
    assert "100 pages" in exc_pages.value.detail
    assert list((tmp_path / "uploads").glob("*")) == []

    # 3. Document producing > 500 chunks rejected with 413 before vector store upsert, and cleaned up
    store_called = False

    def fail_if_called():
        nonlocal store_called
        store_called = True
        return SimpleNamespace()

    monkeypatch.setattr(upload_api, "create_vector_store", fail_if_called)
    monkeypatch.setattr(
        "app.ingestion.ingestion_pipeline.process_document",
        lambda path, source_name=None: [
            Document(page_content=f"Chunk {i}", metadata={"source": "dense.pdf", "page": 1})
            for i in range(501)
        ],
    )
    monkeypatch.setattr("app.ingestion.ingestion_pipeline.clean_documents", lambda docs: docs)
    monkeypatch.setattr("app.ingestion.ingestion_pipeline.split_documents", lambda docs: docs)

    with pytest.raises(HTTPException) as exc_chunks:
        _run_coro(upload_api.upload_document(_FakeUploadFile("many_chunks.pdf", _valid_pdf_bytes(), "application/pdf")))
    assert exc_chunks.value.status_code == 413
    assert "500 chunks" in exc_chunks.value.detail
    assert store_called is False
    assert list((tmp_path / "uploads").glob("*")) == []
