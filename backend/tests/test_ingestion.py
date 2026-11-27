from pathlib import Path
from unittest.mock import MagicMock

import pdfplumber
import pypdf
import pytest
from docx import Document as WordDocument
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.api import upload as upload_api
from app.ingestion.clean_documents import clean_documents
from app.ingestion.docx_loader import load_docx
from app.ingestion.ingestion_pipeline import ingest_document
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.text_cleaner import clean_text
from app.retrieval.chunking import split_documents


def _write_synthetic_pdf(path: Path, page_ops: list[bytes]) -> Path:
    """Create a valid PDF file where each entry in page_ops is a content stream."""
    writer = pypdf.PdfWriter()
    for ops in page_ops:
        page = writer.add_blank_page(width=612, height=792)
        if ops:
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
            stream.set_data(ops)
            page[NameObject("/Contents")] = writer._add_object(stream)

    with path.open("wb") as f:
        writer.write(f)
    return path


def test_clean_text():
    result = clean_text("  Hello     World   ")

    assert result == "Hello World"


def test_clean_documents():
    documents = [
        Document(
            page_content="  Agentic   RAG  ",
            metadata={"source": "test.pdf"},
        )
    ]

    result = clean_documents(documents)

    assert result[0].page_content == "Agentic RAG"
    assert result[0].metadata["source"] == "test.pdf"


def test_load_pdf_extracts_text_tables_and_1_based_pages(tmp_path: Path):
    page1_ops = (
        b"BT /F1 12 Tf 50 740 Td (Intro paragraph text.) Tj ET\n"
        b"50 660 200 60 re S\n"
        b"50 690 m 250 690 l S\n"
        b"150 660 m 150 720 l S\n"
        b"BT /F1 10 Tf 60 700 Td (Name) Tj ET\n"
        b"BT /F1 10 Tf 160 700 Td (Role) Tj ET\n"
        b"BT /F1 10 Tf 60 670 Td (Alice) Tj ET\n"
        b"BT /F1 10 Tf 160 670 Td (Admin) Tj ET\n"
        b"BT /F1 12 Tf 50 620 Td (Closing note.) Tj ET"
    )
    page2_ops = b""  # Simulates an image-only / scanned / blank page
    page3_ops = b"BT /F1 12 Tf 50 740 Td (Third page summary.) Tj ET"

    pdf_path = _write_synthetic_pdf(tmp_path / "sample.pdf", [page1_ops, page2_ops, page3_ops])

    docs = load_pdf(str(pdf_path), source_name="original_report.pdf")

    assert len(docs) == 2
    assert docs[0].metadata["source"] == "original_report.pdf"
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 3
    assert docs[0].metadata["empty_pages"] == [2]
    assert docs[0].metadata["table_count"] == 1

    assert "Intro paragraph text." in docs[0].page_content
    assert "| Name | Role |\n| --- | --- |\n| Alice | Admin |" in docs[0].page_content
    assert "Closing note." in docs[0].page_content
    # Ensure table cells were not duplicated as loose prose above/below the table
    assert docs[0].page_content.count("Alice") == 1

    assert docs[1].metadata["page"] == 3
    assert "Third page summary." in docs[1].page_content


def test_load_pdf_rejects_scanned_or_empty_pdf(tmp_path: Path):
    pdf_path = _write_synthetic_pdf(tmp_path / "scanned.pdf", [b"", b""])

    with pytest.raises(ValueError, match="scanned or image-only"):
        load_pdf(str(pdf_path))


def test_load_pdf_rejects_corrupt_pdf(tmp_path: Path):
    corrupt_path = tmp_path / "corrupt.pdf"
    corrupt_path.write_bytes(b"%PDF-1.4\nnot-a-real-pdf-body")

    with pytest.raises(ValueError, match="corrupt or invalid"):
        load_pdf(str(corrupt_path))


def test_load_pdf_enforces_page_limit_before_processing(tmp_path: Path, monkeypatch):
    writer = pypdf.PdfWriter()
    for _ in range(4):
        writer.add_blank_page(width=200, height=200)
    pdf_path = tmp_path / "four_pages.pdf"
    with pdf_path.open("wb") as f:
        writer.write(f)

    extracted = False

    def spy_extract(page):
        nonlocal extracted
        extracted = True
        return "text", 0

    monkeypatch.setattr("app.ingestion.pdf_loader._extract_page_content", spy_extract)

    with pytest.raises(ValueError, match="maximum allowed limit of 3 pages"):
        load_pdf(str(pdf_path), source_name="four_pages.pdf", max_pages=3)
    assert extracted is False


def test_load_docx_extracts_headings_paragraphs_and_tables(tmp_path: Path):
    docx_path = tmp_path / "temp_uuid.docx"
    word_doc = WordDocument()
    word_doc.add_heading("Main Title", level=1)
    word_doc.add_paragraph("First paragraph explaining the system.")
    word_doc.add_heading("Metrics", level=2)

    table = word_doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Score"
    table.cell(1, 0).text = "Recall"
    table.cell(1, 1).text = "0.95"

    word_doc.add_paragraph("Final remarks.")
    word_doc.save(str(docx_path))

    docs = load_docx(str(docx_path), source_name="spec.docx")

    assert len(docs) == 1
    content = docs[0].page_content
    assert docs[0].metadata["source"] == "spec.docx"
    assert docs[0].metadata["page"] == "N/A"
    assert docs[0].metadata["file_type"] == "docx"
    assert docs[0].metadata["table_count"] == 1
    assert "# Main Title" in content
    assert "## Metrics" in content
    assert "| Metric | Score |\n| --- | --- |\n| Recall | 0.95 |" in content
    assert content.index("## Metrics") < content.index("| Metric | Score |") < content.index("Final remarks.")


def test_load_docx_rejects_empty_and_corrupt_docx(tmp_path: Path):
    empty_path = tmp_path / "empty.docx"
    WordDocument().save(str(empty_path))

    with pytest.raises(ValueError, match="No extractable text"):
        load_docx(str(empty_path))

    corrupt_path = tmp_path / "corrupt.docx"
    corrupt_path.write_bytes(b"not a zip docx file")

    with pytest.raises(ValueError, match="corrupt or invalid"):
        load_docx(str(corrupt_path))


def test_split_documents_preserves_metadata_and_keeps_tables_intact():
    table_md = (
        "| ID | Value |\n"
        "| --- | --- |\n"
        "| A1 | Alpha |\n"
        "| B2 | Beta |\n"
        "| C3 | Gamma |"
    )
    prose = " ".join(["Word"] * 120)
    doc = Document(
        page_content=f"{prose}\n\n{table_md}",
        metadata={"source": "manual.pdf", "page": 4},
    )

    chunks = split_documents([doc], chunk_size=300, chunk_overlap=40)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["source"] == "manual.pdf"
        assert chunk.metadata["page"] == 4

    table_chunks = [c for c in chunks if "| ID | Value |" in c.page_content]
    assert len(table_chunks) == 1
    assert "| C3 | Gamma |" in table_chunks[0].page_content


def test_split_documents_splits_large_table_with_repeated_headers():
    rows = "\n".join(f"| Row{i} | Value{i} |" for i in range(1, 25))
    large_table = f"| Key | Val |\n| --- | --- |\n{rows}"
    doc = Document(
        page_content=large_table,
        metadata={"source": "big_table.pdf", "page": 2},
    )

    chunks = split_documents([doc], chunk_size=180, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.page_content.startswith("| Key | Val |\n| --- | --- |")
        assert chunk.metadata["source"] == "big_table.pdf"
        assert chunk.metadata["page"] == 2


def test_ingest_document_end_to_end_preserves_source_and_page(tmp_path: Path):
    pdf_path = _write_synthetic_pdf(
        tmp_path / "uuid123.pdf",
        [b"BT /F1 12 Tf 50 740 Td (End to end ingestion check.) Tj ET"],
    )

    chunks = ingest_document(str(pdf_path), source_name="quarterly.pdf")

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "quarterly.pdf"
    assert chunks[0].metadata["page"] == 1
    assert "End to end ingestion check." in chunks[0].page_content


def test_upload_endpoint_preserves_contract_and_handles_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(upload_api, "find_document_by_hash", lambda h: None)
    monkeypatch.setattr(upload_api, "save_uploaded_file", lambda filename, content: _save_tmp(tmp_path, filename, content))
    mock_store = MagicMock()
    monkeypatch.setattr(upload_api, "create_vector_store", lambda: mock_store)

    test_app = FastAPI()
    test_app.include_router(upload_api.router)
    client = TestClient(test_app)

    # 1. Valid PDF upload preserves API contract and real filename metadata
    valid_pdf = _write_synthetic_pdf(
        tmp_path / "src.pdf",
        [b"BT /F1 12 Tf 50 740 Td (Uploaded PDF content.) Tj ET"],
    )
    resp = client.post(
        "/upload",
        files={"file": ("report.pdf", valid_pdf.read_bytes(), "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "report.pdf"
    assert body["message"] == "File uploaded successfully."
    assert body["chunks_created"] == 1
    assert "saved_path" in body
    added_chunks = mock_store.add_documents.call_args[0][0]
    assert added_chunks[0].metadata["source"] == "report.pdf"
    assert added_chunks[0].metadata["page"] == 1

    # 2. Scanned/empty PDF returns 400 and cleans up saved file
    scanned_pdf = _write_synthetic_pdf(tmp_path / "scanned_src.pdf", [b""])
    resp_scanned = client.post(
        "/upload",
        files={"file": ("scanned.pdf", scanned_pdf.read_bytes(), "application/pdf")},
    )
    assert resp_scanned.status_code == 400
    assert "scanned" in resp_scanned.json()["detail"].lower()

    # 3. Corrupt DOCX returns 400
    resp_corrupt = client.post(
        "/upload",
        files={"file": ("broken.docx", b"not-a-docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert resp_corrupt.status_code == 400

    # 4. Unsupported extension returns 400
    resp_txt = client.post(
        "/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp_txt.status_code == 400

    # 5. Empty file returns 400
    resp_empty = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp_empty.status_code == 400


def _save_tmp(tmp_path: Path, filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    target = tmp_path / f"saved_{Path(filename).stem}{ext}"
    target.write_bytes(content)
    return str(target)


def test_load_pdf_skips_find_tables_on_text_only_pages_and_keeps_table_pages(tmp_path: Path, monkeypatch):
    prose_only_ops = b"BT /F1 12 Tf 50 740 Td (Pure text page without tables.) Tj ET"
    table_ops = (
        b"50 660 200 60 re S\n"
        b"50 690 m 250 690 l S\n"
        b"150 660 m 150 720 l S\n"
        b"BT /F1 10 Tf 60 700 Td (ColA) Tj ET\n"
        b"BT /F1 10 Tf 160 700 Td (ColB) Tj ET\n"
        b"BT /F1 10 Tf 60 670 Td (Val1) Tj ET\n"
        b"BT /F1 10 Tf 160 670 Td (Val2) Tj ET"
    )
    pdf_path = _write_synthetic_pdf(tmp_path / "mixed.pdf", [prose_only_ops, table_ops])

    orig_find_tables = pdfplumber.page.Page.find_tables
    find_tables_pages: list[int] = []

    def spy_find_tables(self, *args, **kwargs):
        find_tables_pages.append(self.page_number)
        return orig_find_tables(self, *args, **kwargs)

    monkeypatch.setattr(pdfplumber.page.Page, "find_tables", spy_find_tables)

    docs = load_pdf(str(pdf_path), source_name="mixed.pdf")

    assert len(docs) == 2
    assert find_tables_pages == [2]  # Skipped expensive table scan on page 1
    assert "Pure text page without tables." in docs[0].page_content
    assert docs[0].metadata["page"] == 1
    assert "| ColA | ColB |\n| --- | --- |\n| Val1 | Val2 |" in docs[1].page_content
    assert docs[1].metadata["page"] == 2
