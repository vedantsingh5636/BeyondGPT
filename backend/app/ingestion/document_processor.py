from pathlib import Path

from app.ingestion.docx_loader import load_docx
from app.ingestion.pdf_loader import load_pdf


def process_document(file_path: str, source_name: str | None = None):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        documents = load_pdf(file_path, source_name=source_name)

    elif extension == ".docx":
        documents = load_docx(file_path, source_name=source_name)

    else:
        raise ValueError("Unsupported document format.")

    return documents
