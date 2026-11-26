from pathlib import Path

from app.config import settings
from app.ingestion.clean_documents import clean_documents
from app.ingestion.document_processor import process_document
from app.retrieval.chunking import split_documents
from app.utils.file_utils import DocumentLimitError


def ingest_document(
    file_path: str,
    source_name: str | None = None,
    max_chunks: int | None = None,
):
    extension = Path(file_path).suffix.lower()
    display_source = source_name or Path(file_path).name
    chunk_limit = settings.MAX_DOCUMENT_CHUNKS if max_chunks is None else max_chunks

    if extension not in {".pdf", ".docx"}:
        raise ValueError("Only PDF and DOCX files are supported.")

    # Step 1: Load document
    documents = process_document(file_path, source_name=source_name)

    # Step 2: Clean document text
    documents = clean_documents(documents)

    # Step 3: Split into chunks
    chunks = split_documents(documents)

    if chunk_limit > 0 and len(chunks) > chunk_limit:
        raise DocumentLimitError(
            f"Document '{display_source}' generated {len(chunks)} chunks, exceeding the maximum allowed limit of {chunk_limit} chunks."
        )

    return chunks
