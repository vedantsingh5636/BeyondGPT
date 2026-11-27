import logging
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from git.exc import GitError
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ApiException

from app.config import settings
from app.ingestion.clean_documents import clean_documents
from app.ingestion.github_loader import load_github_repo
from app.ingestion.ingestion_pipeline import ingest_document
from app.ingestion.web_loader import load_website
from app.retrieval.chunking import split_documents
from app.retrieval.vector_store import (
    add_documents_batched,
    create_vector_store,
    enrich_chunks_metadata,
    find_document_by_hash,
    invalidate_document_cache,
)
from app.utils.file_utils import (
    DocumentLimitError,
    compute_file_hash,
    generate_document_id,
    sanitize_filename,
    save_uploaded_file,
    validate_file,
    validate_file_signature,
    validate_mime_type,
)
from app.utils.rate_limiter import enforce_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter()

_READ_CHUNK_BYTES = 1024 * 1024


class WebsiteRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)


class GitHubRequest(BaseModel):
    repo_url: str = Field(..., min_length=8, max_length=2048)
    branch: str = Field(default="main", min_length=1, max_length=100)


async def _run_blocking(func, *args, **kwargs):
    return await run_in_threadpool(func, *args, **kwargs)


async def _read_upload_with_limit(file: UploadFile, max_bytes: int) -> bytes:
    """Read uploaded file incrementally and reject oversized payloads before buffering all."""
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
            )
        chunks.append(chunk)

    return b"".join(chunks)


def _ingest_and_store_file(
    file_path: str,
    safe_filename: str,
    file_hash: str,
    document_id: str,
) -> int:
    chunks = ingest_document(file_path, source_name=safe_filename)
    if not chunks:
        raise ValueError("No extractable content chunks generated from document.")

    chunks, chunk_ids = enrich_chunks_metadata(
        chunks,
        filename=safe_filename,
        file_hash=file_hash,
        document_id=document_id,
        saved_path=file_path,
    )

    vector_store = create_vector_store()
    add_documents_batched(vector_store, chunks, ids=chunk_ids, batch_size=64)
    invalidate_document_cache()
    return len(chunks)


@router.post("/upload", dependencies=[Depends(enforce_rate_limit)])
async def upload_document(file: Annotated[UploadFile, File()]):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    safe_filename = sanitize_filename(file.filename)
    if not validate_file(safe_filename):
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are allowed.",
        )

    content_type = getattr(file, "content_type", None)
    if not validate_mime_type(safe_filename, content_type):
        raise HTTPException(
            status_code=400,
            detail="File MIME type does not match the allowed PDF or DOCX format.",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    try:
        content = await _read_upload_with_limit(file, max_bytes)
    except TypeError:
        # Support simplified test doubles whose read() takes no size argument.
        content = await file.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB} MB.",
            ) from None

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if not validate_file_signature(safe_filename, content):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file content is corrupt or does not match a valid PDF/DOCX signature.",
        )

    file_hash = compute_file_hash(content)
    document_id = generate_document_id(file_hash)

    try:
        existing_metadata = await _run_blocking(find_document_by_hash, file_hash)
    except (ApiException, httpx.HTTPError, OSError, RuntimeError) as exc:
        logger.warning("Vector database unavailable during duplicate check: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Vector database is currently unavailable. Please try again shortly.",
        ) from exc

    if existing_metadata is not None:
        return {
            "filename": safe_filename,
            "message": "Duplicate file detected. Document already exists in knowledge base.",
            "saved_path": existing_metadata.get("saved_path", ""),
            "chunks_created": 0,
            "duplicate": True,
            "document_id": existing_metadata.get("document_id", document_id),
            "file_hash": file_hash,
        }

    try:
        file_path = await _run_blocking(
            save_uploaded_file,
            filename=safe_filename,
            content=content,
        )
    except OSError as exc:
        logger.exception("Failed to save uploaded file '%s': %s", safe_filename, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error saving uploaded file '%s': %s", safe_filename, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded file.",
        ) from exc

    try:
        chunks_created = await _run_blocking(
            _ingest_and_store_file,
            file_path,
            safe_filename,
            file_hash,
            document_id,
        )

        return {
            "filename": safe_filename,
            "message": "File uploaded successfully.",
            "saved_path": file_path,
            "chunks_created": chunks_created,
            "duplicate": False,
            "document_id": document_id,
            "file_hash": file_hash,
        }
    except DocumentLimitError as exc:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        Path(file_path).unlink(missing_ok=True)
        raise
    except (ApiException, httpx.HTTPError, OSError, RuntimeError) as exc:
        Path(file_path).unlink(missing_ok=True)
        logger.warning("Failed to store uploaded file in vector database: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Vector database is currently unavailable. Please try again shortly.",
        ) from exc
    except Exception as exc:
        Path(file_path).unlink(missing_ok=True)
        logger.exception("Unexpected error processing uploaded file")
        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded file.",
        ) from exc


def _ingest_website_sync(url: str) -> int:
    documents = load_website(url)
    if not documents:
        raise ValueError("No content found on website.")

    documents = clean_documents(documents)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No chunks generated from website.")

    vector_store = create_vector_store()
    vector_store.add_documents(chunks)
    invalidate_document_cache()
    return len(chunks)


@router.post("/ingest-website", dependencies=[Depends(enforce_rate_limit)])
async def ingest_website(request: WebsiteRequest):
    try:
        chunks_created = await _run_blocking(_ingest_website_sync, request.url)
        return {
            "message": "Website ingested successfully.",
            "url": request.url,
            "chunks_created": chunks_created,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ApiException, httpx.HTTPError, OSError, RuntimeError) as exc:
        logger.exception("Website ingestion failed")
        raise HTTPException(
            status_code=502,
            detail="Unable to fetch or store website content.",
        ) from exc


def _ingest_github_sync(repo_url: str, branch: str) -> int:
    documents = load_github_repo(repo_url, branch=branch)
    if not documents:
        raise ValueError("No documents found in repository.")

    documents = clean_documents(documents)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No chunks generated from repository.")

    vector_store = create_vector_store()
    vector_store.add_documents(chunks)
    invalidate_document_cache()
    return len(chunks)


@router.post("/ingest-github", dependencies=[Depends(enforce_rate_limit)])
async def ingest_github(request: GitHubRequest):
    try:
        chunks_created = await _run_blocking(
            _ingest_github_sync,
            request.repo_url,
            request.branch,
        )
        return {
            "message": "GitHub repository ingested successfully.",
            "repo_url": request.repo_url,
            "chunks_created": chunks_created,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ApiException, GitError, httpx.HTTPError, OSError, RuntimeError) as exc:
        logger.exception("GitHub ingestion failed")
        raise HTTPException(
            status_code=502,
            detail="Unable to clone or store GitHub repository content.",
        ) from exc
