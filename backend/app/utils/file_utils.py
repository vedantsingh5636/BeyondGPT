import hashlib
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from app.config import settings

_DEFAULT_UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads"
UPLOAD_DIR = _DEFAULT_UPLOAD_DIR


class DocumentLimitError(ValueError):
    """Raised when an uploaded document exceeds page or chunk processing limits."""

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    ".pdf": {
        "application/pdf",
        "application/x-pdf",
        "application/octet-stream",
    },
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
}


def get_upload_dir() -> Path:
    """Return the configured persistent upload directory."""
    if settings.UPLOAD_DIR:
        return Path(settings.UPLOAD_DIR).expanduser().resolve()
    return UPLOAD_DIR


def sanitize_filename(filename: str) -> str:
    """Strip directory traversal and unsafe control characters from a filename."""
    base = Path(filename.replace("\\", "/")).name
    cleaned = re.sub(r"[\x00-\x1f\x7f<>:\"/\\|?*]+", "_", base).strip().strip(".")
    return cleaned or "document"


def validate_file(filename: str) -> bool:
    extension = Path(filename).suffix.lower()
    return extension in ALLOWED_EXTENSIONS


def validate_mime_type(filename: str, content_type: str | None) -> bool:
    """Validate uploaded Content-Type against the file extension."""
    if not content_type:
        return True
    normalized = content_type.split(";")[0].strip().lower()
    if not normalized:
        return True
    extension = Path(filename).suffix.lower()
    allowed = ALLOWED_MIME_TYPES.get(extension, set())
    return normalized in allowed


def validate_file_signature(filename: str, content: bytes) -> bool:
    """Check magic-byte signatures for PDF and DOCX files."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return content.lstrip()[:5] == b"%PDF-"
    if extension == ".docx":
        return content.startswith((b"PK\x03\x04", b"PK\x05\x06"))
    return False


def compute_file_hash(content: bytes) -> str:
    """Return the SHA-256 hex digest of raw uploaded file bytes."""
    return hashlib.sha256(content).hexdigest()


def generate_document_id(file_hash: str) -> str:
    """Return a deterministic UUID string for a document's SHA-256 content hash."""
    return str(uuid5(NAMESPACE_URL, f"beyondgpt:doc:{file_hash}"))


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Return a deterministic UUID point ID for a chunk within a document."""
    return str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk_index}"))


def save_uploaded_file(filename: str, content: bytes) -> str:
    target_dir = get_upload_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_filename(filename)
    extension = Path(safe_name).suffix.lower()
    safe_filename = f"{uuid4().hex}{extension}"

    file_path = target_dir / safe_filename
    file_path.write_bytes(content)

    return str(file_path)
