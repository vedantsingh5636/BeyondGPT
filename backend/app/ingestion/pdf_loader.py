import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document as LCDocument
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException

from app.config import settings
from app.ingestion.table_utils import format_markdown_table
from app.utils.file_utils import DocumentLimitError

BBox = tuple[float, float, float, float]
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


class PdfSubprocessError(Exception):
    """Raised when the isolated PDF parsing subprocess crashes, times out, or returns invalid output."""


def _obj_outside_bboxes(obj: dict, bboxes: list[BBox]) -> bool:
    """Return True if a pdfplumber character lies outside all table bounding boxes."""
    if obj.get("object_type") != "char":
        return True
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return not any(
        x0 <= cx <= x1 and top <= cy <= bottom
        for x0, top, x1, bottom in bboxes
    )


def _extract_prose_region(page: pdfplumber.page.Page, top: float, bottom: float) -> str:
    """Extract clean prose text from a vertical band of a pdfplumber page."""
    clamped_top = max(0.0, min(top, float(page.height)))
    clamped_bottom = max(0.0, min(bottom, float(page.height)))
    if clamped_bottom - clamped_top <= 1.0:
        return ""
    cropped = page.crop((0.0, clamped_top, float(page.width), clamped_bottom), strict=False)
    text = cropped.extract_text() or ""
    return text.strip()


def _has_table_primitives(page: pdfplumber.page.Page) -> bool:
    """Return True only when the page has geometry capable of forming a table."""
    return bool(page.rects or page.curves or len(page.lines) >= 2)


def _extract_page_content(page: pdfplumber.page.Page) -> tuple[str, int]:
    """Extract prose and structured Markdown tables in reading order from one page."""
    found_tables = (page.find_tables() or []) if _has_table_primitives(page) else []
    table_entries: list[tuple[BBox, str]] = []

    for table in found_tables:
        markdown = format_markdown_table(table.extract())
        if markdown:
            table_entries.append((table.bbox, markdown))

    if not table_entries:
        text = (page.extract_text() or "").strip()
        return text, 0

    table_entries.sort(key=lambda item: (item[0][1], item[0][0]))
    table_bboxes = [bbox for bbox, _ in table_entries]
    prose_page = page.filter(lambda obj: _obj_outside_bboxes(obj, table_bboxes))

    # Merge vertically overlapping tables (e.g. side-by-side) into ordered bands.
    bands: list[list] = []
    for bbox, markdown in table_entries:
        top, bottom = float(bbox[1]), float(bbox[3])
        if bands and top <= bands[-1][1] + 2.0:
            bands[-1][1] = max(bands[-1][1], bottom)
            bands[-1][2].append(markdown)
        else:
            bands.append([top, bottom, [markdown]])

    parts: list[str] = []
    cursor_y = 0.0
    for band_top, band_bottom, tables_md in bands:
        prose = _extract_prose_region(prose_page, cursor_y, band_top)
        if prose:
            parts.append(prose)
        parts.extend(tables_md)
        cursor_y = max(cursor_y, band_bottom)

    trailing = _extract_prose_region(prose_page, cursor_y, float(page.height))
    if trailing:
        parts.append(trailing)

    return "\n\n".join(parts).strip(), len(table_entries)


def load_pdf(
    file_path: str,
    source_name: str | None = None,
    max_pages: int | None = None,
) -> list[LCDocument]:
    """Load a PDF file into page-level Documents with 1-based page numbers and tables.

    Raises FileNotFoundError if the file does not exist, DocumentLimitError when
    a PDF exceeds max_pages, and ValueError for unsupported, corrupt,
    password-protected, empty, or scanned image-only PDFs.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    display_source = source_name or path.name
    page_limit = settings.MAX_PDF_PAGES if max_pages is None else max_pages

    try:
        with pdfplumber.open(str(path)) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                raise ValueError(f"PDF '{display_source}' contains no pages.")
            if page_limit > 0 and total_pages > page_limit:
                raise DocumentLimitError(
                    f"PDF '{display_source}' has {total_pages} pages, exceeding the maximum allowed limit of {page_limit} pages."
                )

            documents: list[LCDocument] = []
            empty_pages: list[int] = []

            for page_number, page in enumerate(pdf.pages, start=1):
                try:
                    content, table_count = _extract_page_content(page)
                finally:
                    if hasattr(page, "flush_cache"):
                        page.flush_cache()

                if page_number % 10 == 0:
                    gc.collect()

                has_meaningful_text = bool(re.sub(r"\s+", "", content))

                if not has_meaningful_text:
                    empty_pages.append(page_number)
                    continue

                documents.append(
                    LCDocument(
                        page_content=content,
                        metadata={
                            "source": display_source,
                            "page": page_number,
                            "total_pages": total_pages,
                            "file_type": "pdf",
                            "table_count": table_count,
                        },
                    )
                )
            if hasattr(pdf, "flush_cache"):
                pdf.flush_cache()
        if total_pages >= 10:
            gc.collect()
    except ValueError:
        raise
    except PDFPasswordIncorrect as exc:
        raise ValueError(
            f"PDF '{display_source}' is password-protected and cannot be read."
        ) from exc
    except PdfminerException as exc:
        if any(isinstance(arg, PDFPasswordIncorrect) for arg in exc.args) or "password" in str(exc).lower():
            raise ValueError(
                f"PDF '{display_source}' is password-protected and cannot be read."
            ) from exc
        raise ValueError(
            f"Failed to read PDF '{display_source}': file appears corrupt or invalid."
        ) from exc
    except (PDFSyntaxError, OSError) as exc:
        raise ValueError(
            f"Failed to read PDF '{display_source}': file appears corrupt or invalid."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Failed to process PDF '{display_source}': {exc}"
        ) from exc

    if not documents:
        raise ValueError(
            f"No extractable text found in PDF '{display_source}'. "
            "It appears to be scanned or image-only."
        )

    if empty_pages:
        for doc in documents:
            doc.metadata["empty_pages"] = empty_pages

    return documents


def load_pdf_in_subprocess(
    file_path: str,
    source_name: str | None = None,
    max_pages: int | None = None,
    timeout_seconds: float = 120.0,
) -> list[LCDocument]:
    """Parse a PDF inside a short-lived spawned Python subprocess so pdfminer memory is reclaimed on exit."""
    effective_max_pages = settings.MAX_PDF_PAGES if max_pages is None else max_pages
    payload = json.dumps(
        {
            "file_path": file_path,
            "source_name": source_name,
            "max_pages": effective_max_pages,
        }
    )
    env = dict(os.environ)
    env.setdefault("MALLOC_ARENA_MAX", "2")

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "app.ingestion.pdf_loader"],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_seconds,
            cwd=str(_BACKEND_ROOT),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PdfSubprocessError(
            f"PDF parsing timed out after {timeout_seconds:.0f}s."
        ) from exc
    except OSError as exc:
        raise PdfSubprocessError(f"Failed to launch PDF parsing subprocess: {exc}") from exc

    if completed.returncode != 0:
        stderr_snippet = (completed.stderr or "").strip()[-400:]
        raise PdfSubprocessError(
            f"PDF parsing subprocess exited with code {completed.returncode}"
            + (f": {stderr_snippet}" if stderr_snippet else ".")
        )

    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise PdfSubprocessError("PDF parsing subprocess returned invalid JSON.") from exc

    if not isinstance(result, dict):
        raise PdfSubprocessError("PDF parsing subprocess returned an unexpected payload.")

    if not result.get("ok"):
        error_type = result.get("error_type")
        detail = str(result.get("detail") or "Failed to process PDF.")
        if error_type == "FileNotFoundError":
            raise FileNotFoundError(detail)
        if error_type == "DocumentLimitError":
            raise DocumentLimitError(detail)
        if error_type == "ValueError":
            raise ValueError(detail)
        raise PdfSubprocessError(detail)

    raw_docs = result.get("documents")
    if not isinstance(raw_docs, list):
        raise PdfSubprocessError("PDF parsing subprocess omitted extracted documents.")

    return [
        LCDocument(
            page_content=item["page_content"],
            metadata=dict(item.get("metadata") or {}),
        )
        for item in raw_docs
    ]


def _run_subprocess_worker() -> int:
    raw_input = sys.stdin.read()
    try:
        request = json.loads(raw_input)
        docs = load_pdf(
            request["file_path"],
            source_name=request.get("source_name"),
            max_pages=request.get("max_pages"),
        )
        response = {
            "ok": True,
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
        }
    except FileNotFoundError as exc:
        response = {"ok": False, "error_type": "FileNotFoundError", "detail": str(exc)}
    except DocumentLimitError as exc:
        response = {"ok": False, "error_type": "DocumentLimitError", "detail": str(exc)}
    except ValueError as exc:
        response = {"ok": False, "error_type": "ValueError", "detail": str(exc)}
    except Exception as exc:
        response = {"ok": False, "error_type": type(exc).__name__, "detail": str(exc)}

    sys.stdout.write(json.dumps(response, ensure_ascii=False))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_subprocess_worker())
