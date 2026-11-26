import re
from pathlib import Path
from zipfile import BadZipFile

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document as LCDocument

from app.ingestion.table_utils import format_markdown_table


def _format_paragraph(paragraph: Paragraph) -> str:
    """Format a DOCX paragraph while preserving heading and list structure."""
    text = paragraph.text.strip()
    if not text:
        return ""

    style_name = (paragraph.style.name if paragraph.style else "") or ""
    normalized_style = style_name.lower()

    if normalized_style == "title":
        return f"# {text}"
    if normalized_style == "subtitle":
        return f"## {text}"

    heading_match = re.match(r"^heading\s*(\d+)$", normalized_style)
    if heading_match:
        level = max(1, min(6, int(heading_match.group(1))))
        return f"{'#' * level} {text}"

    if "list bullet" in normalized_style:
        return f"- {text}"

    return text


def _extract_docx_table(table: Table) -> str:
    """Extract a python-docx Table into a Markdown table."""
    rows = [[cell.text for cell in row.cells] for row in table.rows]
    formatted = format_markdown_table(rows)
    if formatted:
        return formatted

    # Fallback for 1-column or single-row layout tables with text
    lines = []
    for row in rows:
        cells = [re.sub(r"\s+", " ", c).strip() for c in row if c and c.strip()]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def load_docx(file_path: str, source_name: str | None = None) -> list[LCDocument]:
    """Load a DOCX file preserving headings, paragraphs, tables, and source metadata."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"DOCX not found: {file_path}")

    if path.suffix.lower() != ".docx":
        raise ValueError("Only DOCX files are supported.")

    display_source = source_name or path.name

    try:
        word_doc = Document(str(path))
    except (PackageNotFoundError, BadZipFile, KeyError, ValueError, OSError) as exc:
        raise ValueError(
            f"Failed to read DOCX '{display_source}': file appears corrupt or invalid."
        ) from exc

    blocks: list[str] = []
    table_count = 0

    for child in word_doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            formatted = _format_paragraph(Paragraph(child, word_doc))
            if formatted:
                blocks.append(formatted)
        elif isinstance(child, CT_Tbl):
            table_text = _extract_docx_table(Table(child, word_doc))
            if table_text:
                blocks.append(table_text)
                table_count += 1

    full_text = "\n\n".join(blocks).strip()
    if not re.sub(r"\s+", "", full_text):
        raise ValueError(
            f"No extractable text or tables found in DOCX '{display_source}'."
        )

    document = LCDocument(
        page_content=full_text,
        metadata={
            "source": display_source,
            "page": "N/A",
            "file_type": "docx",
            "table_count": table_count,
        },
    )

    return [document]
