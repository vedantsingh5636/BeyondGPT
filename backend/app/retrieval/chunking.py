import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_TABLE_SEP_RE = re.compile(r"^\|\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*\|?$")


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _segment_blocks(text: str) -> list[tuple[str, str]]:
    """Split text into ordered ('text', ...) and ('table', ...) blocks."""
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    text_lines: list[str] = []
    i = 0

    def flush_text() -> None:
        nonlocal text_lines
        joined = "\n".join(text_lines).strip()
        if joined:
            blocks.append(("text", joined))
        text_lines = []

    while i < len(lines):
        if (
            i + 1 < len(lines)
            and _is_table_line(lines[i])
            and _TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            flush_text()
            table_lines = [lines[i].strip(), lines[i + 1].strip()]
            i += 2
            while i < len(lines) and _is_table_line(lines[i]):
                table_lines.append(lines[i].strip())
                i += 1
            blocks.append(("table", "\n".join(table_lines)))
        else:
            text_lines.append(lines[i])
            i += 1

    flush_text()
    return blocks


def _split_large_table(table_text: str, chunk_size: int) -> list[str]:
    """Split an oversized Markdown table on row boundaries, repeating the header."""
    lines = table_text.splitlines()
    if len(lines) <= 2:
        return [table_text]

    header_block = lines[:2]
    header_len = sum(len(line) + 1 for line in header_block)
    rows = lines[2:]

    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = header_len

    for row in rows:
        addition = len(row) + 1
        if current_rows and current_len + addition > chunk_size:
            chunks.append("\n".join([*header_block, *current_rows]))
            current_rows = [row]
            current_len = header_len + addition
        else:
            current_rows.append(row)
            current_len += addition

    if current_rows:
        chunks.append("\n".join([*header_block, *current_rows]))

    return chunks


def _split_single_document(
    document: Document,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> list[Document]:
    """Split one Document into metadata-preserving chunks without breaking tables."""
    base_metadata = dict(document.metadata or {})
    base_metadata.setdefault("source", "Unknown")
    base_metadata.setdefault("page", "N/A")

    blocks = _segment_blocks(document.page_content)
    if not blocks:
        return []

    pieces: list[str] = []
    buffer = ""

    def flush_buffer() -> None:
        nonlocal buffer
        if buffer:
            pieces.append(buffer)
            buffer = ""

    for kind, content in blocks:
        if kind == "table":
            if len(content) > chunk_size:
                flush_buffer()
                pieces.extend(_split_large_table(content, chunk_size))
            elif buffer and len(buffer) + 2 + len(content) <= chunk_size:
                buffer = f"{buffer}\n\n{content}"
            else:
                flush_buffer()
                buffer = content
        else:
            text_parts = (
                [content]
                if len(content) <= chunk_size
                else splitter.split_text(content)
            )
            for part in text_parts:
                part = part.strip()
                if not part:
                    continue
                if buffer and len(buffer) + 2 + len(part) <= chunk_size:
                    buffer = f"{buffer}\n\n{part}"
                else:
                    flush_buffer()
                    buffer = part

    flush_buffer()

    return [
        Document(page_content=piece, metadata=dict(base_metadata))
        for piece in pieces
    ]


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into chunks while preserving metadata and keeping tables intact."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for document in documents:
        chunks.extend(_split_single_document(document, splitter, chunk_size))
    return chunks
