import re
from collections.abc import Sequence


def _clean_cell(value: object) -> str:
    """Normalize whitespace and escape pipe characters inside a table cell."""
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.replace("|", "\\|")


def format_markdown_table(rows: Sequence[Sequence[object]] | None) -> str:
    """Convert raw extracted table rows into a structured Markdown table.

    Returns an empty string when the input does not represent a meaningful
    multi-cell table (at least 2 non-empty rows and at least 2 columns).
    """
    if not rows:
        return ""

    cleaned_rows: list[list[str]] = []
    for row in rows:
        cleaned = [_clean_cell(cell) for cell in row]
        if any(cleaned):
            cleaned_rows.append(cleaned)

    if len(cleaned_rows) < 2:
        return ""

    max_cols = max(len(row) for row in cleaned_rows)
    if max_cols < 2:
        return ""

    padded_rows = [row + [""] * (max_cols - len(row)) for row in cleaned_rows]
    header = padded_rows[0]
    separator = ["---"] * max_cols
    body = padded_rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines)
