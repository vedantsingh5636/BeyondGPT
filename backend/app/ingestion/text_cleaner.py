
import re


def clean_text(text: str) -> str:
    # Remove extra spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Replace multiple newlines with a single newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove unnecessary spaces at the beginning and end
    text = text.strip()

    return text
