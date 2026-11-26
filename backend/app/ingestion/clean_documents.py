
from app.ingestion.text_cleaner import clean_text


def clean_documents(documents):
    for document in documents:
        document.page_content = clean_text(document.page_content)

    return documents
