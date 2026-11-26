from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ENV = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    GROQ_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "rag_documents"
    QDRANT_TIMEOUT_SECONDS: float = 5.0
    GROQ_TIMEOUT_SECONDS: float = 30.0
    UPLOAD_DIR: str | None = None
    MAX_UPLOAD_SIZE_MB: int = 10
    MAX_PDF_PAGES: int = 100
    MAX_DOCUMENT_CHUNKS: int = 500
    MAX_CHAT_MESSAGE_LENGTH: int = 4000
    RATE_LIMIT_PER_MINUTE: int = 60
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=(".env", str(_BACKEND_ENV)),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
