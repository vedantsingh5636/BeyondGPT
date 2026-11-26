from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.upload import router as upload_router
from app.config import settings
from app.retrieval.vector_store import check_qdrant_health

app = FastAPI(
    title="Agentic RAG Chatbot",
    description="An AI-powered Agentic RAG Chatbot using Groq",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(upload_router)


@app.get("/")
async def root():
    return {
        "message": "Agentic RAG Chatbot API is running!",
        "status": "success",
    }


@app.get("/health")
def health():
    qdrant_status = check_qdrant_health()
    overall = "healthy" if qdrant_status["status"] == "ok" else "degraded"
    return {
        "status": overall,
        "backend": "ok",
        "qdrant": qdrant_status,
        "groq_configured": bool(settings.GROQ_API_KEY.strip()),
    }
