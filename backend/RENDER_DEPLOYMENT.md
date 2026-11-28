# Render Free Deployment Notes (512 MB RAM)

## Recommended Render Service Settings

- **Root Directory:** `backend`
- **Build Command:** `pip install -r requirements-render.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`

## Required & Recommended Environment Variables

Set these in the Render Dashboard (never commit real secrets):

- `GROQ_API_KEY` — Groq API key
- `QDRANT_URL` — Qdrant Cloud cluster URL
- `QDRANT_API_KEY` — Qdrant Cloud API key
- `QDRANT_COLLECTION_NAME=rag_documents`
- `QDRANT_TIMEOUT_SECONDS=15.0`
- `CORS_ORIGINS=["https://beyond-gpt-orpin.vercel.app"]`
- `MALLOC_ARENA_MAX=2` — limits Linux `glibc` per-thread malloc arenas so freed ONNX/PDF memory is returned via `malloc_trim(0)` instead of accumulating across arenas on 512 MB Render Free containers.
