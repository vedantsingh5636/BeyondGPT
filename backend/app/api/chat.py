import json
import logging
import re
from collections.abc import Iterator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from groq import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    GroqError,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ApiException

from app.agents.prompt_utils import build_agent_messages
from app.agents.retrieval_agent import retrieval_agent
from app.agents.supervisor import supervisor_agent
from app.config import settings
from app.graph.workflow import build_graph
from app.llm.groq import get_groq_llm
from app.utils.rate_limiter import enforce_rate_limit

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_HISTORY_MESSAGES = 10
_SECRET_RE = re.compile(r"(gsk_[A-Za-z0-9_-]+)")


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryItem] = Field(default_factory=list)


def _sanitize_error_message(exc: Exception) -> tuple[int, str]:
    """Map internal backend exceptions to safe HTTP status codes and user-facing messages."""
    raw = str(exc)
    if settings.GROQ_API_KEY and settings.GROQ_API_KEY in raw:
        raw = raw.replace(settings.GROQ_API_KEY, "[REDACTED]")
    raw = _SECRET_RE.sub("[REDACTED]", raw)

    if isinstance(exc, (httpx.TimeoutException, TimeoutError, APITimeoutError)):
        logger.warning("Upstream AI/vector request timed out: %s", raw)
        return 504, "The AI service timed out. Please try again."

    if isinstance(exc, RateLimitError):
        logger.warning("Groq rate limit exceeded: %s", raw)
        return 429, "The AI service is currently rate-limited. Please wait a moment and try again."

    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        logger.warning("Groq authentication error: %s", raw)
        return 502, "AI service authentication failed. Please check the configured API key."

    if isinstance(exc, (APIConnectionError, ApiException, httpx.ConnectError, ConnectionError)):
        logger.warning("Vector or AI service unavailable: %s", raw)
        return 503, "Vector or AI service is temporarily unavailable. Please try again shortly."

    if isinstance(exc, GroqError):
        logger.warning("Groq API error: %s", raw)
        return 502, "The AI service returned an error. Please try again."

    logger.warning("Chat request failed: %s", raw)
    return 502, "Unable to complete the AI request right now. Please try again."


def _prepare_chat_inputs(request: ChatRequest) -> tuple[str, list[dict[str, str]]]:
    message = request.message.strip()
    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    if len(message) > settings.MAX_CHAT_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message exceeds maximum length of {settings.MAX_CHAT_MESSAGE_LENGTH} characters.",
        )

    recent_history = [
        {"role": item.role, "content": item.content[:settings.MAX_CHAT_MESSAGE_LENGTH]}
        for item in request.history[-_MAX_HISTORY_MESSAGES:]
        if item.role in {"user", "assistant"} and item.content and item.content.strip()
    ]
    return message, recent_history


def _format_sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def stream_chat_events(message: str, recent_history: list[dict[str, str]]) -> Iterator[str]:
    """Stream grounded Groq response tokens as SSE while preserving citations and history."""
    try:
        state = {
            "query": message,
            "standalone_query": None,
            "chat_history": recent_history,
            "documents": [],
            "retrieved_documents": [],
            "answer": None,
            "source": None,
            "next_agent": None,
        }
        routing = supervisor_agent(state)
        state.update(routing)

        retrieval_update = retrieval_agent(state)
        state.update(retrieval_update)

        documents = state.get("retrieved_documents") or []
        next_agent = state.get("next_agent") or "document_agent"

        if not documents:
            fallback = (
                "I couldn't find relevant code or technical documentation in your knowledge base "
                "for that question. Please upload or ingest the relevant files first."
                if next_agent == "code_agent"
                else "I couldn't find relevant information in your uploaded documents for that question. "
                "Please upload a PDF or DOCX document or try rephrasing your query."
            )
            yield _format_sse({"type": "token", "content": fallback})
            yield _format_sse({"type": "done", "response": fallback})
            return

        role_instruction = (
            "You are BeyondGPT, a programming assistant explaining code and technical solutions clearly."
            if next_agent == "code_agent"
            else "You are BeyondGPT, a precise knowledge assistant answering questions from uploaded documents."
        )

        messages = build_agent_messages(
            role_instruction=role_instruction,
            query=message,
            documents=documents,
            chat_history=recent_history,
        )

        llm = get_groq_llm()
        collected: list[str] = []
        for chunk in llm.stream(messages):
            delta = getattr(chunk, "content", "")
            if isinstance(delta, str) and delta:
                collected.append(delta)
                yield _format_sse({"type": "token", "content": delta})

        full_response = "".join(collected).strip()
        yield _format_sse({"type": "done", "response": full_response})
    except (
        ApiException,
        GroqError,
        httpx.HTTPError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        _, safe_message = _sanitize_error_message(exc)
        yield _format_sse({"type": "error", "detail": safe_message})


@router.post("/chat", dependencies=[Depends(enforce_rate_limit)])
def chat(request: ChatRequest):
    message, recent_history = _prepare_chat_inputs(request)
    graph = build_graph()

    try:
        result = graph.invoke(
            {
                "query": message,
                "standalone_query": None,
                "chat_history": recent_history,
                "documents": [],
                "retrieved_documents": [],
                "answer": None,
                "source": None,
                "next_agent": None,
            }
        )
    except (
        ApiException,
        GroqError,
        httpx.HTTPError,
        OSError,
        RuntimeError,
        TimeoutError,
        TypeError,
        ValueError,
    ) as exc:
        status_code, safe_message = _sanitize_error_message(exc)
        raise HTTPException(status_code=status_code, detail=safe_message) from exc

    return {
        "response": result["answer"],
    }


@router.post("/chat/stream", dependencies=[Depends(enforce_rate_limit)])
def chat_stream(request: ChatRequest):
    message, recent_history = _prepare_chat_inputs(request)
    return StreamingResponse(
        stream_chat_events(message, recent_history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
