import logging
import re

import httpx
from groq import GroqError
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq import get_groq_llm

logger = logging.getLogger(__name__)

_FOLLOWUP_INDICATOR_RE = re.compile(
    r"\b("
    r"it|its|they|them|their|this|that|these|those|he|she|his|her|"
    r"former|latter|above|previous|earlier|same|more|details|elaborate|"
    r"clarify|shorter|simpler|translate|summarize\s+that|what\s+about|"
    r"how\s+about|why\s+did|and\s+what"
    r")\b",
    re.IGNORECASE,
)

_FILENAME_HINT_RE = re.compile(r"\b[\w.-]+\.(pdf|docx|doc|py|js|ts|md)\b", re.IGNORECASE)
_PAGE_CITATION_SUFFIX_RE = re.compile(
    r"([\[【][^\]】\n]+?)\s*[,·]\s*Pages?\s+(?:\d+(?:\s*[-–]\s*\d+)?|N/?A)([\]】])",
    re.IGNORECASE,
)


def needs_query_contextualization(query: str, chat_history: list[dict[str, str]]) -> bool:
    """Return True when a query likely relies on previous conversation turns."""
    if not chat_history:
        return False
    stripped = query.strip()
    if _FOLLOWUP_INDICATOR_RE.search(stripped):
        return True
    # Short follow-up questions without an explicit filename often omit the topic noun.
    return len(stripped.split()) <= 5 and not _FILENAME_HINT_RE.search(stripped)


def format_chat_history(chat_history: list[dict[str, str]], max_messages: int = 6) -> str:
    """Format recent conversation messages into a compact readable block."""
    if not chat_history:
        return ""
    recent = chat_history[-max_messages:]
    lines = []
    for item in recent:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = (item.get("content") or "").strip()
        if content:
            if role == "Assistant":
                content = _PAGE_CITATION_SUFFIX_RE.sub(r"\1\2", content)
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def contextualize_query(query: str, chat_history: list[dict[str, str]]) -> str:
    """Rewrite a follow-up question into a standalone search query using history."""
    if not needs_query_contextualization(query, chat_history):
        return query

    history_text = format_chat_history(chat_history, max_messages=6)
    if not history_text:
        return query

    try:
        llm = get_groq_llm()
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Rewrite the user's latest follow-up message into a single, "
                        "self-contained search query using the conversation history. "
                        "Preserve exact filenames, technical terms, numbers, and the "
                        "user's language. Return only the rewritten query with no quotes "
                        "or explanation."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Conversation history:\n{history_text}\n\n"
                        f"Latest user message:\n{query}"
                    )
                ),
            ]
        )
        rewritten = (getattr(response, "content", "") or "").strip().strip('"').strip("'")
        if rewritten:
            return rewritten
    except (GroqError, httpx.HTTPError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Query contextualization fallback triggered: %s", exc)

    # Deterministic fallback: combine last user message topic with the follow-up.
    last_user_msgs = [
        (item.get("content") or "").strip()
        for item in chat_history
        if item.get("role") == "user" and (item.get("content") or "").strip()
    ]
    if last_user_msgs:
        return f"{last_user_msgs[-1]} {query}".strip()
    return query


def format_retrieved_context(documents: list[Document]) -> tuple[str, list[str]]:
    """Format retrieved chunks using filenames only while preserving chunk metadata internally."""
    if not documents:
        return "", []

    blocks: list[str] = []
    allowed_citations: list[str] = []
    seen_citations: set[str] = set()

    for idx, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source_name = str(metadata.get("filename") or metadata.get("source") or "Unknown")

        if source_name not in seen_citations:
            seen_citations.add(source_name)
            allowed_citations.append(source_name)

        blocks.append(
            f"[{idx}] Source: {source_name}\n"
            f"{document.page_content.strip()}"
        )

    return "\n\n---\n\n".join(blocks), allowed_citations


def build_agent_messages(
    *,
    role_instruction: str,
    query: str,
    documents: list[Document],
    chat_history: list[dict[str, str]] | None = None,
) -> list:
    """Build grounded System/Human messages with filename-only citation rules."""
    context_text, allowed_citations = format_retrieved_context(documents)
    history_text = format_chat_history(chat_history or [], max_messages=6)
    allowed_list = "\n".join(f"- {c}" for c in allowed_citations)

    system_msg = SystemMessage(
        content=(
            f"{role_instruction}\n\n"
            "Rules:\n"
            "1. Answer clearly, directly, and naturally in the same language as the user's question.\n"
            "2. Rely strictly on the retrieved context as your factual source. "
            "Use conversation history only to understand follow-up references.\n"
            "3. Avoid repetitive filler or unnecessary disclaimers.\n"
            "4. If the retrieved context lacks the answer, state briefly what information is missing "
            "without inventing facts.\n"
            "5. Citations: Cite ONLY source filenames from the Allowed Citations list below "
            "(e.g., [report.pdf] or [notes.docx]). Do NOT include page numbers, and never invent, "
            "guess, or modify filenames."
        )
    )

    human_parts = []
    if history_text:
        human_parts.append(f"Recent conversation:\n{history_text}")
    human_parts.append(f"Allowed Citations:\n{allowed_list}")
    human_parts.append(f"Retrieved Context:\n{context_text}")
    human_parts.append(f"Question:\n{query}")

    return [system_msg, HumanMessage(content="\n\n".join(human_parts))]
