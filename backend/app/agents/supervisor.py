import re

from app.agents.state import AgentState

_DOCUMENT_INTENT_RE = re.compile(
    r"\.(pdf|docx|doc|txt|md)\b|"
    r"\b(pdf|docx|document|documents|report|paper|manual|resume|invoice|"
    r"contract|policy|syllabus|handbook|chapter|page|pages|paragraph|table|slides?)\b",
    re.IGNORECASE,
)

# Clear document-domain phrases that contain words like "code", "class", "error", or "function".
_NON_PROGRAMMING_PHRASE_RE = re.compile(
    r"\b("
    r"code\s+of\s+conduct|zip\s+code|postal\s+code|area\s+code|country\s+code|"
    r"status\s+code|error\s+code|error\s+margin|error\s+rate|margin\s+of\s+error|"
    r"class\s+schedule|social\s+class|asset\s+class|weight\s+class|"
    r"function\s+of\s+(the|a|an)\s+[a-z]+"
    r")\b",
    re.IGNORECASE,
)

# Strong programming/codebase signals requiring code_agent.
_STRONG_CODE_SIGNAL_RE = re.compile(
    r"\b("
    r"python|javascript|typescript|react|jsx|tsx|fastapi|django|flask|"
    r"sql|postgres|postgresql|github|repo|repository|codebase|"
    r"traceback|stack\s*trace|syntax\s*error|compile|compiler|refactor|"
    r"unit\s*test|pytest|programming|debug|debugging|snippet|algorithm"
    r")\b|\.(py|js|ts|jsx|tsx|java|cpp|go|rs)\b",
    re.IGNORECASE,
)

_CODE_TASK_ACTION_RE = re.compile(
    r"\b(write|implement|fix|debug|refactor|generate|build|create)\b"
    r".{0,40}\b(code|function|method|class|script|component|endpoint|api|bug)\b",
    re.IGNORECASE,
)


def supervisor_agent(state: AgentState) -> dict[str, str]:
    """Route queries based on document/source intent vs explicit programming intent."""
    query = (state.get("standalone_query") or state.get("query") or "").strip()

    has_document_intent = bool(_DOCUMENT_INTENT_RE.search(query))
    has_non_programming_phrase = bool(_NON_PROGRAMMING_PHRASE_RE.search(query))
    has_strong_code_signal = bool(_STRONG_CODE_SIGNAL_RE.search(query))
    has_code_action = bool(_CODE_TASK_ACTION_RE.search(query))

    if has_document_intent and not (has_strong_code_signal and has_code_action):
        return {"next_agent": "document_agent"}

    if has_non_programming_phrase and not has_strong_code_signal:
        return {"next_agent": "document_agent"}

    if has_strong_code_signal or has_code_action:
        return {"next_agent": "code_agent"}

    return {"next_agent": "document_agent"}
