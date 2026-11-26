import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from app.agents import code_agent as code_agent_module
from app.agents import document_agent as document_agent_module
from app.agents import prompt_utils
from app.agents import retrieval_agent as retrieval_agent_module
from app.agents.supervisor import supervisor_agent
from app.api import chat as chat_api
from app.graph import workflow as workflow_module
from app.llm import groq as groq_module


def test_code_routing():
    result = supervisor_agent({"query": "Write a Python function"})
    assert result["next_agent"] == "code_agent"

    react_bug = supervisor_agent({"query": "Debug the React component error in App.jsx"})
    assert react_bug["next_agent"] == "code_agent"


def test_document_routing():
    result = supervisor_agent({"query": "Explain database normalization"})
    assert result["next_agent"] == "document_agent"

    # Normal PDF/DOCX questions containing "code", "class", "error", or "function"
    assert supervisor_agent({"query": "What is the code of conduct in hr.pdf?"})["next_agent"] == "document_agent"
    assert supervisor_agent({"query": "What is the error margin in report.pdf?"})["next_agent"] == "document_agent"
    assert supervisor_agent({"query": "Describe the class schedule in syllabus.docx"})["next_agent"] == "document_agent"
    assert supervisor_agent({"query": "What is the function of the liver in biology.pdf?"})["next_agent"] == "document_agent"
    assert supervisor_agent({"query": "What is the status code on page 2?"})["next_agent"] == "document_agent"


def test_graph_compilation():
    graph = workflow_module.build_graph()
    assert graph is not None
    assert workflow_module.build_graph() is graph


def test_groq_client_is_cached_with_existing_model(monkeypatch):
    calls: list[dict] = []

    def fake_chat_groq(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**kwargs)

    groq_module.get_groq_llm.cache_clear()
    try:
        monkeypatch.setattr(groq_module, "ChatGroq", fake_chat_groq)
        llm1 = groq_module.get_groq_llm()
        llm2 = groq_module.get_groq_llm()

        assert llm1 is llm2
        assert len(calls) == 1
        assert calls[0]["model"] == "openai/gpt-oss-120b"
        assert calls[0]["temperature"] == 0.2
    finally:
        groq_module.get_groq_llm.cache_clear()


def test_citations_and_answer_generation(monkeypatch):
    docs = [
        Document(
            page_content="BeyondGPT uses FastEmbed and CrossEncoder reranking.",
            metadata={"source": "architecture.pdf", "filename": "architecture.pdf", "page": 3},
        ),
        Document(
            page_content="All uploaded DOCX tables are converted to Markdown.",
            metadata={"source": "spec.docx", "filename": "spec.docx", "page": "N/A"},
        ),
    ]

    context_text, allowed = prompt_utils.format_retrieved_context(docs)
    assert "Source: architecture.pdf" in context_text
    assert "Source: spec.docx" in context_text
    assert "Page:" not in context_text
    assert allowed == ["architecture.pdf", "spec.docx"]
    # Internal page metadata remains intact on chunk objects
    assert docs[0].metadata["page"] == 3

    captured_messages: list = []

    def fake_invoke(messages):
        captured_messages.append(messages)
        return SimpleNamespace(
            content="BeyondGPT uses FastEmbed and CrossEncoder reranking [architecture.pdf]."
        )

    monkeypatch.setattr(
        document_agent_module,
        "get_groq_llm",
        lambda: SimpleNamespace(invoke=fake_invoke),
    )

    res = document_agent_module.document_agent(
        {
            "query": "What reranker does BeyondGPT use?",
            "retrieved_documents": docs,
            "chat_history": [
                {"role": "user", "content": "Tell me about architecture.pdf"},
                {"role": "assistant", "content": "Details [architecture.pdf, Page 3]."},
            ],
        }
    )

    assert "[architecture.pdf]" in res["answer"]
    assert "Page 3" not in res["answer"]
    system_prompt = captured_messages[0][0].content
    human_prompt = captured_messages[0][1].content
    assert "Cite ONLY source filenames" in system_prompt
    assert "Do NOT include page numbers" in system_prompt
    assert "architecture.pdf" in human_prompt
    assert "spec.docx" in human_prompt
    assert "Page 3" not in human_prompt

    # Empty retrieval returns clean fallback without calling Groq or inventing citations
    empty_res = document_agent_module.document_agent({"query": "Anything?", "retrieved_documents": []})
    assert "upload a PDF or DOCX" in empty_res["answer"]

    empty_code_res = code_agent_module.code_agent({"query": "Fix bug", "retrieved_documents": []})
    assert "knowledge base" in empty_code_res["answer"]


def test_followup_contextualization_and_chat_endpoint_compatibility(monkeypatch):
    history = [
        {"role": "user", "content": "What does finance_2026.pdf say about Q3 revenue?"},
        {"role": "assistant", "content": "Q3 revenue was 42.5 million USD [finance_2026.pdf, Page 1]."},
    ]

    assert prompt_utils.needs_query_contextualization("What about Q4?", history) is True
    assert prompt_utils.needs_query_contextualization("What about its margin?", history) is True
    assert prompt_utils.needs_query_contextualization("What does hr_policy.docx say about leave?", history) is False
    assert prompt_utils.needs_query_contextualization("What about it?", []) is False

    monkeypatch.setattr(
        prompt_utils,
        "get_groq_llm",
        lambda: SimpleNamespace(
            invoke=lambda msgs: SimpleNamespace(content="What does finance_2026.pdf say about Q4 revenue?")
        ),
    )
    rewritten = prompt_utils.contextualize_query("What about Q4?", history)
    assert rewritten == "What does finance_2026.pdf say about Q4 revenue?"

    # Verify retrieval_agent passes rewritten standalone query to retrieve_documents
    retrieved_queries: list[str] = []
    monkeypatch.setattr(
        retrieval_agent_module,
        "retrieve_documents",
        lambda query, top_k=5: retrieved_queries.append(query) or [
            Document(page_content="Q4 revenue rose 10%.", metadata={"source": "finance_2026.pdf", "page": 2})
        ],
    )
    ret_state = retrieval_agent_module.retrieval_agent(
        {"query": "What about Q4?", "chat_history": history}
    )
    assert ret_state["standalone_query"] == "What does finance_2026.pdf say about Q4 revenue?"
    assert retrieved_queries == ["What does finance_2026.pdf say about Q4 revenue?"]

    # Verify POST /chat preserves {"message": "..."} compatibility and forwards optional history
    invoked_inputs: list[dict] = []
    monkeypatch.setattr(
        chat_api,
        "build_graph",
        lambda: SimpleNamespace(
            invoke=lambda state: invoked_inputs.append(state) or {"answer": "Mocked answer."}
        ),
    )

    legacy_req = chat_api.ChatRequest(message="Summarize finance_2026.pdf")
    legacy_resp = chat_api.chat(legacy_req)
    assert legacy_resp == {"response": "Mocked answer."}
    assert invoked_inputs[-1]["chat_history"] == []

    followup_req = chat_api.ChatRequest(
        message="What about Q4?",
        history=[chat_api.ChatHistoryItem(**item) for item in history],
    )
    followup_resp = chat_api.chat(followup_req)
    assert followup_resp == {"response": "Mocked answer."}
    assert len(invoked_inputs[-1]["chat_history"]) == 2

    with pytest.raises(HTTPException) as exc_info:
        chat_api.chat(chat_api.ChatRequest(message="   "))
    assert exc_info.value.status_code == 400


def test_chat_stream_preserves_citations_history_and_errors(monkeypatch):
    docs = [
        Document(
            page_content="BeyondGPT streams grounded answers via Groq.",
            metadata={"source": "guide.pdf", "filename": "guide.pdf", "page": 2},
        )
    ]
    captured_history: list = []

    def fake_retrieval(state):
        captured_history.append(state["chat_history"])
        return {"standalone_query": state["query"], "retrieved_documents": docs}

    monkeypatch.setattr(chat_api, "retrieval_agent", fake_retrieval)
    monkeypatch.setattr(
        chat_api,
        "get_groq_llm",
        lambda: SimpleNamespace(
            stream=lambda msgs: iter(
                [
                    SimpleNamespace(content="BeyondGPT streams via Groq "),
                    SimpleNamespace(content="[guide.pdf]."),
                ]
            )
        ),
    )

    events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in chat_api.stream_chat_events(
            "How does it stream?",
            [{"role": "user", "content": "Tell me about guide.pdf"}],
        )
    ]
    assert captured_history == [[{"role": "user", "content": "Tell me about guide.pdf"}]]
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert events[-1]["response"] == "BeyondGPT streams via Groq [guide.pdf]."

    # Empty retrieved documents emits fallback without calling Groq
    monkeypatch.setattr(
        chat_api,
        "retrieval_agent",
        lambda state: {"standalone_query": state["query"], "retrieved_documents": []},
    )
    empty_events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in chat_api.stream_chat_events("Missing?", [])
    ]
    assert empty_events[0]["type"] == "token"
    assert empty_events[-1]["type"] == "done"
    assert "upload a PDF or DOCX" in empty_events[-1]["response"]

    # Streaming error emits an SSE error event instead of crashing silently
    monkeypatch.setattr(chat_api, "retrieval_agent", fake_retrieval)

    def broken_stream(msgs):
        raise httpx.ConnectError("Groq unreachable")

    monkeypatch.setattr(chat_api, "get_groq_llm", lambda: SimpleNamespace(stream=broken_stream))
    err_events = [
        json.loads(line.removeprefix("data: ").strip())
        for line in chat_api.stream_chat_events("Fail test", [])
    ]
    assert len(err_events) == 1
    assert err_events[0]["type"] == "error"
    assert "temporarily unavailable" in err_events[0]["detail"].lower()

    with pytest.raises(HTTPException) as exc_info:
        chat_api.chat_stream(chat_api.ChatRequest(message="  "))
    assert exc_info.value.status_code == 400
