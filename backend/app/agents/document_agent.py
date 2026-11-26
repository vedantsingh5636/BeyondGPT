from app.agents.prompt_utils import build_agent_messages
from app.agents.state import AgentState
from app.llm.groq import get_groq_llm


def document_agent(state: AgentState) -> dict[str, str]:
    """Generate a grounded, cited answer from retrieved document chunks."""
    documents = state.get("retrieved_documents") or []
    if not documents:
        return {
            "answer": (
                "I couldn't find relevant information in your uploaded documents for that question. "
                "Please upload a PDF or DOCX document or try rephrasing your query."
            )
        }

    llm = get_groq_llm()
    messages = build_agent_messages(
        role_instruction="You are BeyondGPT, a precise knowledge assistant answering questions from uploaded documents.",
        query=state["query"],
        documents=documents,
        chat_history=state.get("chat_history") or [],
    )
    response = llm.invoke(messages)

    return {"answer": response.content}
