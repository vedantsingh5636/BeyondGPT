from app.agents.prompt_utils import build_agent_messages
from app.agents.state import AgentState
from app.llm.groq import get_groq_llm


def code_agent(state: AgentState) -> dict[str, str]:
    """Generate a grounded, cited programming answer from retrieved context."""
    documents = state.get("retrieved_documents") or []
    if not documents:
        return {
            "answer": (
                "I couldn't find relevant code or technical documentation in your knowledge base "
                "for that question. Please upload or ingest the relevant files first."
            )
        }

    llm = get_groq_llm()
    messages = build_agent_messages(
        role_instruction="You are BeyondGPT, a programming assistant explaining code and technical solutions clearly.",
        query=state["query"],
        documents=documents,
        chat_history=state.get("chat_history") or [],
    )
    response = llm.invoke(messages)

    return {"answer": response.content}
