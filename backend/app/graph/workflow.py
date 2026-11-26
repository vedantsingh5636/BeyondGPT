from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.code_agent import code_agent
from app.agents.document_agent import document_agent
from app.agents.retrieval_agent import retrieval_agent
from app.agents.state import AgentState
from app.agents.supervisor import supervisor_agent


@lru_cache(maxsize=1)
def build_graph():
    """Compile and cache the LangGraph multi-agent workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("retrieval", retrieval_agent)
    workflow.add_node("document", document_agent)
    workflow.add_node("code", code_agent)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],
        {
            "document_agent": "retrieval",
            "code_agent": "retrieval",
        },
    )

    workflow.add_conditional_edges(
        "retrieval",
        lambda state: state["next_agent"],
        {
            "document_agent": "document",
            "code_agent": "code",
        },
    )

    workflow.add_edge("document", END)
    workflow.add_edge("code", END)

    return workflow.compile()
