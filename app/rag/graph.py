from langgraph.graph import StateGraph, START, END
from app.rag.state import RAGState
from app.rag.nodes import (
    analyze_query,
    retrieve_documents_node,
    grade_documents,
    rewrite_query
)
from app.rag.generator import (
    generate_answer,
    fallback_answer,
    handle_general_query
)
from app.config import MAX_RETRIES


def decide_after_analysis(state: RAGState) -> str:
    if state.get("query_type") == "general":
        return "general"

    return "retrieve"


def decide_after_grading(state: RAGState) -> str:
    if state.get("relevant_documents"):
        return "generate"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"

    return "no_answer"


def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("analyze", analyze_query)
    workflow.add_node("general", handle_general_query)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("fallback", fallback_answer)

    workflow.add_edge(START, "analyze")

    workflow.add_conditional_edges(
        "analyze",
        decide_after_analysis,
        {
            "general": "general",
            "retrieve": "retrieve"
        }
    )

    workflow.add_edge("general", END)
    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        decide_after_grading,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "no_answer": "fallback"
        }
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


rag_graph = build_graph()
