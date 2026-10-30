from unittest.mock import patch
from app.rag.graph import build_graph

def fake_analyze_query(state):
    return {
        "query_type": "technical",
        "rewritten_query": state["question"]
    }


def fake_retrieve_documents(state):
    return {
        "documents": [
            {
                "content": "FastAPI uses Python type hints to declare path parameters.",
                "metadata": {"source": "path-params.md"},
                "distance": 0.2
            }
        ]
    }


def fake_grade_documents(state):
    return {
        "relevant_documents": state["documents"]
    }


def fake_generate_answer(state):
    return {
        "answer": "FastAPI uses Python type hints to declare path parameters.",
        "sources": ["path-params.md"]
    }


def test_successful_workflow():

    with (
        patch("app.rag.graph.analyze_query", fake_analyze_query),
        patch("app.rag.graph.retrieve_documents_node", fake_retrieve_documents),
        patch("app.rag.graph.grade_documents", fake_grade_documents),
        patch("app.rag.graph.generate_answer", fake_generate_answer)
    ):

        # Build AFTER applying mocks
        test_graph = build_graph()

        result = test_graph.invoke({
            "question": "How do I define path parameters in FastAPI?",
            "retry_count": 0,
            "documents": [],
            "relevant_documents": [],
            "rewritten_query": "",
            "query_type": "",
            "answer": "",
            "sources": []
        })

    assert result["answer"] != ""
    assert "path-params.md" in result["sources"]

    print("Workflow test passed!")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])

def test_retry_workflow():

    retrieval_count = 0

    def fake_retrieve_with_retry(state):
        nonlocal retrieval_count
        retrieval_count += 1

        return {
            "documents": [
                {
                    "content": "FastAPI path parameters are declared using Python type hints.",
                    "metadata": {"source": "path-params.md"},
                    "distance": 0.2
                }
            ]
        }

    def fake_grade_with_retry(state):
        if retrieval_count < 3:
            return {"relevant_documents": []}

        return {"relevant_documents": state["documents"]}

    with (
        patch("app.rag.graph.analyze_query", fake_analyze_query),
        patch("app.rag.graph.retrieve_documents_node", fake_retrieve_with_retry),
        patch("app.rag.graph.grade_documents", fake_grade_with_retry),
        patch("app.rag.graph.rewrite_query", lambda state: {
            "rewritten_query": "FastAPI path parameter syntax",
            "retry_count": state.get("retry_count", 0) + 1
        }),
        patch("app.rag.graph.generate_answer", fake_generate_answer)
    ):

        test_graph = build_graph()

        result = test_graph.invoke({
            "question": "How do I define path parameters?",
            "retry_count": 0,
            "documents": [],
            "relevant_documents": [],
            "rewritten_query": "",
            "query_type": "",
            "answer": "",
            "sources": []
        })

    assert retrieval_count == 3
    assert result["answer"] != ""
    assert result["sources"] == ["path-params.md"]

    print("\nRetry workflow test passed!")
    print("Retrieval attempts:", retrieval_count)
    print("Answer:", result["answer"])


def test_retry_exhaustion():

    retrieval_count = 0

    def fake_retrieve(state):
        nonlocal retrieval_count
        retrieval_count += 1

        return {
            "documents": [
                {
                    "content": "Unrelated documentation about request bodies.",
                    "metadata": {"source": "body.md"},
                    "distance": 0.8
                }
            ]
        }

    def fake_grade(state):
        return {"relevant_documents": []}

    with (
        patch("app.rag.graph.analyze_query", fake_analyze_query),
        patch("app.rag.graph.retrieve_documents_node", fake_retrieve),
        patch("app.rag.graph.grade_documents", fake_grade),
        patch("app.rag.graph.rewrite_query", lambda state: {
            "rewritten_query": "improved search query",
            "retry_count": state.get("retry_count", 0) + 1
        })
    ):

        test_graph = build_graph()

        result = test_graph.invoke({
            "question": "How does FastAPI handle WebSockets?",
            "retry_count": 0,
            "documents": [],
            "relevant_documents": [],
            "rewritten_query": "",
            "query_type": "",
            "answer": "",
            "sources": []
        })

    assert retrieval_count == 3
    assert result["answer"] != ""
    assert result["sources"] == []

    print("\nRetry exhaustion test passed!")
    print("Retrieval attempts:", retrieval_count)
    print("Fallback answer:", result["answer"])

def test_general_query():

    def fake_general_query(state):
        return {
            "answer": "Hello! I'm your FastAPI Documentation Assistant.",
            "sources": []
        }

    with (
        patch("app.rag.graph.analyze_query", lambda state: {
            "query_type": "general",
            "rewritten_query": state["question"]
        }),
        patch("app.rag.graph.handle_general_query", fake_general_query),
        patch(
            "app.rag.graph.retrieve_documents_node",
            side_effect=AssertionError(
                "General queries should not trigger retrieval"
            )
        )
    ):

        test_graph = build_graph()

        result = test_graph.invoke({
            "question": "Hello",
            "retry_count": 0,
            "documents": [],
            "relevant_documents": [],
            "rewritten_query": "",
            "query_type": "",
            "answer": "",
            "sources": []
        })

    assert "Hello" in result["answer"]
    assert result["sources"] == []

    print("\nGeneral query test passed!")

if __name__ == "__main__":
    test_successful_workflow()
    test_retry_workflow()
    test_retry_exhaustion()
    test_general_query()