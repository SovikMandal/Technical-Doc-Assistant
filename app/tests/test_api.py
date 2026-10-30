from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()


def test_query_endpoint():
    fake_result = {
        "question": "How do FastAPI dependencies work?",
        "answer": "FastAPI uses Depends to declare dependencies.",
        "sources": ["dependencies.md"]
    }

    with patch(
        "app.main.rag_graph.invoke",
        return_value=fake_result
    ):
        response = client.post(
            "/query",
            json={
                "question": "How do FastAPI dependencies work?"
            }
        )

    assert response.status_code == 200
    assert response.json()["sources"] == ["dependencies.md"]
    assert "Depends" in response.json()["answer"]


def test_query_validation():
    response = client.post(
        "/query",
        json={"question": "Hi"}
    )

    assert response.status_code == 422


def test_general_query_response():
    fake_result = {
        "question": "Hello",
        "answer": "Hello! How can I help?",
        "sources": []
    }

    with patch(
        "app.main.rag_graph.invoke",
        return_value=fake_result
    ):
        response = client.post(
            "/query",
            json={"question": "Hello"}
        )

    assert response.status_code == 200
    assert response.json()["sources"] == []


def test_feedback_validation():
    response = client.post(
        "/feedback",
        json={
            "question": "Test question",
            "answer": "Test answer",
            "rating": 6
        }
    )

    assert response.status_code == 422
