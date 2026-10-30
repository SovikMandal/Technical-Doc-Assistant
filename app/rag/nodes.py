import json
import re
import time
from app.llm import generate_text
from app.database.retriever import retrieve_documents
from app.config import TOP_K


def extract_json(text: str) -> dict:
    """Safely extract and validate a JSON object from an LLM response."""

    if not text or not text.strip():
        raise ValueError("LLM returned an empty response.")

    text = text.strip()

    # Remove optional Markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        result = json.loads(text)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"LLM returned invalid JSON: {error}"
        ) from error

    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object.")

    return result


def analyze_query(state: dict) -> dict:
    """Classify incoming queries using simple deterministic rules."""
    question = state["question"].strip()
    normalized = question.lower().strip("?!., ")

    general_messages = {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "who are you",
        "what can you do"
    }

    query_type = (
        "general"
        if normalized in general_messages
        else "technical"
    )

    return {
        "query_type": query_type,
        "rewritten_query": question
    }


def retrieve_documents_node(state: dict) -> dict:
    """Retrieve relevant document chunks from ChromaDB."""

    query = state.get("rewritten_query") or state["question"]

    print("Retrieving documents...")

    documents = retrieve_documents(
        query=query,
        top_k=TOP_K
    )

    print(f"Retrieved {len(documents)} document chunks.")

    return {
        "documents": documents
    }


def grade_documents(state: dict) -> dict:
    """Grade retrieved documents using local cosine similarity."""

    documents = state.get("documents", [])

    if not documents:
        print("No documents retrieved.")
        return {"relevant_documents": []}

    MIN_RELEVANCE_SCORE = 0.50

    relevant_documents = []

    print("Grading documents locally...")

    for document in documents:
        distance = document.get("distance")

        if distance is None:
            continue

        # ChromaDB cosine distance = 1 - cosine similarity
        similarity = 1 - distance

        source = document["metadata"]["source"]

        print(
            f"Source: {source} | "
            f"Similarity: {similarity:.4f}"
        )

        if similarity >= MIN_RELEVANCE_SCORE:
            document["relevance_score"] = similarity
            relevant_documents.append(document)

    print(
        f"Local grading completed. "
        f"Relevant documents: {len(relevant_documents)}/{len(documents)}"
    )

    return {
        "relevant_documents": relevant_documents
    }


def rewrite_query(state: dict) -> dict:
    """Rewrite the question if retrieval finds no relevant documents."""

    question = state["question"]
    previous_query = state.get("rewritten_query") or question

    prompt = f"""
You are a search query optimization assistant.

The previous search did not find relevant documentation.

Original question:
{question}

Previous search query:
{previous_query}

Create a better search query that may retrieve useful FastAPI documentation.

Return ONLY the improved search query.
Do not answer the question.
"""

    print("Rewriting search query with Gemini...")

    start_time = time.perf_counter()

    improved_query = generate_text(prompt).strip()

    print(
        f"Query rewriting time: "
        f"{time.perf_counter() - start_time:.2f} seconds"
    )

    if not improved_query:
        raise ValueError("Gemini returned an empty rewritten query.")

    return {
        "rewritten_query": improved_query,
        "retry_count": state.get("retry_count", 0) + 1
    }