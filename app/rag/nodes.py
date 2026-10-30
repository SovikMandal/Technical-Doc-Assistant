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
    """Grade retrieved document chunks using Gemini."""

    documents = state.get("documents", [])

    if not documents:
        print("No documents retrieved.")
        return {"relevant_documents": []}

    print("Grading documents with Gemini...")

    formatted_documents = []

    for index, document in enumerate(documents):
        formatted_documents.append({
            "index": index,
            "source": document.get("metadata", {}).get("source", "unknown"),
            "content": document.get("content", "")
        })

    prompt = f"""
You are a strict document relevance evaluator for a
FastAPI technical documentation assistant.

Evaluate each document against the user's question.

USER QUESTION:
{state["question"]}

RETRIEVED DOCUMENTS:
{json.dumps(formatted_documents, indent=2)}

Instructions:
1. Mark a document relevant only if its content contains
   information that helps answer the question.
2. Do not mark a document relevant merely because it
   mentions FastAPI.
3. Do not use outside knowledge.
4. Evaluate every document independently.

Return ONLY valid JSON in this exact format:
{{
    "grades": [
        {{
            "index": 0,
            "relevant": true,
            "reason": "Brief explanation"
        }}
    ]
}}

Include one grade for every document.
"""

    start_time = time.perf_counter()

    response = generate_text(prompt)
    result = extract_json(response)

    grades = result.get("grades")

    if not isinstance(grades, list):
        raise ValueError("Gemini returned an invalid grading response.")

    relevant_documents = []
    processed_indices = set()

    for grade in grades:
        if not isinstance(grade, dict):
            continue

        index = grade.get("index")
        relevant = grade.get("relevant")

        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(documents)
            and index not in processed_indices
            and isinstance(relevant, bool)
        ):
            processed_indices.add(index)

            if relevant:
                document = documents[index].copy()
                document["grading_reason"] = grade.get("reason", "")
                relevant_documents.append(document)

    print(
        f"LLM grading completed in "
        f"{time.perf_counter() - start_time:.2f} seconds."
    )

    print(
        f"Relevant documents: "
        f"{len(relevant_documents)}/{len(documents)}"
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