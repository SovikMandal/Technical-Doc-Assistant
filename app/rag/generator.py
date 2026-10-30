import time
from app.llm import generate_text

def generate_answer(state: dict) -> dict:
    """Generate an answer grounded in relevant documentation."""

    question = state["question"]
    documents = state.get("relevant_documents", [])

    if not documents:
        return fallback_answer(state)

    context_parts = []
    source_names = []

    for document in documents:
        source = document["metadata"]["source"]

        if source not in source_names:
            source_names.append(source)

        context_parts.append(
            f"""
DOCUMENT: {source}

CONTENT:
{document['content']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a technical documentation assistant.

Answer the user's question using ONLY the documentation provided below.

USER QUESTION:
{question}

DOCUMENTATION:
{context}

STRICT INSTRUCTIONS:
1. Use only information supported by the documentation.
2. Do not use outside knowledge or invent details.
3. If the documentation does not contain enough information,
   clearly state that.
4. Cite sources using their exact filenames.
5. Never use numbered citations such as [Source 1] or [Source 2].
6. Put citations immediately after the claims they support.
7. Do not cite a document unless it supports the claim.
8. Keep the answer clear and concise.

Example citation:
According to the documentation, query parameters can have
default values such as None [query-params.md].

Answer:
"""

    start_time = time.perf_counter()

    print("Generating answer with Gemini...")

    answer = generate_text(prompt)

    generation_time = time.perf_counter() - start_time

    print("Answer generation completed.")
    print(f"Generation time: {generation_time:.2f} seconds")

    return {
        "answer": answer,
        "sources": source_names
    }


def fallback_answer(state: dict) -> dict:
    """Return a safe response when no relevant documents are found."""

    return {
        "answer": (
            "I could not find sufficient information in the available "
            "FastAPI documentation to answer your question."
        ),
        "sources": []
    }


def handle_general_query(state: dict) -> dict:
    """Handle greetings and simple conversational messages."""

    question = state["question"].strip().lower()

    if question in {"hi", "hello", "hey"}:
        answer = "Hello! I'm your FastAPI Documentation Assistant. How can I help you?"

    elif question.startswith("thank"):
        answer = "You're welcome! Let me know if you have another question."

    elif question == "who are you":
        answer = "I'm a technical documentation assistant that answers questions using the available FastAPI documentation."

    else:
        answer = "I'm a technical documentation assistant. Ask me something about the available FastAPI documentation."

    return {
        "answer": answer,
        "sources": []
    }