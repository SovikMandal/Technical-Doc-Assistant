from typing import TypedDict

class RAGState(TypedDict):
    # Original user question
    question: str

    # Improved question for retrieval
    rewritten_query: str

    # Query classification
    query_type: str

    # Retrieved documents
    documents: list

    # Documents that passed relevance grading
    relevant_documents: list

    # Number of retrieval retries
    retry_count: int

    # Final generated answer
    answer: str

    # Sources used for the answer
    sources: list
