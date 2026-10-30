from fastapi import APIRouter, HTTPException
from app.ingestion.indexer import index_documents
from app.database.vector_store import collection
from app.database.feedback_store import (
    save_feedback,
    get_all_feedback
)
from app.schemas import FeedbackRequest

router = APIRouter()

@router.post("/ingest")
def ingest_documents():
    index_documents()

    return {
        "message": "Documents ingested successfully",
        "total_chunks": collection.count()
    }


@router.get("/documents")
def get_documents():
    results = collection.get(include=["metadatas"])

    documents = {}

    for metadata in results["metadatas"]:
        source = metadata["source"]
        documents[source] = documents.get(source, 0) + 1

    return {
        "total_documents": len(documents),
        "documents": [
            {
                "filename": filename,
                "chunks": chunks
            }
            for filename, chunks in documents.items()
        ]
    }


@router.post("/feedback")
def submit_feedback(request: FeedbackRequest):
    try:
        feedback_id = save_feedback(
            question=request.question,
            answer=request.answer,
            rating=request.rating,
            feedback=request.feedback
        )

        return {
            "message": "Feedback saved successfully",
            "feedback_id": feedback_id,
            "rating": request.rating
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to save feedback"
        )


@router.get("/feedback")
def list_feedback():
    return {
        "feedback": get_all_feedback()
    }