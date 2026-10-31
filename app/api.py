from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.ingestion.indexer import (
    index_documents,
    index_uploaded_documents
)
from app.database.vector_store import collection
from app.database.feedback_store import (
    save_feedback,
    get_all_feedback
)
from app.schemas import FeedbackRequest


router = APIRouter()

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".md", ".txt", ".html"}


@router.post("/ingest")
async def ingest_documents(
    files: list[UploadFile] = File(default=[])
):
    # Preserve existing behavior when no files are uploaded.
    if not files:
        index_documents()

        return {
            "message": "Existing documents ingested successfully",
            "total_chunks": collection.count()
        }

    uploaded_files = []
    seen_filenames = set()

    for file in files:
        original_filename = file.filename or ""
        filename = Path(
            original_filename.replace("\\", "/")
        ).name

        if not filename:
            raise HTTPException(
                status_code=400,
                detail="Every uploaded file must have a filename."
            )

        extension = Path(filename).suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {extension}. "
                    "Allowed formats: .md, .txt, .html"
                )
            )

        if filename.lower() in seen_filenames:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate filename: {filename}"
            )

        seen_filenames.add(filename.lower())

        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail=f"{filename} is empty."
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"{filename} exceeds the 5 MB file limit."
            )

        uploaded_files.append((filename, content))

    try:
        result = index_uploaded_documents(uploaded_files)

        return {
            "message": "Uploaded documents indexed successfully",
            **result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


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
