from typing import Annotated
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.ingestion.indexer import index_chunks, index_documents
from app.ingestion.sources import IngestionError, load_uploaded_file, load_url
from app.database.vector_store import collection
from app.database.feedback_store import (
    save_feedback,
    get_all_feedback
)
from app.schemas import FeedbackRequest
from app.config import MAX_INGEST_ITEMS


router = APIRouter()


def _split_urls(values: list[str] | None) -> list[str]:
    """Accept one URL per form field, or several separated by commas/newlines."""

    urls = []
    for value in values or []:
        for part in value.replace(",", "\n").splitlines():
            if part.strip():
                urls.append(part.strip())

    # Remove duplicates while keeping order
    return list(dict.fromkeys(urls))


@router.post("/ingest")
def ingest_documents(
    # "| str" lets Swagger UI's empty file field ("files=") through
    files: Annotated[
        list[UploadFile | str],
        File(description="Markdown or text files (.md, .markdown, .txt)")
    ] = [],
    urls: Annotated[
        list[str],
        Form(description="Public http(s) URLs of web pages or text files")
    ] = [],
):
    """Ingest uploaded files and/or URLs into the vector database.

    If no files or URLs are sent, the predefined local corpus
    (documents/fastapi/) is ingested instead. Re-ingesting a source with
    the same filename or URL replaces its previous chunks.
    """

    # A non-empty text value means the client sent text instead of a file
    if any(isinstance(file, str) and file.strip() for file in files):
        raise HTTPException(
            status_code=422,
            detail="'files' must be sent as file uploads, not text."
        )

    # Skip empty file fields (Swagger UI sends them when no file is chosen)
    uploads = [
        file for file in files
        if not isinstance(file, str) and file.filename
    ]
    url_list = _split_urls(urls)

    if not uploads and not url_list:
        counts = index_documents()
        source_types = {source: "corpus" for source in counts}

    else:
        if len(uploads) + len(url_list) > MAX_INGEST_ITEMS:
            raise HTTPException(
                status_code=422,
                detail=f"At most {MAX_INGEST_ITEMS} files and URLs "
                       "can be ingested per request."
            )

        # Load everything first so one bad item leaves the index unchanged
        try:
            documents = [
                load_uploaded_file(upload.filename, upload.file)
                for upload in uploads
            ]
            documents += [load_url(url) for url in url_list]

        except IngestionError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=str(error)
            )

        sources = [document.metadata["source"] for document in documents]
        duplicates = {s for s in sources if sources.count(s) > 1}
        if duplicates:
            raise HTTPException(
                status_code=422,
                detail="Duplicate sources in one request: "
                       + ", ".join(sorted(duplicates))
            )

        counts = index_chunks(documents)
        source_types = {
            document.metadata["source"]: document.metadata["source_type"]
            for document in documents
        }

    return {
        "message": "Documents ingested successfully",
        "ingested": [
            {
                "source": source,
                "source_type": source_types[source],
                "chunks": chunks
            }
            for source, chunks in counts.items()
        ],
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
