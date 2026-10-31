from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from app.schemas import QueryRequest, QueryResponse
from app.database.feedback_store import initialize_feedback_db
from app.rag.graph import rag_graph
from app.config import PROJECT_NAME
from app.api import router
from app.llm import GeminiServiceError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources during startup."""
    initialize_feedback_db()
    yield


app = FastAPI(
    title=PROJECT_NAME,
    description="RAG-based Technical Documentation Assistant",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)


def custom_openapi():
    """Customize OpenAPI schema for Swagger file uploads."""

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes
    )

    openapi_schema["openapi"] = "3.0.3"

    ingest_schema = (
        openapi_schema
        .get("components", {})
        .get("schemas", {})
        .get("Body_ingest_documents_ingest_post", {})
    )

    files_schema = (
        ingest_schema
        .get("properties", {})
        .get("files", {})
    )

    if files_schema:
        files_schema.clear()
        files_schema.update({
            "type": "array",
            "items": {
                "type": "string",
                "format": "binary"
            },
            "title": "Files"
        })

    if ingest_schema:
        required_fields = ingest_schema.get("required", [])
        if "files" in required_fields:
            required_fields.remove("files")

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
def home():
    return {
        "message": "Technical Documentation Assistant API is running"
    }


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """Process a user query through the LangGraph RAG workflow."""

    try:
        result = rag_graph.invoke({
            "question": request.question,
            "rewritten_query": "",
            "query_type": "",
            "documents": [],
            "relevant_documents": [],
            "retry_count": 0,
            "answer": "",
            "sources": []
        })

        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=result["sources"]
        )

    except GeminiServiceError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=str(error)
        )

    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="The query could not be processed."
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your query."
        )
