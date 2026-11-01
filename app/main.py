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


def _mark_binary_fields(node):
    """Add format: binary next to contentMediaType: application/octet-stream.

    FastAPI describes UploadFile fields with contentMediaType only, but
    Swagger UI needs format: binary to show a file picker. Without it,
    Swagger sends uploads as text and /ingest rejects them with 422.
    """

    if isinstance(node, dict):
        if node.get("contentMediaType") == "application/octet-stream":
            node.setdefault("format", "binary")
        for value in node.values():
            _mark_binary_fields(value)

        # "UploadFile | str" produces anyOf [file, string]. Show only the
        # file option so Swagger UI renders a file picker.
        options = node.get("anyOf")
        if isinstance(options, list):
            binary = [o for o in options
                      if isinstance(o, dict) and o.get("format") == "binary"]
            if binary:
                del node["anyOf"]
                node.update(binary[0])
    elif isinstance(node, list):
        for item in node:
            _mark_binary_fields(item)


def custom_openapi():
    if app.openapi_schema is None:
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        _mark_binary_fields(schema)
        app.openapi_schema = schema

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
