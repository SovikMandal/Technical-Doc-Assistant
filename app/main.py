
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

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
