from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=1000,
        description="User's technical question"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int = Field(ge=1, le=5)
    feedback: str | None = None