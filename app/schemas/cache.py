from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to search/generate response for.")
    context_hash: Optional[str] = Field(None, description="The context hash of the source documents to validate cache consistency.")
    threshold: Optional[float] = Field(None, description="Dynamic similarity threshold to override system default.")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The generated or cached response.")
    status: str = Field(..., description="Cache status: 'HIT' or 'MISS'.")
    similarity: Optional[float] = Field(None, description="Cosine similarity score if cache HIT.")

class InvalidateRequest(BaseModel):
    context_hash: str = Field(..., description="The context hash of the source documents to invalidate.")

class InvalidateResponse(BaseModel):
    invalidated_count: int = Field(..., description="The number of cache entries invalidated.")
