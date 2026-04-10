from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str = Field(..., example="What are the key benefits of RAG?")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class IngestResponse(BaseModel):
    message: str
    status: str
    chunks_processed: int
