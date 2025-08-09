# ===== app/models.py =====
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    PYTHON = "python"

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the documents")
    include_context: bool = Field(True, description="Whether to include retrieved context")
    retrieval_k: int = Field(3, description="Number of relevant chunks to retrieve", ge=1, le=10)

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: Optional[List[str]] = None
    processing_time: float

class UploadResponse(BaseModel):
    message: str
    document_id: str
    document_type: DocumentType
    chunks_created: int

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    service_ready: bool
