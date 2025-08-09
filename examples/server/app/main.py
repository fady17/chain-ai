# ===== app/main.py =====
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import time

from .models import QueryRequest, QueryResponse, UploadResponse, HealthResponse
from .rag_service import RAGService
from .config import settings

app = FastAPI(
    title=settings.app_name,
    description="RAG API service for document querying",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get RAG service instance
def get_rag_service() -> RAGService:
    return RAGService()

@app.get("/", response_model=HealthResponse)
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """Health check endpoint"""
    status = rag_service.get_status()
    return HealthResponse(**status)

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Upload and index a document"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_path = Path(file.filename)
    if file_path.suffix.lower() not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Allowed: {settings.allowed_extensions}"
        )
    
    # Save uploaded file
    upload_path = settings.upload_dir / f"{int(time.time())}_{file.filename}"
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add to RAG system
        result = rag_service.add_document(upload_path)
        
        return UploadResponse(
            message="Document uploaded and indexed successfully",
            **result
        )
        
    except Exception as e:
        # Clean up file on error
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Query the indexed documents"""
    try:
        result = rag_service.query(
            question=request.question,
            include_context=request.include_context,
            retrieval_k=request.retrieval_k
        )
        return QueryResponse(**result)
        
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/documents")
async def list_documents(rag_service: RAGService = Depends(get_rag_service)):
    """List all indexed documents"""
    return {
        "documents": list(rag_service._document_registry.values()),
        "total_count": len(rag_service._document_registry)
    }

