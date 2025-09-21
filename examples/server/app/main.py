import re
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pathlib import Path
import shutil
import json
from typing import AsyncGenerator
import uuid
from fastapi.middleware.cors import CORSMiddleware
from chain.rag_runner import create_smart_rag
from chain.vectors.faiss import FAISSVectorStore
from chain.embeddings.local import LocalEmbeddings
from chain.chat_models import LocalChatModel, LocalChatConfig
from chain.core.types import HumanMessage, SystemMessage
from .models import QueryRequest
from .api import ingestion , chat
# --- Configuration ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingestion.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
async def read_root():
    """A simple root endpoint to confirm the server is running."""
    return {"message": "Welcome to the Legal AI RAG API"}

# STORAGE_PATH = Path("./persistent_storage")
# FAISS_INDEX_PATH = str(STORAGE_PATH / "faiss_index")
# UPLOAD_DIR = STORAGE_PATH / "uploads"
# STORAGE_PATH.mkdir(exist_ok=True)
# UPLOAD_DIR.mkdir(exist_ok=True)

# # --- Endpoints ---
# async def process_and_stream_chunks(chat_model, messages) -> AsyncGenerator[str, None]:
#     """
#     Streams chunks from the model, parsing for <think> tags to separate
#     reasoning from text output.
#     """
#     is_thinking = False
#     buffer = ""
#     for chunk in chat_model.stream(messages):
#         buffer += chunk
#         # Process content between <think> tags
#         if is_thinking:
#             end_match = re.search(r"</think>", buffer)
#             if end_match:
#                 # We found the end of the thought block
#                 reasoning_content = buffer[:end_match.start()]
#                 reasoning_obj = {"type": "reasoning", "delta": reasoning_content}
#                 yield f"{json.dumps(reasoning_obj)}\n"
                
#                 # Reset buffer and state
#                 buffer = buffer[end_match.end():]
#                 is_thinking = False
#             else:
#                 # The thought block hasn't ended yet, send the whole buffer as reasoning
#                 reasoning_obj = {"type": "reasoning", "delta": buffer}
#                 yield f"{json.dumps(reasoning_obj)}\n"
#                 buffer = "" # Clear buffer after sending
        
#         # Process content outside of <think> tags
#         if not is_thinking:
#             start_match = re.search(r"<think>", buffer)
#             if start_match:
#                 # We found the start of a new thought block
#                 text_content = buffer[:start_match.start()]
#                 if text_content:
#                     text_obj = {"type": "text", "delta": text_content}
#                     yield f"{json.dumps(text_obj)}\n"
                
#                 # Reset buffer and state
#                 buffer = buffer[start_match.end():]
#                 is_thinking = True
#             else:
#                 # No thought block detected, treat everything as text
#                 if buffer:
#                     text_obj = {"type": "text", "delta": buffer}
#                     yield f"{json.dumps(text_obj)}\n"
#                     buffer = "" # Clear buffer after sending

#     # Send any remaining content in the buffer as text
#     if buffer and not is_thinking:
#         text_obj = {"type": "text", "delta": buffer}
#         yield f"{json.dumps(text_obj)}\n"


# @app.post("/upload")
# async def upload_document(file: UploadFile = File(...)):
#     temp_path = UPLOAD_DIR / (file.filename or "uploaded_file")
#     with open(temp_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     embeddings = LocalEmbeddings()
#     vector_store = None
#     if Path(FAISS_INDEX_PATH).exists():
#         vector_store = FAISSVectorStore.load_local(FAISS_INDEX_PATH, embeddings)
    
#     # Use the library's factory to create a runner, passing the existing store if it exists
#     rag_runner = create_smart_rag(
#         knowledge_files=[str(temp_path)], 
#         vector_store=vector_store
#     )
    
#     # Use the new save() method we added to the library
#     rag_runner.save(FAISS_INDEX_PATH)
    
#     temp_path.unlink()
#     return {"message": "File indexed successfully", "filename": file.filename}

# async def stream_rag_query_generator(question: str) -> AsyncGenerator[str, None]:
#     if not Path(FAISS_INDEX_PATH).exists():
#         error_obj = {"type": "error", "message": "No documents have been indexed yet."}
#         yield f"{json.dumps(error_obj)}\n"
#         return
        
#     embeddings = LocalEmbeddings()
#     vector_store = FAISSVectorStore.load_local(FAISS_INDEX_PATH, embeddings)
    
#     results = vector_store.similarity_search(question, k=3)
#     # Check if any results were found
#     if not results:
#         text_obj = {"type": "text", "delta": "I could not find any relevant information in the uploaded documents to answer your question."}
#         yield f"{json.dumps(text_obj)}\n"
#         return

#     context = "\n\n".join([doc.page_content for doc, score in results])

#     for doc, score in results:
#         source_data = {
#             "type": "source-url",
#             "sourceId": str(uuid.uuid4()),
#             "url": doc.metadata.get('source', 'Unknown Source'),
#             "title": doc.page_content.strip()
#         }
#         source_obj = {"type": "source", "data": source_data}
#         yield f"{json.dumps(source_obj)}\n"

#     chat_model = LocalChatModel(config=LocalChatConfig())
#     enhanced_message = f"Context:\n{context}\n\nQuestion: {question}"
#     messages = [SystemMessage(content="You are a helpful legal AI assistant. Answer the user's question based ONLY on the provided context. Do not add any information that is not in the context. Use <think> tags to outline your reasoning before providing the final answer."), HumanMessage(content=enhanced_message)]
    
#     # Use the new helper function
#     async for chunk in process_and_stream_chunks(chat_model, messages):
#         yield chunk

# @app.post("/stream_rag_query")
# async def stream_rag_query_endpoint(request: QueryRequest):
#     return StreamingResponse(stream_rag_query_generator(request.question), media_type="text/plain")

# async def stream_chat_generator(question: str) -> AsyncGenerator[str, None]:
#     chat_model = LocalChatModel(config=LocalChatConfig())
#     messages = [SystemMessage(content="You are a helpful legal AI assistant. Use <think> tags to outline your reasoning before providing the final answer."), HumanMessage(content=question)]
    
#     # Use the new helper function
#     async for chunk in process_and_stream_chunks(chat_model, messages):
#         yield chunk

# @app.post("/stream_chat")
# async def stream_chat_endpoint(request: QueryRequest):
#     return StreamingResponse(stream_chat_generator(request.question), media_type="text/plain")
# # # ===== app/main.py =====
# # import uuid
# # from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile, HTTPException, Depends
# # from fastapi.middleware.cors import CORSMiddleware
# # from pathlib import Path
# # import shutil
# # import time

# # from fastapi.responses import StreamingResponse

# # from .models import QueryRequest, QueryResponse, HealthResponse, UploadResponse, V2UploadResponse
# # from .rag_service import RAGService
# # from .config import settings
# # from .chat_service import ChatService # Import the new service


# # app = FastAPI(
# #     title=settings.app_name,
# #     description="RAG API service for document querying",
# #     version="1.0.0"
# # )

# # # CORS middleware for frontend integration
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # Configure appropriately for production
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Dependency to get RAG service instance
# # def get_rag_service() -> RAGService:
# #     return RAGService()


# # # Dependency for the new ChatService
# # def get_chat_service() -> ChatService:
# #     return ChatService()

# # @app.get("/", response_model=HealthResponse)
# # async def health_check(rag_service: RAGService = Depends(get_rag_service)):
# #     """Health check endpoint"""
# #     status = rag_service.get_status()
# #     return HealthResponse(**status)

# # @app.post("/upload", response_model=UploadResponse)
# # async def upload_document(
# #     file: UploadFile = File(...),
# #     rag_service: RAGService = Depends(get_rag_service)
# # ):
# #     """Upload and index a document"""
    
# #     # Validate file
# #     if not file.filename:
# #         raise HTTPException(status_code=400, detail="No filename provided")
    
# #     file_path = Path(file.filename)
# #     if file_path.suffix.lower() not in settings.allowed_extensions:
# #         raise HTTPException(
# #             status_code=400, 
# #             detail=f"File type not supported. Allowed: {settings.allowed_extensions}"
# #         )
    
# #     # Save uploaded file
# #     upload_path = settings.upload_dir / f"{int(time.time())}_{file.filename}"
    
# #     try:
# #         with open(upload_path, "wb") as buffer:
# #             shutil.copyfileobj(file.file, buffer)
        
# #         # Add to RAG system
# #         result = rag_service.add_document(upload_path)
        
# #         return UploadResponse(
# #             message="Document uploaded and indexed successfully",
# #             **result
# #         )
        
# #     except Exception as e:
# #         # Clean up file on error
# #         if upload_path.exists():
# #             upload_path.unlink()
# #         raise HTTPException(status_code=500, detail=str(e))

# # @app.post("/query", response_model=QueryResponse)
# # async def query_documents(
# #     request: QueryRequest,
# #     rag_service: RAGService = Depends(get_rag_service)
# # ):
# #     """Query the indexed documents"""
# #     try:
# #         result = rag_service.query(
# #             question=request.question,
# #             include_context=request.include_context,
# #             retrieval_k=request.retrieval_k
# #         )
# #         return QueryResponse(**result)
        
# #     except RuntimeError as e:
# #         raise HTTPException(status_code=400, detail=str(e))
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# # @app.post("/stream_query")
# # async def stream_query_documents(
# #     request: QueryRequest,
# #     rag_service: RAGService = Depends(get_rag_service)
# # ):
# #     """Query the indexed documents and stream the response."""
# #     try:
# #         # We call the new streaming method from our service
# #         generator = rag_service.stream_query(
# #             question=request.question,
# #             include_context=request.include_context
# #         )
# #         return StreamingResponse(generator, media_type="text/event-stream")
        
# #     except RuntimeError as e:
# #         raise HTTPException(status_code=400, detail=str(e))
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
# # @app.post("/stream_query_with_sources")
# # async def stream_query_with_sources_endpoint(
# #     request: QueryRequest,
# #     rag_service: RAGService = Depends(get_rag_service)
# # ):
# #     try:
# #         generator = rag_service.stream_query_with_sources(
# #             question=request.question,
# #         )
# #         return StreamingResponse(generator, media_type="text/plain")
# #     # THE FIX: Catch the specific RuntimeError and return a clean 400 error.
# #     except RuntimeError as e:
# #         # This sends a proper HTTP error instead of crashing the server.
# #         raise HTTPException(status_code=400, detail=str(e))
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
# # @app.get("/documents")
# # async def list_documents(rag_service: RAGService = Depends(get_rag_service)):
# #     """List all indexed documents"""
# #     return {
# #         "documents": list(rag_service._document_registry.values()),
# #         "total_count": len(rag_service._document_registry)
# #     }

# # @app.post("/stream_chat")
# # async def stream_chat_endpoint(
# #     request: QueryRequest,
# #     chat_service: ChatService = Depends(get_chat_service)
# # ):
# #     """Streams a direct chat response without RAG."""
# #     generator = chat_service.stream_chat(question=request.question)
# #     return StreamingResponse(generator, media_type="text/plain")


# # # This is our new, production-grade file upload endpoint.
# # @app.post("/v2/upload", response_model=V2UploadResponse)
# # async def upload_document_v2(
# #     background_tasks: BackgroundTasks, # For scheduling the embedding job
# #     file: UploadFile = File(...),
# #     # We require user_id and case_id to be sent along with the file
# #     # For now, we'll use placeholder defaults for easy testing.
# #     user_id: str = Form("user_placeholder"), 
# #     case_id: str = Form("case_placeholder"),
# #     rag_service: RAGService = Depends(get_rag_service)
# # ):
# #     """
# #     Accepts, validates, and securely stages a document for processing.
# #     This endpoint is asynchronous and returns immediately.
# #     """
# #     # 1. --- Validation ---
# #     if not file.filename:
# #         raise HTTPException(status_code=400, detail="No filename provided.")
    
# #     # Validate file extension
# #     file_extension = Path(file.filename).suffix.lower()
# #     if file_extension not in settings.allowed_extensions:
# #         raise HTTPException(
# #             status_code=400, 
# #             detail=f"File type '{file_extension}' not supported."
# #         )

# #     # 2. --- Secure Staging ---
# #     # Generate a unique, non-guessable ID for the file. This is our primary key.
# #     file_id = uuid.uuid4()
    
# #     # Create a secure, isolated path based on tenancy.
# #     # e.g., ./uploads/user_placeholder/case_placeholder/
# #     staged_dir = settings.upload_dir / user_id / case_id
# #     staged_dir.mkdir(parents=True, exist_ok=True)
    
#     # The final path uses the unique ID and preserves the original extension.
#     staged_path = staged_dir / f"{file_id}{file_extension}"
    
#     try:
#         # Stream the file to the staging location to handle large files efficiently.
#         with open(staged_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
            
#     except Exception as e:
#         # If anything goes wrong, clean up and raise an error.
#         if staged_path.exists():
#             staged_path.unlink()
#         raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

#     # 3. --- Handoff to Background Task ---
#     # Schedule the slow, heavy work (embedding) to be done in the background.
#     # The RAG service will now need a method to handle this.
#     # background_tasks.add_task(rag_service.process_and_embed, file_id=file_id, staged_path=staged_path)

#     # 4. --- Immediate Response ---
#     # Return a 202 Accepted response to the UI immediately.
#     return V2UploadResponse(
#         message="File accepted and is being processed.",
#         file_id=file_id,
#         original_filename=file.filename,
#         status="pending_embedding"
#     )