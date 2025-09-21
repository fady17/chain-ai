# ===== app/rag_service.py =====
import json
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from chain.core.types import HumanMessage, SystemMessage
from chain.rag_runner import create_smart_rag, RAGRunner
from pathlib import Path
from typing import Dict, Optional
import uuid
import time
from .config import settings
from .models import DocumentType
from typing import AsyncGenerator


class RAGService:
    """Singleton service that manages RAG runners and document indexing"""
    
    _instance = None
    _rag_runner: Optional[RAGRunner] = None
    _document_registry: Dict[str, dict] = {}
    _runner_save_path = Path("rag_runner.pkl") # Define a save path

    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            settings.upload_dir.mkdir(exist_ok=True)
    
    def _save_runner(self):
        """Saves the RAG runner instance to disk."""
        if self._rag_runner:
            with open(self._runner_save_path, "wb") as f:
                pickle.dump(self._rag_runner, f)

    def _load_runner(self):
        """Loads the RAG runner instance from disk if it exists."""
        if self._runner_save_path.exists():
            with open(self._runner_save_path, "rb") as f:
                self._rag_runner = pickle.load(f)
        else:
            self._rag_runner = None

    def add_document(self, file_path: Path) -> dict:
        """Add a document and save the updated runner."""
        # Load the latest runner before adding to it
        self._load_runner()
        document_id = str(uuid.uuid4())
        doc_type = self._detect_document_type(file_path)
        
        try:
            # Create or update RAG runner with new document
            if self._rag_runner is None:
                self._rag_runner = create_smart_rag(
                    knowledge_files=[str(file_path)],
                    retrieval_k=settings.retrieval_k,
                    debug=settings.debug
                )
            else:
                # For simplicity, recreate with all documents
                # In production, you'd want incremental updates
                all_files = [doc['file_path'] for doc in self._document_registry.values()]
                all_files.append(str(file_path))
                
                self._rag_runner = create_smart_rag(
                    knowledge_files=all_files,
                    retrieval_k=settings.retrieval_k,
                    debug=settings.debug
                )
            
            # Register the document
            doc_info = {
                'document_id': document_id,
                'file_path': str(file_path),
                'document_type': doc_type,
                'upload_time': time.time(),
                'file_size': file_path.stat().st_size
            }
            
            self._document_registry[document_id] = doc_info
            self._save_runner()
        
            return {
                'document_id': document_id,
                'document_type': doc_type,
                'chunks_created': self._estimate_chunks(file_path)
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to process document: {str(e)}")
    
    async def stream_chat(self, question: str) -> AsyncGenerator[str, None]:
        """Streams a direct response from the LLM without RAG."""
        # We must load the runner to see if a model is available.
        self._load_runner()
        if self._rag_runner is None or self._rag_runner.chat_model is None:
            # If no runner has ever been created, we can't chat.
            # This could be improved later by creating a default model.
            raise RuntimeError("Chat model is not available. Please upload a document to initialize the system.")

        messages = [HumanMessage(content=question)]
        if self._rag_runner.config.system_prompt:
            messages.insert(0, SystemMessage(content=self._rag_runner.config.system_prompt)) # type: ignore
        
        try:
            for chunk in self._rag_runner.chat_model.stream(messages): # type: ignore
                yield f'0: "{json.dumps(chunk)}"\n'
        except Exception as e:
            error_message = f"Error during chat generation: {str(e)}"
            yield f'0: "{json.dumps(error_message)}"\n'
    def query(self, question: str, include_context: bool = True, retrieval_k: int = None) -> dict: # type: ignore
        """Query the RAG system"""
        if self._rag_runner is None:
            raise RuntimeError("No documents loaded. Upload documents first.")
        
        start_time = time.time()
        
        try:
            # Update retrieval_k if provided
            if retrieval_k:
                self._rag_runner.config.retrieval_k = retrieval_k
            
            answer = self._rag_runner.query(question, include_context=include_context)
            processing_time = time.time() - start_time
            
            # Get retrieved chunks for transparency (optional)
            retrieved_chunks = None
            if include_context:
                context = self._rag_runner._retrieve_context(question)
                retrieved_chunks = context.split('\n\n') if context else []
            
            return {
                'answer': answer,
                'retrieved_chunks': retrieved_chunks,
                'processing_time': processing_time
            }
            
        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")
    
    def get_status(self) -> dict:
        """Get service status"""
        return {
            'status': 'ready' if self._rag_runner else 'no_documents',
            'documents_loaded': len(self._document_registry),
            'service_ready': self._rag_runner is not None
        }
    
    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type from file extension"""
        ext = file_path.suffix.lower()
        if ext == '.pdf':
            return DocumentType.PDF
        elif ext == '.md':
            return DocumentType.MARKDOWN
        elif ext == '.py':
            return DocumentType.PYTHON
        else:
            return DocumentType.TEXT
    
    def _estimate_chunks(self, file_path: Path) -> int:
        """Rough estimate of chunks created (for API response)"""
        try:
            file_size = file_path.stat().st_size
            # Rough estimate: 1 chunk per ~1KB for text files
            return max(1, file_size // 1000)
        except:
            return 1
        
    async def stream_query(self, question: str, include_context: bool = True) -> AsyncGenerator[str, None]:
        """Streams the RAG response token by token."""
        if self._rag_runner is None or self._rag_runner.chat_model is None:

            raise RuntimeError("No documents loaded. Upload documents first.")
        
        context = ""
        if include_context:
            context = self._rag_runner._retrieve_context(question)
        
        enhanced_message = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        
        messages = []
        if self._rag_runner.config.system_prompt:
            messages.append(SystemMessage(content=self._rag_runner.config.system_prompt))
        
        messages.append(HumanMessage(content=enhanced_message))
        
        try:
            # Use the model's stream method and yield each chunk
            for chunk in self._rag_runner.chat_model.stream(messages):
                yield chunk
                
        except Exception as e:
            if self._rag_runner.config.debug:
                print(f"[DEBUG] Error in stream_query: {e}")
            # Yield a formatted error message if something goes wrong during streaming
            yield f"Error during generation: {str(e)}"


    async def stream_query_with_sources(self, question: str) -> AsyncGenerator[str, None]:
        """Streams the RAG response, yielding sources first, then the answer."""
        self._load_runner()
        
        if self._rag_runner is None:
            # This is now the only check we need.
            raise RuntimeError("No documents have been indexed. Please upload a document first.")
        
        # --- 1. Retrieve Context and Yield as Sources ---
        context_chunks = []
        context = self._rag_runner._retrieve_context(question)
        if context:
            context_chunks = context.split('\n\n')
            for i, chunk in enumerate(context_chunks):
                # The Vercel AI SDK expects a specific data prefix format for non-text parts.
                # 2 is the code for 'data' parts. We'll send JSON.
                source_data = json.dumps({
                    "type": "source-url",
                    "text": chunk.strip() # We put the chunk text here for display
                })
                yield f'2: {source_data}\n'
                
        
        # --- 2. Prepare Prompt and Yield the LLM Answer ---
        enhanced_message = f"Context:\n{context}\n\nQuestion: {question}" if context else question
        
        messages = []
        if self._rag_runner.config.system_prompt:
            messages.append(SystemMessage(content=self._rag_runner.config.system_prompt))
        
        messages.append(HumanMessage(content=enhanced_message))
        
        try:
            is_reasoning = False
            for chunk in self._rag_runner.chat_model.stream(messages): # type: ignore
                if '<think>' in chunk:
                    is_reasoning = True
                    chunk = chunk.replace('<think>', '')
                if '</think>' in chunk:
                    is_reasoning = False
                    chunk = chunk.replace('</think>', '')
                
                if is_reasoning and chunk.strip():
                    reasoning_obj = {"type": "reasoning", "delta": chunk}
                    yield f"{json.dumps(reasoning_obj)}\n"
                elif chunk.strip():
                    text_obj = {"type": "text", "delta": chunk}
                    yield f"{json.dumps(text_obj)}\n"
        except Exception as e:
            error_obj = {"type": "error", "message": str(e)}
            yield f"{json.dumps(error_obj)}\n"
