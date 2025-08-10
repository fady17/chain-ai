# ===== app/rag_service.py =====
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from chain.rag_runner import create_smart_rag, RAGRunner
from pathlib import Path
from typing import Dict, List, Optional
import uuid
import time
from .config import settings
from .models import DocumentType

class RAGService:
    """Singleton service that manages RAG runners and document indexing"""
    
    _instance = None
    _rag_runner: Optional[RAGRunner] = None
    _document_registry: Dict[str, dict] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            settings.upload_dir.mkdir(exist_ok=True)
    
    def add_document(self, file_path: Path) -> dict:
        """Add a document to the RAG system"""
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
            
            return {
                'document_id': document_id,
                'document_type': doc_type,
                'chunks_created': self._estimate_chunks(file_path)
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to process document: {str(e)}")
    
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
