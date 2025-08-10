# src/chain/vectors/faiss.py

import os
import pickle
from typing import List, Dict, Literal, Optional, Any, Tuple, Union
import numpy as np
from uuid import uuid4

try:
    import faiss
    import numpy as np
except ImportError:
    raise ImportError(
        "FAISS dependencies not found. Please run `pip install chain-ai[local]` "
        "or `pip install chain-ai[gpu]` to use FAISSVectorStore."
    )

from ..core.types import Document
from ..embeddings.base import BaseEmbeddings
from .base import BaseVectorStore

FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')

FaissIndexType = Literal["IndexFlatL2", "IndexIVFFlat"]

class FAISSVectorStore(BaseVectorStore):
    """A vector store using FAISS that supports CPU/GPU and flexible initialization with custom IDs and filtering."""
    
    def __init__(self, embeddings: BaseEmbeddings, device: str = "cpu", **kwargs: Any):
        super().__init__(embeddings=embeddings, **kwargs)
        self.index: Optional[faiss.Index] = None
        self._docstore: Dict[str, Document] = {}  # Changed to use string IDs
        self._index_to_docstore_id: List[str] = []  # Changed to store string IDs
        self.device = device
        self._gpu_resources: Optional[Any] = None
        
        if self.device == "cuda":
            if not FAISS_GPU_AVAILABLE:
                raise ImportError("FAISS GPU library not installed or CUDA not available.")
            self._gpu_resources = faiss.StandardGpuResources() # type: ignore

    def __len__(self) -> int:
        """Returns the number of documents in the store."""
        return len(self._docstore)

    def add_documents( # type: ignore
        self, 
        documents: List[Document], 
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Embeds documents and adds them to the FAISS index.
        
        Args:
            documents: List of documents to add
            ids: Optional list of custom IDs. If not provided, UUIDs will be generated
            
        Returns:
            List of document IDs that were used
        """
        if not documents:
            return []
            
        # Generate IDs if not provided (backward compatibility)
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]
        elif len(ids) != len(documents):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of documents ({len(documents)})")
        
        # Check for duplicate IDs
        existing_ids = set(self._docstore.keys())
        duplicate_ids = set(ids) & existing_ids
        if duplicate_ids:
            raise ValueError(f"Document IDs already exist: {duplicate_ids}")
        
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        vectors_np = np.array(vectors, dtype=np.float32)

        if self.index is None:
            dimension = vectors_np.shape[1]
            # Use a simple, extendable index by default
            cpu_index = faiss.IndexFlatL2(dimension)
            if self.device == "cuda" and self._gpu_resources:
                self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index) # type: ignore
            else:
                self.index = cpu_index
        
        self.index.add(vectors_np) # type: ignore

        # Store documents with their custom IDs
        for doc_id, doc in zip(ids, documents):
            self._docstore[doc_id] = doc
            self._index_to_docstore_id.append(doc_id)
            
        return ids

    def _filter_documents(
        self, 
        documents_with_scores: List[Tuple[Document, float]], 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Filter documents based on metadata criteria.
        
        Args:
            documents_with_scores: List of (document, score) tuples
            filter_dict: Dictionary of metadata key-value pairs to filter by
            
        Returns:
            Filtered list of (document, score) tuples
        """
        if not filter_dict:
            return documents_with_scores
            
        filtered_results = []
        for doc, score in documents_with_scores:
            # Check if document metadata matches all filter criteria
            matches = True
            for key, value in filter_dict.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    matches = False
                    break
            if matches:
                filtered_results.append((doc, score))
                
        return filtered_results

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Performs a similarity search with optional metadata filtering.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional dictionary of metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        if self.index is None or len(self._docstore) == 0:
            return []
            
        query_vector = self.embedding_function.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)

        # If we have filters, we might need to search more documents initially
        # then filter them down to k results
        search_k = k
        if filter:
            # Search for more documents to account for filtering
            # This is a simple heuristic - you might want to adjust based on your data
            search_k = min(k * 10, len(self._docstore))

        # Ensure search_k is not greater than the number of vectors in the index
        search_k = min(search_k, len(self._docstore))
        distances, indices = self.index.search(query_vector_np, search_k) # type: ignore

        valid_mask = indices[0] != -1
        valid_indices = indices[0][valid_mask]
        valid_distances = distances[0][valid_mask]
        
        # Get all results before filtering
        all_results = [
            (self._docstore[self._index_to_docstore_id[i]], float(dist))
            for i, dist in zip(valid_indices, valid_distances)
        ]
        
        # Apply metadata filtering if specified
        if filter:
            filtered_results = self._filter_documents(all_results, filter)
            # Return up to k filtered results
            return filtered_results[:k]
        else:
            return all_results[:k]

    def similarity_search_without_scores(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Performs a similarity search returning only documents (no scores).
        """
        results_with_scores = self.similarity_search(query, k=k, filter=filter, **kwargs)
        return [doc for doc, _ in results_with_scores]

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve
            
        Returns:
            List of documents (missing IDs will be skipped)
        """
        documents = []
        for doc_id in ids:
            if doc_id in self._docstore:
                documents.append(self._docstore[doc_id])
        return documents

    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Note: This is a simplified implementation. For production use,
        you might want to rebuild the FAISS index to actually remove vectors.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if any documents were deleted
        """
        deleted_any = False
        for doc_id in ids:
            if doc_id in self._docstore:
                del self._docstore[doc_id]
                deleted_any = True
                
        # Remove from index mapping (this doesn't remove from FAISS index itself)
        self._index_to_docstore_id = [
            id_ for id_ in self._index_to_docstore_id if id_ not in ids
        ]
        
        return deleted_any

    def update_document(self, doc_id: str, document: Document) -> bool:
        """
        Update a document by its ID.
        
        Note: This only updates the document store, not the vector embeddings.
        For embedding updates, you'd need to delete and re-add the document.
        
        Args:
            doc_id: ID of the document to update
            document: New document content
            
        Returns:
            True if document was updated, False if ID not found
        """
        if doc_id in self._docstore:
            self._docstore[doc_id] = document
            return True
        return False

    def get_all_document_ids(self) -> List[str]:
        """Get all document IDs in the store."""
        return list(self._docstore.keys())

    def save_local(self, folder_path: str):
        """Saves the FAISS index and document store to a local folder."""
        if self.index is None:
            raise ValueError("Cannot save an empty or uninitialized vector store.")
            
        os.makedirs(folder_path, exist_ok=True)
        index_path = os.path.join(folder_path, "index.faiss")
        docstore_path = os.path.join(folder_path, "docstore.pkl")
        
        index_to_save = self.index
        if FAISS_GPU_AVAILABLE and faiss.get_num_gpus() > 0 and hasattr(self.index, 'setNumProbes'):
             index_to_save = faiss.index_gpu_to_cpu(self.index) # type: ignore
        
        faiss.write_index(index_to_save, index_path)
        
        with open(docstore_path, "wb") as f:
            pickle.dump((self._docstore, self._index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings: BaseEmbeddings, device: str = "cpu"
    ) -> "FAISSVectorStore":
        """Loads a FAISSVectorStore from a local folder to the specified device."""
        index_path = os.path.join(folder_path, "index.faiss")
        docstore_path = os.path.join(folder_path, "docstore.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(docstore_path): 
            raise FileNotFoundError(f"Vector store files not found in: {folder_path}")

        # The index_type is determined by the loaded file, so we can set a default.
        store = cls(embeddings=embeddings, device=device, index_type="IndexFlatL2")
        
        cpu_index = faiss.read_index(index_path)
        
        if store.device == "cuda" and store._gpu_resources is not None:
            store.index = faiss.index_cpu_to_gpu(store._gpu_resources, 0, cpu_index) # type: ignore
        else:
            store.index = cpu_index
        
        with open(docstore_path, "rb") as f:
            docstore_data, index_to_docstore_data = pickle.load(f)
            
            # Handle both old (int keys) and new (string keys) formats
            if docstore_data and isinstance(list(docstore_data.keys())[0], int):
                # Convert old format to new format
                store._docstore = {str(k): v for k, v in docstore_data.items()}
                store._index_to_docstore_id = [str(x) for x in index_to_docstore_data]
            else:
                store._docstore = docstore_data
                store._index_to_docstore_id = index_to_docstore_data
            
        return store
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: BaseEmbeddings,
        device: str = "cpu",
        index_type: FaissIndexType = "IndexFlatL2",
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> "FAISSVectorStore":
        """Creates a FAISSVectorStore from documents with optional custom IDs."""
        store = cls(embeddings=embeddings, device=device, index_type=index_type, **kwargs)
        if documents:
            store.add_documents(documents, ids=ids)
        return store
