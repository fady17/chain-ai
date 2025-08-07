# src/minichain/memory/faiss.py
"""
An in-memory vector store implementation using FAISS, with optional GPU support.
Lightweight optimized version with smart index selection.
"""
import os
import pickle
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

try:
    import faiss
    import numpy as np
except ImportError:
    raise ImportError(
        "FAISS dependencies not found. Please run `pip install minichain-ai[local]` "
        "or `pip install minichain-ai[gpu]` to use FAISSVectorStore."
    )

from ..core.types import Document
from ..embeddings.base import BaseEmbeddings
from .base import BaseVectorStore

FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')

class FAISSVectorStore(BaseVectorStore):
    """
    A vector store using FAISS that supports both CPU and CUDA GPU devices.
    Automatically optimizes index type based on dataset size.
    """
    def __init__(self, embeddings: BaseEmbeddings, device: str = "cpu", **kwargs: Any):
        super().__init__(embeddings=embeddings, **kwargs)
        self.index: Optional[faiss.Index] = None # type: ignore
        self._docstore: Dict[int, Document] = {}
        self._index_to_docstore_id: List[int] = []
        
        self.device = device
        self._gpu_resources: Optional[Any] = None
        
        if self.device == "cuda":
            if not FAISS_GPU_AVAILABLE:
                raise ImportError(
                    "FAISS GPU library is not installed or CUDA is not available. "
                    "Please install `minichain-ai[gpu]`."
                )
            self._gpu_resources = faiss.StandardGpuResources() # type: ignore[attr-defined]

    def add_documents(self, documents: List[Document]) -> None:
        """Embeds documents and adds them to the FAISS index."""
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        vectors_np = np.array(vectors, dtype=np.float32)

        if self.index is None:
            self._create_index(vectors_np.shape[1], len(documents))
        
        # Train index if needed (IVF only)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained: # type: ignore
            self.index.train(vectors_np) # type: ignore
        
        self.index.add(vectors_np) # type: ignore[arg-type]

        start_id = len(self._docstore)
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            self._docstore[doc_id] = doc
            self._index_to_docstore_id.append(doc_id)

    def _create_index(self, dimension: int, num_docs: int = 0):
        """Internal method to create an optimized FAISS index."""
        # Smart index selection based on document count
        if num_docs < 1000:
            # Small datasets: use exact search
            cpu_index = faiss.IndexFlatL2(dimension)
        else:
            # Medium+ datasets: use IVF for ~5-10x speedup
            nlist = min(max(int(num_docs / 10), 10), 100)  # 10-100 clusters
            quantizer = faiss.IndexFlatL2(dimension)
            cpu_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        if self.device == "cuda" and self._gpu_resources is not None:
            self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index) # type: ignore[attr-defined]
        else:
            self.index = cpu_index

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Performs a similarity search."""
        if self.index is None:
            return []
        query_vector = self.embedding_function.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)

        distances, indices = self.index.search(query_vector_np, k) # type: ignore[arg-type]

        # Vectorized processing - faster than loop with conditionals
        valid_mask = indices[0] != -1
        valid_indices = indices[0][valid_mask]
        valid_distances = distances[0][valid_mask]
        
        return [
            (self._docstore[self._index_to_docstore_id[i]], float(dist))
            for i, dist in zip(valid_indices, valid_distances)
        ]

    def save_local(self, folder_path: str):
        """Saves the FAISS index and document store to a local folder."""
        os.makedirs(folder_path, exist_ok=True)
        index_path = os.path.join(folder_path, "index.faiss")
        docstore_path = os.path.join(folder_path, "docstore.pkl")
        
        # A GPU index must be moved to CPU before saving.
        if FAISS_GPU_AVAILABLE and isinstance(self.index, faiss.GpuIndex): # type: ignore[attr-defined]
            cpu_index = faiss.index_gpu_to_cpu(self.index) # type: ignore[attr-defined]
            faiss.write_index(cpu_index, index_path) # type: ignore
        else:
            faiss.write_index(self.index, index_path) # type: ignore
        
        with open(docstore_path, "wb") as f:
            pickle.dump((self._docstore, self._index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings: BaseEmbeddings, device: str = "cpu"
    ) -> "FAISSVectorStore":
        """Loads a FAISSVectorStore from a local folder to the specified device."""
        index_path = os.path.join(folder_path, "index.faiss")
        docstore_path = os.path.join(folder_path, "docstore.pkl")
        
        if not os.path.exists(index_path): 
            raise FileNotFoundError(f"Index file not found: {index_path}")

        store = cls(embeddings=embeddings, device=device)
        cpu_index = faiss.read_index(index_path) # type: ignore
        
        if store.device == "cuda" and store._gpu_resources is not None:
            store.index = faiss.index_cpu_to_gpu(store._gpu_resources, 0, cpu_index) # type: ignore[attr-defined]
        else:
            store.index = cpu_index
        
        with open(docstore_path, "rb") as f:
            store._docstore, store._index_to_docstore_id = pickle.load(f)
            
        return store
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embeddings: BaseEmbeddings,
        device: str = "cpu",
        **kwargs: Any
    ) -> "FAISSVectorStore":
        store = cls(embeddings=embeddings, device=device, **kwargs)
        store.add_documents(documents)
        return store
# # src/minichain/memory/faiss.py
# """
# An in-memory vector store implementation using FAISS, with optional GPU support.
# """
# import os
# import pickle
# from typing import List, Dict, Optional, Any, Tuple
# import numpy as np

# try:
#     import faiss
#     import numpy as np
# except ImportError:
#     raise ImportError(
#         "FAISS dependencies not found. Please run `pip install minichain-ai[local]` "
#         "or `pip install minichain-ai[gpu]` to use FAISSVectorStore."
#     )


# from ..core.types import Document
# from ..embeddings.base import BaseEmbeddings
# from .base import BaseVectorStore

# # Perform a single, reliable check for GPU availability at import time.
# # `StandardGpuResources` is a key class that only exists in the GPU build.
# FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')

# class FAISSVectorStore(BaseVectorStore):
#     """
#     A vector store using FAISS that supports both CPU and CUDA GPU devices.
#     """
#     def __init__(self, embeddings: BaseEmbeddings, device: str = "cpu", **kwargs: Any):
#         super().__init__(embeddings=embeddings, **kwargs)
#         self.index: Optional[faiss.Index] = None # type: ignore
#         self._docstore: Dict[int, Document] = {}
#         self._index_to_docstore_id: List[int] = []
        
#         self.device = device
#         self._gpu_resources: Optional[Any] = None
        
#         if self.device == "cuda":
#             if not FAISS_GPU_AVAILABLE:
#                 raise ImportError(
#                     "FAISS GPU library is not installed or CUDA is not available. "
#                     "Please install `faiss-gpu`."
#                 )
#             # This is where the linter might complain, but our check above ensures it's safe.
#             self._gpu_resources = faiss.StandardGpuResources() # type: ignore[attr-defined]

#     def add_documents(self, documents: List[Document]) -> None:
#         """Embeds documents and adds them to the FAISS index."""
#         if not documents: return
#         texts = [doc.page_content for doc in documents]
#         vectors = self.embedding_function.embed_documents(texts)
#         if not vectors: return
#         vectors_np = np.array(vectors, dtype=np.float32)

#         if self.index is None:
#             self._create_index(vectors_np.shape[1])

#         assert self.index is not None, "FAISS index must be initialized before adding documents."
        
#         # The high-level Python API for an Index object's `add` method takes only the
#         # numpy array of vectors. The type stubs are confusing the linter.
#         self.index.add(vectors_np) # type: ignore[arg-type]

#         start_id = len(self._docstore)
#         for i, doc in enumerate(documents):
#             doc_id = start_id + i
#             self._docstore[doc_id] = doc
#             self._index_to_docstore_id.append(doc_id)

#     def _create_index(self, dimension: int):
#         """Internal method to create a new FAISS index on the configured device."""
#         cpu_index = faiss.IndexFlatL2(dimension) # type: ignore
        
#         if self.device == "cuda" and self._gpu_resources is not None:
#             # We add a type ignore here because Pylance cannot statically see
#             # the dynamically loaded GPU symbols. Our runtime check handles safety.
#             self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index) # type: ignore[attr-defined]
#         else:
#             self.index = cpu_index

#     def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
#         """Performs a similarity search."""
#         if self.index is None: return []

#         query_vector = self.embedding_function.embed_query(query)
#         query_vector_np = np.array([query_vector], dtype=np.float32)

#         # The high-level API for search is simpler than the full C++ signature.
#         distances, indices = self.index.search(query_vector_np, k) # type: ignore[arg-type]

#         results: List[Tuple[Document, float]] = []
#         for i, dist in zip(indices[0], distances[0]):
#             if i != -1:
#                 docstore_id = self._index_to_docstore_id[i]
#                 document = self._docstore[docstore_id]
#                 results.append((document, float(dist)))
        
#         return results

#     def save_local(self, folder_path: str):
#         """Saves the FAISS index and document store to a local folder."""
#         if self.index is None: raise ValueError("Cannot save an empty vector store.")
            
#         os.makedirs(folder_path, exist_ok=True)
#         index_path = os.path.join(folder_path, "index.faiss")
#         docstore_path = os.path.join(folder_path, "docstore.pkl")
        
#         # A GPU index must be moved to CPU before saving.
#         if FAISS_GPU_AVAILABLE and isinstance(self.index, faiss.GpuIndex): # type: ignore[attr-defined]
#             cpu_index = faiss.index_gpu_to_cpu(self.index) # type: ignore[attr-defined]
#             faiss.write_index(cpu_index, index_path) # type: ignore
#         else:
#             faiss.write_index(self.index, index_path) # type: ignore
        
#         with open(docstore_path, "wb") as f:
#             pickle.dump((self._docstore, self._index_to_docstore_id), f)

#     @classmethod
#     def load_local(
#         cls, folder_path: str, embeddings: BaseEmbeddings, device: str = "cpu"
#     ) -> "FAISSVectorStore":
#         """Loads a FAISSVectorStore from a local folder to the specified device."""
#         index_path = os.path.join(folder_path, "index.faiss")
#         docstore_path = os.path.join(folder_path, "docstore.pkl")
#         if not os.path.exists(index_path): raise FileNotFoundError(f"Index file not found: {index_path}")

#         store = cls(embeddings=embeddings, device=device)
        
#         cpu_index = faiss.read_index(index_path) # type: ignore
        
#         if store.device == "cuda" and store._gpu_resources is not None:
#             store.index = faiss.index_cpu_to_gpu(store._gpu_resources, 0, cpu_index) # type: ignore[attr-defined]
#         else:
#             store.index = cpu_index
        
#         with open(docstore_path, "rb") as f:
#             store._docstore, store._index_to_docstore_id = pickle.load(f)
            
#         return store
    
#     @classmethod
#     def from_documents(
#         cls,
#         documents: List[Document],
#         embeddings: BaseEmbeddings,
#         device: str = "cpu",
#         **kwargs: Any
#     ) -> "FAISSVectorStore":
#         store = cls(embeddings=embeddings, device=device, **kwargs)
#         store.add_documents(documents)
#         return store
