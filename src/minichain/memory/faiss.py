# src/minichain/memory/faiss.py
"""
An in-memory vector store implementation using FAISS, with optional GPU support.
"""
import os
import pickle
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is not installed. Please install `faiss-cpu` for CPU support "
        "or `faiss-gpu` for NVIDIA GPU support."
    )

from ..core.types import Document
from ..embeddings.base import BaseEmbeddings
from .base import BaseVectorStore

# Perform a single, reliable check for GPU availability at import time.
# `StandardGpuResources` is a key class that only exists in the GPU build.
FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources')

class FAISSVectorStore(BaseVectorStore):
    """
    A vector store using FAISS that supports both CPU and CUDA GPU devices.
    """
    def __init__(self, embeddings: BaseEmbeddings, device: str = "cpu", **kwargs: Any):
        super().__init__(embeddings=embeddings, **kwargs)
        self.index: Optional[faiss.Index] = None
        self._docstore: Dict[int, Document] = {}
        self._index_to_docstore_id: List[int] = []
        
        self.device = device
        self._gpu_resources: Optional[Any] = None
        
        if self.device == "cuda":
            if not FAISS_GPU_AVAILABLE:
                raise ImportError(
                    "FAISS GPU library is not installed or CUDA is not available. "
                    "Please install `faiss-gpu`."
                )
            # This is where the linter might complain, but our check above ensures it's safe.
            self._gpu_resources = faiss.StandardGpuResources() # type: ignore[attr-defined]

    def add_documents(self, documents: List[Document]) -> None:
        """Embeds documents and adds them to the FAISS index."""
        if not documents: return
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        if not vectors: return
        vectors_np = np.array(vectors, dtype=np.float32)

        if self.index is None:
            self._create_index(vectors_np.shape[1])

        assert self.index is not None, "FAISS index must be initialized before adding documents."
        
        # The high-level Python API for an Index object's `add` method takes only the
        # numpy array of vectors. The type stubs are confusing the linter.
        self.index.add(vectors_np) # type: ignore[arg-type]

        start_id = len(self._docstore)
        for i, doc in enumerate(documents):
            doc_id = start_id + i
            self._docstore[doc_id] = doc
            self._index_to_docstore_id.append(doc_id)

    def _create_index(self, dimension: int):
        """Internal method to create a new FAISS index on the configured device."""
        cpu_index = faiss.IndexFlatL2(dimension)
        
        if self.device == "cuda" and self._gpu_resources is not None:
            # We add a type ignore here because Pylance cannot statically see
            # the dynamically loaded GPU symbols. Our runtime check handles safety.
            self.index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, cpu_index) # type: ignore[attr-defined]
        else:
            self.index = cpu_index

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Performs a similarity search."""
        if self.index is None: return []

        query_vector = self.embedding_function.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)

        # The high-level API for search is simpler than the full C++ signature.
        distances, indices = self.index.search(query_vector_np, k) # type: ignore[arg-type]

        results: List[Tuple[Document, float]] = []
        for i, dist in zip(indices[0], distances[0]):
            if i != -1:
                docstore_id = self._index_to_docstore_id[i]
                document = self._docstore[docstore_id]
                results.append((document, float(dist)))
        
        return results

    def save_local(self, folder_path: str):
        """Saves the FAISS index and document store to a local folder."""
        if self.index is None: raise ValueError("Cannot save an empty vector store.")
            
        os.makedirs(folder_path, exist_ok=True)
        index_path = os.path.join(folder_path, "index.faiss")
        docstore_path = os.path.join(folder_path, "docstore.pkl")
        
        # A GPU index must be moved to CPU before saving.
        if FAISS_GPU_AVAILABLE and isinstance(self.index, faiss.GpuIndex): # type: ignore[attr-defined]
            cpu_index = faiss.index_gpu_to_cpu(self.index) # type: ignore[attr-defined]
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        
        with open(docstore_path, "wb") as f:
            pickle.dump((self._docstore, self._index_to_docstore_id), f)

    @classmethod
    def load_local(
        cls, folder_path: str, embeddings: BaseEmbeddings, device: str = "cpu"
    ) -> "FAISSVectorStore":
        """Loads a FAISSVectorStore from a local folder to the specified device."""
        index_path = os.path.join(folder_path, "index.faiss")
        docstore_path = os.path.join(folder_path, "docstore.pkl")
        if not os.path.exists(index_path): raise FileNotFoundError(f"Index file not found: {index_path}")

        store = cls(embeddings=embeddings, device=device)
        
        cpu_index = faiss.read_index(index_path)
        
        if store.device == "cuda" and store._gpu_resources is not None:
            store.index = faiss.index_cpu_to_gpu(store._gpu_resources, 0, cpu_index) # type: ignore[attr-defined]
        else:
            store.index = cpu_index
        
        with open(docstore_path, "rb") as f:
            store._docstore, store._index_to_docstore_id = pickle.load(f)
            
        return store
# # src/minichain/memory/faiss.py
# """
# FAISS vector store implementation.
# """
# import faiss
# import numpy as np
# from typing import List, Dict, Optional, Any
# from ..core.types import Document
# from ..embeddings.base import BaseEmbeddings
# from .base import BaseVectorStore

# class FAISSVectorStore(BaseVectorStore):
#     """
#     A vector store that uses Facebook AI Similarity Search (FAISS) for
#     efficient, local similarity searches.
#     """

#     def __init__(self, embeddings: BaseEmbeddings, **kwargs: Any):
#         super().__init__(embeddings=embeddings, **kwargs)
#         self.index: Optional[faiss.Index] = None
        
#         # Mappings to retrieve documents from FAISS indices
#         # _docstore: maps a unique ID to each Document
#         # _index_to_docstore_id: maps a FAISS index position to a docstore ID
#         self._docstore: Dict[int, Document] = {}
#         self._index_to_docstore_id: List[int] = []

#     def add_documents(self, documents: List[Document]) -> None:
#         """
#         Adds documents to the FAISS index.

#         It embeds the document texts and stores them in a FAISS index,
#         while keeping a mapping to the original Document objects.
#         """
#         if not documents:
#             return

#         # Embed all document texts in a single batch
#         texts = [doc.page_content for doc in documents]
#         vectors = self.embedding_function.embed_documents(texts)
#         vectors_np = np.array(vectors, dtype=np.float32)

#         if self.index is None:
#             # First time adding documents, create the index
#             dimension = vectors_np.shape[1]
#             # Using IndexFlatL2 for exact, brute-force L2 distance search
#             self.index = faiss.IndexFlatL2(dimension)
#             print(f"✅ Initialized FAISS index with dimension {dimension}.")

#         # Add new vectors to the existing index
#         self.index.add(vectors_np) # type: ignore
        
#         # Store the documents and update mappings
#         start_index = len(self._docstore)
#         for i, doc in enumerate(documents):
#             doc_id = start_index + i
#             self._docstore[doc_id] = doc
#             self._index_to_docstore_id.append(doc_id)
        
#         print(f"✅ Added {len(documents)} documents. Index now contains {self.index.ntotal} vectors.")

#     def similarity_search(self, query: str, k: int = 4) -> List[Document]:
#         """
#         Performs a similarity search using the FAISS index.
#         """
#         if self.index is None:
#             raise ValueError("Vector store is empty. Please add documents before searching.")

#         # 1. Embed the query
#         query_vector = self.embedding_function.embed_query(query)
#         query_vector_np = np.array([query_vector], dtype=np.float32)

#         # 2. Search the FAISS index
#         # The search returns distances and the indices of the nearest neighbors
#         distances, indices = self.index.search(query_vector_np, k) # type: ignore

#         # 3. Retrieve the corresponding documents
#         results: List[Document] = []
#         for i in indices[0]:
#             # FAISS returns -1 if there are fewer than k results
#             if i != -1:
#                 docstore_id = self._index_to_docstore_id[i]
#                 document = self._docstore[docstore_id]
#                 results.append(document)
        
#         return results