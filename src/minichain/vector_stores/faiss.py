# src/minichain/vector_stores/faiss.py
"""
FAISS vector store implementation.
"""
import faiss
import numpy as np
from typing import List, Dict, Optional, Any
from ..core.types import Document
from ..embeddings.base import BaseEmbeddings
from .base import BaseVectorStore

class FAISSVectorStore(BaseVectorStore):
    """
    A vector store that uses Facebook AI Similarity Search (FAISS) for
    efficient, local similarity searches.
    """

    def __init__(self, embeddings: BaseEmbeddings, **kwargs: Any):
        super().__init__(embeddings=embeddings, **kwargs)
        self.index: Optional[faiss.Index] = None
        
        # Mappings to retrieve documents from FAISS indices
        # _docstore: maps a unique ID to each Document
        # _index_to_docstore_id: maps a FAISS index position to a docstore ID
        self._docstore: Dict[int, Document] = {}
        self._index_to_docstore_id: List[int] = []

    def add_documents(self, documents: List[Document]) -> None:
        """
        Adds documents to the FAISS index.

        It embeds the document texts and stores them in a FAISS index,
        while keeping a mapping to the original Document objects.
        """
        if not documents:
            return

        # Embed all document texts in a single batch
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        vectors_np = np.array(vectors, dtype=np.float32)

        if self.index is None:
            # First time adding documents, create the index
            dimension = vectors_np.shape[1]
            # Using IndexFlatL2 for exact, brute-force L2 distance search
            self.index = faiss.IndexFlatL2(dimension)
            print(f"✅ Initialized FAISS index with dimension {dimension}.")

        # Add new vectors to the existing index
        self.index.add(vectors_np) # type: ignore
        
        # Store the documents and update mappings
        start_index = len(self._docstore)
        for i, doc in enumerate(documents):
            doc_id = start_index + i
            self._docstore[doc_id] = doc
            self._index_to_docstore_id.append(doc_id)
        
        print(f"✅ Added {len(documents)} documents. Index now contains {self.index.ntotal} vectors.")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search using the FAISS index.
        """
        if self.index is None:
            raise ValueError("Vector store is empty. Please add documents before searching.")

        # 1. Embed the query
        query_vector = self.embedding_function.embed_query(query)
        query_vector_np = np.array([query_vector], dtype=np.float32)

        # 2. Search the FAISS index
        # The search returns distances and the indices of the nearest neighbors
        distances, indices = self.index.search(query_vector_np, k) # type: ignore

        # 3. Retrieve the corresponding documents
        results: List[Document] = []
        for i in indices[0]:
            # FAISS returns -1 if there are fewer than k results
            if i != -1:
                docstore_id = self._index_to_docstore_id[i]
                document = self._docstore[docstore_id]
                results.append(document)
        
        return results