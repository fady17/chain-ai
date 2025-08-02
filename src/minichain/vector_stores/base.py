# src/minichain/vector_stores/base.py
"""
Base vector store interface.
"""
from abc import ABC, abstractmethod
from typing import List, Type
from ..core.types import Document
from ..embeddings.base import BaseEmbeddings

class BaseVectorStore(ABC):
    """Abstract base class for all vector stores."""

    def __init__(self, embeddings: BaseEmbeddings, **kwargs):
        """
        Initializes the vector store with an embedding model.
        
        Args:
            embeddings: An object that inherits from BaseEmbeddings.
        """
        self.embedding_function = embeddings

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add a list of documents to the vector store.

        This involves embedding the documents and storing them.
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform a similarity search for a given query.

        Args:
            query: The query string to search for.
            k: The number of results to return.

        Returns:
            A list of the k most similar Document objects.
        """
        pass
    
    @classmethod
    def from_documents(
        cls: Type['BaseVectorStore'],
        documents: List[Document],
        embeddings: BaseEmbeddings,
        **kwargs
    ) -> 'BaseVectorStore':
        """
        Convenience method to create a vector store from documents.
        
        This creates an instance and adds the documents in one step.
        """
        store = cls(embeddings=embeddings, **kwargs)
        store.add_documents(documents)
        return store