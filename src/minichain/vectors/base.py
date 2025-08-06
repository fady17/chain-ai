# src/minichain/memory/base.py
"""
Defines the abstract base class for all vector stores in the Mini-Chain framework.
"""
from abc import ABC, abstractmethod
from typing import List, Type, Tuple, TypeVar

from minichain.core.types import Document
from minichain.embeddings.base import BaseEmbeddings

# --- FIX: Introduce a TypeVar bound to BaseVectorStore ---
# This allows our factory method to return the specific subclass type.
Self = TypeVar("Self", bound="BaseVectorStore")

class BaseVectorStore(ABC):
    """Abstract base class for all vector stores."""

    def __init__(self, embeddings: BaseEmbeddings, **kwargs):
        self.embedding_function = embeddings

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add a list of documents to the vector store."""
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Perform a similarity search and return documents with scores."""
        pass
    
    @classmethod
    def from_documents(
        # --- FIX: Use the generic TypeVar for the class and return type ---
        cls: Type[Self],
        documents: List[Document],
        embeddings: BaseEmbeddings,
        **kwargs
    ) -> Self:
        """
        A generic factory method that creates a vector store from documents.
        This will return an instance of the specific subclass it is called on
        (e.g., FAISSVectorStore, AzureAISearchVectorStore).
        """
        # The 'cls' here will be the specific subclass, e.g., FAISSVectorStore
        store = cls(embeddings=embeddings, **kwargs)
        store.add_documents(documents)
        return store
