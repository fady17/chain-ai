# src/minichain/embeddings/base.py
"""
Base embeddings interface
"""

from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddings(ABC):
    """Abstract base class for all embedding models"""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: A list of documents to embed.

        Returns:
            A list of embeddings, one for each document.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: The query text to embed.

        Returns:
            The embedding for the query text.
        """
        pass

    def __call__(self, text: str) -> List[float]:
        """Syntactic sugar for embed_query."""
        return self.embed_query(text)