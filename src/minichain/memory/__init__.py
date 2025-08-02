# src/minichain/memory/__init__.py
"""
This module provides classes for storing and retrieving vectorized data,
a core component of "memory" in RAG applications.

The key components exposed are:
    - BaseVectorStore: The abstract interface for all vector stores.
    - FAISSVectorStore: An efficient, in-memory vector store ideal for
      local development and fast prototyping.
    - AzureAISearchVectorStore: A scalable, cloud-native vector store for
      production applications using Microsoft Azure.
"""
from .base import BaseVectorStore
from .faiss import FAISSVectorStore
from .azure_ai_search import AzureAISearchVectorStore

__all__ = [
    "BaseVectorStore",
    "FAISSVectorStore",
    "AzureAISearchVectorStore",
]