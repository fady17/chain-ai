# src/minichain/memory/__init__.py
"""
This module provides classes for storing and retrieving vectorized data.
"""
from .base import BaseVectorStore

# --- Graceful import for FAISS ---
try:
    from .faiss import FAISSVectorStore
    _faiss_installed = True
except ImportError:
    _faiss_installed = False

# --- Graceful import for Azure AI Search ---
try:
    from .azure_ai_search import AzureAISearchVectorStore
    _azure_search_installed = True
except ImportError:
    _azure_search_installed = False


# Define the public API with __all__
__all__ = ["BaseVectorStore"]

if _faiss_installed:
    __all__.append("FAISSVectorStore")
if _azure_search_installed:
    __all__.append("AzureAISearchVectorStore")