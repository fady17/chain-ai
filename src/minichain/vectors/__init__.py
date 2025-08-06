# src/minichain/vectors/__init__.py
"""
This module provides classes for storing and retrieving vectorized data.
Optional stores are available based on installed dependencies.
"""
from minichain.vectors.base import BaseVectorStore
# from ..memory.buffer import ConversationBufferMemory

# A list to hold the names of all successfully imported classes.
__all__ = ["BaseVectorStore"]
# __all__.append("ConversationBufferMemory")

# --- Graceful import for FAISS ---
try:
    from minichain.vectors.faiss import FAISSVectorStore # type: ignore
    __all__.append("FAISSVectorStore")
except ImportError:
    # This block is executed if 'faiss' or 'numpy' is not installed.
    # We define a placeholder class that raises a helpful error when used.
    class FAISSVectorStore:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "FAISS dependencies not found. Please run `pip install minichain-ai[local]` "
                "or `pip install minichain-ai[gpu]` to use FAISSVectorStore."
            )

# --- Graceful import for Azure AI Search ---
try:
    from .azure_ai_search import AzureAISearchVectorStore # type: ignore
    __all__.append("AzureAISearchVectorStore")
except ImportError:
    class AzureAISearchVectorStore:
         def __init__(self, *args, **kwargs):
            raise ImportError(
                "Azure Search dependencies not found. Please run `pip install minichain-ai[azure]` "
                "to use AzureAISearchVectorStore."
            )
# """
# This module provides classes for storing and retrieving vectorized data.
# """
# from .base import BaseVectorStore

# # --- Graceful import for FAISS ---
# try:
#     from .faiss import FAISSVectorStore
#     _faiss_installed = True
# except ImportError:
#     _faiss_installed = False

# # --- Graceful import for Azure AI Search ---
# try:
#     from .azure_ai_search import AzureAISearchVectorStore
#     _azure_search_installed = True
# except ImportError:
#     _azure_search_installed = False


# # Define the public API with __all__
# __all__ = ["BaseVectorStore"]

# if _faiss_installed:
#     __all__.append("FAISSVectorStore")
# if _azure_search_installed:
#     __all__.append("AzureAISearchVectorStore")