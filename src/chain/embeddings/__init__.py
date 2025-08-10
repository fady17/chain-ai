# src/chain/embeddings/__init__.py
"""
This module provides classes for generating vector embeddings from text,
supporting both cloud-based and local models.

The key components exposed are:
    - BaseEmbeddings: The abstract interface for all embedding models.
    - AzureOpenAIEmbeddings: For generating embeddings using Azure OpenAI.
    - LocalEmbeddings: For generating embeddings using a local, OpenAI-compatible
      server like LM Studio.
"""
from dotenv import load_dotenv
from .base import BaseEmbeddings
from .azure import AzureOpenAIEmbeddings
from .local import LocalEmbeddings

load_dotenv()
__all__ = [
    "BaseEmbeddings",
    "AzureOpenAIEmbeddings",
    "LocalEmbeddings",
]