# src/minichain/text_splitters/__init__.py
"""
This module provides classes for splitting large pieces of text into smaller,
semantically meaningful chunks. This is a crucial preprocessing step for
many RAG (Retrieval-Augmented Generation) applications.

The key components exposed are:
    - TokenTextSplitter: The recommended, modern splitter that operates on
      language model tokens. It is language-agnostic and respects model
      context window limits accurately.
    - RecursiveCharacterTextSplitter: A flexible splitter that operates on
      characters, attempting to split on semantic boundaries like paragraphs
      and sentences first.
"""
from .base import BaseTextSplitter
from .character import RecursiveCharacterTextSplitter
from .token import TokenTextSplitter

__all__ = [
    "BaseTextSplitter",
    "RecursiveCharacterTextSplitter",
    "TokenTextSplitter",
]