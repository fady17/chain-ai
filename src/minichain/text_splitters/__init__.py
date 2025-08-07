# src/minichain/text_splitters/__init__.py
"""
This module provides classes for splitting text into smaller chunks.
The TokenTextSplitter is available as an optional dependency.
"""
from .base import BaseTextSplitter
from .character import RecursiveCharacterTextSplitter


__all__ = [
    "BaseTextSplitter",
    "RecursiveCharacterTextSplitter",
    
   
]

# --- Graceful import for TokenTextSplitter ---
try:
    from .token import TokenTextSplitter # type: ignore
    __all__.append("TokenTextSplitter")
except ImportError:
    class TokenTextSplitter:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "TikToken dependency not found. Please run `pip install minichain-ai[token_splitter]` "
                "to use TokenTextSplitter."
            )
# """
# This module provides classes for splitting large pieces of text into smaller,
# semantically meaningful chunks. This is a crucial preprocessing step for
# many RAG (Retrieval-Augmented Generation) applications.

# The key components exposed are:
#     - TokenTextSplitter: The recommended, modern splitter that operates on
#       language model tokens. It is language-agnostic and respects model
#       context window limits accurately.
#     - RecursiveCharacterTextSplitter: A flexible splitter that operates on
#       characters, attempting to split on semantic boundaries like paragraphs
#       and sentences first.
# """
# from .base import BaseTextSplitter
# from .character import RecursiveCharacterTextSplitter
# from .token import TokenTextSplitter
# from .streaming import StreamingArabicSentenceSplitter
# __all__ = [
#     "BaseTextSplitter",
#     "RecursiveCharacterTextSplitter",
#     "TokenTextSplitter",
#     "StreamingArabicSentenceSplitter",
# ]