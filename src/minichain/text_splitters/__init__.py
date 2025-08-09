# src/minichain/text_splitters/__init__.py

"""
This module re-exports the robust and battle-tested text splitters, helpers,
and enums from the lightweight `langchain-text-splitters` package.

This provides a rich, consistent, and powerful text splitting API for users 
of `minichain` without reinventing the wheel. Users can now access a wide
variety of splitters for different languages and document types.
"""

# Import all the desired classes, enums, and helpers directly from the real library.
from langchain_text_splitters import (
    # Core Classes
    TextSplitter,
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    
    # Language-Specific Splitters
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
    LatexTextSplitter,
    KonlpyTextSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    JSFrameworkTextSplitter,
    
    # Document-Type-Specific Splitters
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveJsonSplitter,
    ExperimentalMarkdownSyntaxTextSplitter,

    # Tokenizer-Specific Splitters
    SentenceTransformersTokenTextSplitter,
    
    # Helper Enums and Functions
    Language,
    Tokenizer,
    ElementType,
    HeaderType,
    LineType,
    split_text_on_tokens,
)

# This __all__ list defines the public API of this module.
# It's a direct copy from the langchain-text-splitters __all__ list you provided,
# ensuring we expose everything.
__all__ = [
    "CharacterTextSplitter",
    "ElementType",
    "ExperimentalMarkdownSyntaxTextSplitter",
    "HTMLHeaderTextSplitter",
    "HTMLSectionSplitter",
    "HTMLSemanticPreservingSplitter",
    "HeaderType",
    "JSFrameworkTextSplitter",
    "KonlpyTextSplitter",
    "Language",
    "LatexTextSplitter",
    "LineType",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "NLTKTextSplitter",
    "PythonCodeTextSplitter",
    "RecursiveCharacterTextSplitter",
    "RecursiveJsonSplitter",
    "SentenceTransformersTokenTextSplitter",
    "SpacyTextSplitter",
    "TextSplitter",
    "TokenTextSplitter",
    "Tokenizer",
    "split_text_on_tokens",
]
