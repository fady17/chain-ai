# tests/text_splitters/test_token_splitter.py
"""
Unit tests for the `TokenTextSplitter`.

These tests validate that the splitter correctly chunks text based on token counts
and handles `Document` objects as expected.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.core.types import Document
from minichain.text_splitters import TokenTextSplitter

def test_token_splitter_chunks_long_text():
    """
    Tests that a text longer than the chunk size is split into multiple chunks.
    """
    # ARRANGE
    splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=2, model_name="gpt-4")
    text = "This is a test sentence for the token-based splitter component that is deliberately long."

    # ACT
    docs = splitter.create_documents(texts=[text])

    # ASSERT
    assert len(docs) > 1
    assert all(isinstance(doc, Document) for doc in docs)

def test_token_splitter_preserves_metadata_when_creating_documents():
    """
    Tests that the original metadata is correctly passed to all created chunks.
    """
    # ARRANGE
    splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=2, model_name="gpt-4")
    texts = ["A short test text."]
    metadatas = [{"source": "test_source.md", "id": 123}]

    # ACT
    docs = splitter.create_documents(texts=texts, metadatas=metadatas)

    # ASSERT
    assert docs[0].metadata["source"] == "test_source.md"
    assert docs[0].metadata["id"] == 123
    assert "chunk_index" in docs[0].metadata

def test_token_splitter_preserves_metadata_when_splitting_documents():
    """
    Tests that metadata from a pre-existing Document is preserved in the new chunks.
    """
    # ARRANGE
    splitter = TokenTextSplitter(chunk_size=5, chunk_overlap=1, model_name="gpt-4")
    original_doc = Document(page_content="A document to be split.", metadata={"author": "Amin"})
    
    # ACT
    split_docs = splitter.split_documents([original_doc])
    
    # ASSERT
    assert len(split_docs) > 1
    # Check metadata on the first and last chunks
    assert split_docs[0].metadata["author"] == "Amin"
    assert split_docs[-1].metadata["author"] == "Amin"
    assert "source_document_index" in split_docs[0].metadata

def test_token_splitter_handles_empty_input():
    """
    Tests that the splitter does not raise an error and returns an empty list
    for empty or whitespace-only text.
    """
    # ARRANGE
    splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=2)
    
    # ACT
    docs_from_empty = splitter.create_documents(texts=[""])
    docs_from_whitespace = splitter.create_documents(texts=["   "])
    
    # ASSERT
    assert docs_from_empty == []
    assert docs_from_whitespace == [] # After stripping, this should also be empty