# tests/text_splitters/test_character_splitter.py
"""
Unit tests for the `RecursiveCharacterTextSplitter`.

These tests validate that the splitter correctly chunks text based on character
counts and its recursive separator logic.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.core.types import Document
from minichain.text_splitters import RecursiveCharacterTextSplitter

def test_recursive_splitter_uses_primary_separator():
    """
    Tests that the splitter prioritizes the first separator in the list (\n\n)
    when it exists in the text.
    """
    # ARRANGE
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=5,
        separators=["\n\n", " ", ""]
    )
    text = "First paragraph.\n\nSecond paragraph."

    # ACT
    chunks = splitter.split_text(text)
    
    # ASSERT
    assert len(chunks) == 2
    assert chunks[0] == "First paragraph."
    assert chunks[1] == "Second paragraph."

def test_recursive_splitter_falls_back_to_secondary_separator():
    """
    Tests that the splitter uses a lower-priority separator (space) when the
    primary separator (\n\n) is not present.
    """
    # ARRANGE
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=15,
        chunk_overlap=5,
        separators=["\n\n", " ", ""]
    )
    text = "A single sentence to be split by spaces."

    # ACT
    chunks = splitter.split_text(text)
    
    # ASSERT
    assert len(chunks) > 1
    # The first chunk should be "A single" or similar
    assert "A single" in chunks[0]

def test_recursive_splitter_preserves_metadata_when_creating_documents():
    """
    Tests that the original metadata is correctly passed to all created chunks.
    """
    # ARRANGE
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    texts = ["A short test text."]
    metadatas = [{"source": "test_source.md"}]

    # ACT
    docs = splitter.create_documents(texts=texts, metadatas=metadatas)
    
    # ASSERT
    assert docs[0].metadata["source"] == "test_source.md"
    assert "total_chunks" in docs[0].metadata

def test_recursive_splitter_reconstructs_original_content():
    """
    Tests the integrity of the splitter by ensuring that the combined content of
    the chunks is equivalent to the original text (ignoring overlap details).
    This confirms no data is lost.
    """
    # ARRANGE
    splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=5)
    original_text = "This is a slightly longer sentence designed to test the integrity of the recursive text splitter."
    
    # ACT
    docs = splitter.create_documents(texts=[original_text])
    reconstructed_text = "".join(doc.page_content.replace(" ", "") for doc in docs) # Normalize by removing spaces
    
    # ASSERT
    assert original_text.replace(" ", "") in reconstructed_text