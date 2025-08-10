# tests/embeddings/test_local_embeddings.py
"""
Unit tests for the `LocalEmbeddings` class.

These tests validate the ability to connect to a local, OpenAI-compatible
server (like LM Studio) and generate embeddings.
"""
import pytest
import socket
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.embeddings import LocalEmbeddings

# --- Test Fixtures and Configuration ---

def is_server_running(host='localhost', port=1234):
    """Checks if a server is running on the specified host and port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

# Pytest marker to skip tests if the local server is not running
requires_local_server = pytest.mark.skipif(
    not is_server_running(),
    reason="Local embedding server (e.g., LM Studio) not running on port 1234."
)

# Expected dimension for nomic-embed-text-v1.5. Adjust if using a different model.
EXPECTED_DIMENSION = 768

# --- Test Functions ---

@requires_local_server
def test_local_embeddings_initialization_succeeds():
    """
    Tests that the LocalEmbeddings class can be initialized without errors.
    """
    # No assert needed, the test passes if no exception is raised
    LocalEmbeddings()

@requires_local_server
def test_local_embed_query_returns_correct_shape():
    """
    Tests that embed_query returns a single list of floats with the
    expected dimension for the Nomic model.
    """
    # ARRANGE
    embeddings = LocalEmbeddings()
    query_text = "This is a test query for the local model."
    
    # ACT
    result = embeddings.embed_query(query_text)
    
    # ASSERT
    assert isinstance(result, list)
    assert len(result) == EXPECTED_DIMENSION
    assert all(isinstance(val, float) for val in result)

@requires_local_server
def test_local_embed_documents_returns_correct_shape():
    """
    Tests that embed_documents returns a list of lists, with each inner
    list matching the expected dimension for the Nomic model.
    """
    # ARRANGE
    embeddings = LocalEmbeddings()
    doc_texts = ["First local test document.", "Second local test document."]
    
    # ACT
    results = embeddings.embed_documents(doc_texts)
    
    # ASSERT
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(vec, list) for vec in results)
    assert all(len(vec) == EXPECTED_DIMENSION for vec in results)

@requires_local_server
def test_local_embeddings_handles_empty_list_gracefully():
    """
    Tests that calling embed_documents with an empty list does not make an
    API call and returns an empty list, as handled by the base class.
    """
    # ARRANGE
    embeddings = LocalEmbeddings()
    
    # ACT
    result = embeddings.embed_documents([])
    
    # ASSERT
    assert result == []