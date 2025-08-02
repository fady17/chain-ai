# tests/embeddings/test_azure_embeddings.py
"""
Unit tests for the `AzureOpenAIEmbeddings` class.

These tests are designed to run only if Azure credentials are available
in the environment. They validate the connection and the basic functionality
of generating embeddings for documents and queries.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pytest
from dotenv import load_dotenv
from minichain.embeddings import AzureOpenAIEmbeddings

# Load environment variables from .env file for local testing
load_dotenv()

# --- Test Fixtures and Configuration ---

# Get credentials from environment. The test will be skipped if they're not found.
AZURE_CREDS_AVAILABLE = all(
    os.getenv(var) for var in [
        "AZURE_OPENAI_ENDPOINT_EMBEDDINGS",
        "AZURE_OPENAI_EMBEDDINGS_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
    ]
)
TEST_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "")
# Expected dimension for text-embedding-3-small, a common choice. Adjust if using a different model.
EXPECTED_DIMENSION = 1536

# Pytest marker to skip tests if credentials are not available
requires_azure_creds = pytest.mark.skipif(
    not AZURE_CREDS_AVAILABLE,
    reason="Azure embedding credentials not found in environment variables."
)

# --- Test Functions ---

@requires_azure_creds
def test_azure_embeddings_initialization_succeeds():
    """
    Tests that the AzureOpenAIEmbeddings class can be initialized without errors
    when credentials are provided.
    """
    # ACT
    # No assert needed, the test passes if no exception is raised
    AzureOpenAIEmbeddings(deployment_name=TEST_DEPLOYMENT_NAME)

def test_azure_embeddings_initialization_fails_without_creds():
    """
    Tests that a ValueError is raised if the class is initialized
    without the necessary environment variables being set.
    """
    # ARRANGE
    # Temporarily unset environment variables
    original_key = os.environ.pop("AZURE_OPENAI_EMBEDDINGS_API_KEY", None)
    
    # ACT & ASSERT
    with pytest.raises(ValueError, match="environment variables must be set"):
        AzureOpenAIEmbeddings(deployment_name="test")
        
    # CLEANUP
    if original_key:
        os.environ["AZURE_OPENAI_EMBEDDINGS_API_KEY"] = original_key

@requires_azure_creds
def test_azure_embed_query_returns_correct_shape():
    """
    Tests that embed_query returns a single list of floats with the
    expected dimension.
    """
    # ARRANGE
    embeddings = AzureOpenAIEmbeddings(deployment_name=TEST_DEPLOYMENT_NAME)
    query_text = "This is a test query."
    
    # ACT
    result = embeddings.embed_query(query_text)
    
    # ASSERT
    assert isinstance(result, list)
    assert len(result) == EXPECTED_DIMENSION
    assert all(isinstance(val, float) for val in result)

@requires_azure_creds
def test_azure_embed_documents_returns_correct_shape():
    """
    Tests that embed_documents returns a list of lists, with each inner
    list matching the expected dimension.
    """
    # ARRANGE
    embeddings = AzureOpenAIEmbeddings(deployment_name=TEST_DEPLOYMENT_NAME)
    doc_texts = ["First test document.", "Second test document."]
    
    # ACT
    results = embeddings.embed_documents(doc_texts)
    
    # ASSERT
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(vec, list) for vec in results)
    assert all(len(vec) == EXPECTED_DIMENSION for vec in results)