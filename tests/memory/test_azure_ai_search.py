# tests/memory/test_azure_ai_search.py
"""
Integration tests for the `AzureAISearchVectorStore`.

These tests validate the full lifecycle of interacting with the Azure AI Search
service: creating an index, adding documents, and performing a similarity search.
Tests are skipped if Azure credentials are not available.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import time
import pytest
from dotenv import load_dotenv
from chain.core.types import Document
from chain.vectors import AzureAISearchVectorStore
from chain.embeddings.base import BaseEmbeddings
import numpy as np

# Load environment variables for local testing
load_dotenv()

# --- Mocking and Configuration ---

class MockEmbeddings(BaseEmbeddings):
    """A mock embedding class for deterministic testing."""
    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return np.random.rand(self.dimension).astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [np.random.rand(self.dimension).astype(np.float32).tolist() for _ in texts]

AZURE_CREDS_AVAILABLE = all(
    os.getenv(var) for var in ["AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_ADMIN_KEY"]
)

requires_azure_creds = pytest.mark.skipif(
    not AZURE_CREDS_AVAILABLE,
    reason="Azure AI Search credentials not found in environment variables."
)

@pytest.fixture(scope="module")
def temp_index_name():
    """Pytest fixture to create a unique, temporary index name for the test run."""
    return f"chain-test-index-{int(time.time())}"

# --- Test Functions ---

@requires_azure_creds
def test_azure_search_lifecycle(temp_index_name):
    """
    Tests the full end-to-end lifecycle: index creation, document addition,
    and similarity search. Also handles cleanup.
    """
    # ARRANGE
    embeddings = MockEmbeddings()
    try:
        # ACT (Initialization and Index Creation)
        vector_store = AzureAISearchVectorStore(
            embeddings=embeddings,
            index_name=temp_index_name
        )
        
        # ACT (Add Documents)
        docs = [Document(page_content="The sun is a star.", metadata={"source": "space"})]
        vector_store.add_documents(docs)
        
        # Give Azure a moment to index the documents
        time.sleep(5)
        
        # ACT (Similarity Search)
        results = vector_store.similarity_search(query="What is the sun?", k=1)
        
        # ASSERT
        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert "star" in results[0].page_content
        assert results[0].metadata["source"] == "space"
        
    finally:
        # CLEANUP: Ensure the temporary index is deleted after the test
        # This is crucial to keep the Azure service clean and avoid test pollution.
        try:
            cleanup_client = AzureAISearchVectorStore(embeddings, temp_index_name).index_client
            cleanup_client.delete_index(temp_index_name)
        except Exception as e:
            print(f"Cleanup failed for index '{temp_index_name}': {e}")

def test_azure_search_initialization_fails_without_creds():
    """
    Tests that a ValueError is raised if the class is initialized
    without the necessary environment variables.
    """
    original_key = os.environ.pop("AZURE_AI_SEARCH_ADMIN_KEY", None)
    
    with pytest.raises(ValueError, match="must be set in environment"):
        AzureAISearchVectorStore(embeddings=MockEmbeddings(), index_name="test")
        
    if original_key:
        os.environ["AZURE_AI_SEARCH_ADMIN_KEY"] = original_key