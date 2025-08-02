# tests/test_azure_ai_search.py
"""
Test suite for the Azure AI Search vector store.
"""
import os
import sys
import time
import numpy as np
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.core.types import Document
from minichain.embeddings.base import BaseEmbeddings
from minichain.vector_stores.azure_ai_search import AzureAISearchVectorStore

# Load environment variables
load_dotenv()

# --- Mock Embeddings for Predictable Testing ---
class MockEmbeddings(BaseEmbeddings):
    """A mock embedding class for deterministic testing."""
    def __init__(self, dimension: int = 8):
        super().__init__(model_name="mock")
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        seed = hash(text) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        vec = rng.rand(self.dimension).astype(np.float32)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

# --- Test Data ---
SAMPLE_DOCUMENTS = [
    Document(page_content="The lead architect of Mini-Chain is Dr. Evelyn Reed.", metadata={"source": "doc1", "chunk_index": 0}),
    Document(page_content="The primary colors are red, yellow, and blue.", metadata={"source": "doc2", "chunk_index": 0}),
    Document(page_content="Amin is a legendary AI engineer on the review team.", metadata={"source": "doc3", "chunk_index": 0}),
]

# --- Test Function ---
def test_azure_ai_search_vector_store():
    """
    Performs a comprehensive test of the AzureAISearchVectorStore.
    """
    print("\n" + "="*60)
    print(" 🧪 TESTING AZURE AI SEARCH VECTOR STORE ".center(60, "="))
    print("="*60)

    # --- Step 1: Check for Credentials ---
    print("\nSTEP 1: CHECKING CREDENTIALS")
    if not os.getenv("AZURE_AI_SEARCH_ENDPOINT") or not os.getenv("AZURE_AI_SEARCH_ADMIN_KEY"):
        print("⚠️ Skipping test: Azure AI Search credentials not found in .env file.")
        return True
    
    print("✅ Credentials found.")
    
    test_index_name = f"minichain-test-index-{int(time.time())}"
    print(f"Using temporary index name: {test_index_name}")

    # --- Step 2: Initialization & Index Creation ---
    print("\nSTEP 2: INITIALIZATION & INDEX CREATION")
    mock_embeddings = MockEmbeddings(dimension=8)
    vector_store = AzureAISearchVectorStore(
        embeddings=mock_embeddings,
        index_name=test_index_name
    )
    print("✅ Vector store initialized.")

    # --- Step 3: Add Documents ---
    print("\nSTEP 3: ADDING DOCUMENTS")
    vector_store.add_documents(SAMPLE_DOCUMENTS)
    
    print("Waiting 5 seconds for indexing to complete...")
    time.sleep(5) 
    
    # --- Step 4: Similarity Search ---
    print("\nSTEP 4: TESTING SIMILARITY SEARCH")

    # --- FIX: Query for the EXACT text of a document ---
    # Because our MockEmbeddings are not semantic, we test the end-to-end
    # mechanism by searching for a vector we know must be in the index.
    # The vector for this query will be identical to the vector for SAMPLE_DOCUMENTS[2].
    query = "Amin is a legendary AI engineer on the review team."
    print(f"Querying for an exact match: '{query}' with k=1")
    
    results = vector_store.similarity_search(query, k=1)
    
    print(f"Returned {len(results)} result(s).")
    assert len(results) >= 1, "Should return at least one result."
    
    top_result = results[0]
    print(f"Top result content: '{top_result.page_content}'")
    print(f"Top result metadata: {top_result.metadata}")

    # --- FIX: The assertion is now a direct equality check ---
    assert top_result.page_content == query
    print("✅ Top result is a perfect match, as expected.")

    # --- Step 5: Clean up the index ---
    print("\nSTEP 5: CLEANING UP")
    try:
        vector_store.index_client.delete_index(test_index_name)
        print(f"✅ Successfully deleted index '{test_index_name}'.")
    except Exception as e:
        print(f"⚠️ Failed to delete index '{test_index_name}': {e}")
        print("Please delete it manually from the Azure Portal.")

    print("\n" + "="*60)
    print(" 🎉 AZURE AI SEARCH TESTS PASSED ".center(60, "="))
    print("="*60)
    return True

if __name__ == "__main__":
    test_azure_ai_search_vector_store()