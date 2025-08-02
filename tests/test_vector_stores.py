# tests/test_vector_stores.py
"""
Test suite for vector store implementations.
"""
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.core.types import Document
from minichain.embeddings.base import BaseEmbeddings
from minichain.vector_stores.faiss import FAISSVectorStore

# --- Mock Embeddings for Predictable Testing ---
class MockEmbeddings(BaseEmbeddings):
    """A mock embedding class for deterministic testing."""
    def __init__(self, dimension: int = 4):
        super().__init__(model_name="mock")
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        # Create a predictable vector from the query text's hash
        # This is just to make it deterministic but different for different queries
        seed = hash(text) % (2**32 - 1)
        rng = np.random.RandomState(seed)
        vec = rng.rand(self.dimension).astype(np.float32)
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Return a list of predictable vectors for each document
        return [self.embed_query(text) for text in texts]

# --- Test Data ---
SAMPLE_DOCUMENTS = [
    Document(page_content="The capital of France is Paris.", metadata={"source": "doc1"}),
    Document(page_content="The primary colors are red, yellow, and blue.", metadata={"source": "doc2"}),
    Document(page_content="Apples are a type of fruit that grow on trees.", metadata={"source": "doc3"}),
]

# --- Test Function ---
def test_faiss_vector_store():
    """
    Performs a comprehensive test of the FAISSVectorStore.
    """
    print("\n" + "="*60)
    print(" 🧪 TESTING FAISS VECTOR STORE ".center(60, "="))
    print("="*60)

    # --- Step 1: Initialization ---
    print("\nSTEP 1: INITIALIZATION")
    mock_embeddings = MockEmbeddings(dimension=8) # Use a small dimension for testing
    print(f"✅ Using MockEmbeddings with dimension: {mock_embeddings.dimension}")
    
    # --- Step 2: Create store using `from_documents` ---
    print("\nSTEP 2: CREATING STORE WITH `from_documents`")
    vector_store = FAISSVectorStore.from_documents(
        documents=SAMPLE_DOCUMENTS,
        embeddings=mock_embeddings
    )
    assert vector_store.index is not None, "Index should be initialized" # type: ignore
    assert vector_store.index.ntotal == len(SAMPLE_DOCUMENTS), "Index should have correct number of vectors" # type: ignore
    print("✅ Store created successfully.")
    print(f"✅ Docstore contains {len(vector_store._docstore)} documents.") # type: ignore

    # --- Step 3: Similarity Search ---
    print("\nSTEP 3: TESTING SIMILARITY SEARCH")
    # This query will have a vector identical to "The capital of France is Paris."
    query = "The capital of France is Paris."
    print(f"Querying for: '{query}' with k=1")
    
    results = vector_store.similarity_search(query, k=1)
    
    print(f"Returned {len(results)} result(s).")
    assert len(results) == 1, "Should return exactly one result for k=1"
    
    top_result = results[0]
    print(f"Top result content: '{top_result.page_content}'")
    print(f"Top result metadata: {top_result.metadata}")
    assert top_result.page_content == SAMPLE_DOCUMENTS[0].page_content
    assert top_result.metadata["source"] == "doc1"
    print("✅ Top result is correct.")

    # --- Step 4: Test `add_documents` to an existing store ---
    print("\nSTEP 4: ADDING MORE DOCUMENTS TO EXISTING STORE")
    new_doc = Document(page_content="Python is a popular programming language.", metadata={"source": "doc4"})
    vector_store.add_documents([new_doc])
    
    assert vector_store.index.ntotal == len(SAMPLE_DOCUMENTS) + 1 # type: ignore
    print(f"✅ Index size is now correct: {vector_store.index.ntotal}") # type: ignore
    
    # Search for the newly added document
    new_query = "Python is a popular programming language."
    print(f"Querying for new document: '{new_query}'")
    new_results = vector_store.similarity_search(new_query, k=1)
    
    assert len(new_results) == 1
    assert new_results[0].page_content == new_doc.page_content
    print("✅ Successfully found the newly added document.")

    # --- Step 5: Test `k` parameter ---
    print("\nSTEP 5: TESTING THE `k` PARAMETER")
    print("Querying for 'colors' with k=3")
    k_results = vector_store.similarity_search("colors", k=3)
    
    print(f"Returned {len(k_results)} results.")
    assert len(k_results) == 3
    # The 'colors' doc should be first, but we just check the count here
    print("✅ `k` parameter respected.")

    print("\n" + "="*60)
    print(" 🎉 FAISS VECTOR STORE TESTS PASSED ".center(60, "="))
    print("="*60)
    return True

if __name__ == "__main__":
    test_faiss_vector_store()