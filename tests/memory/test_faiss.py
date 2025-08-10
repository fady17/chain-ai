# tests/memory/test_faiss.py
"""
Unit and integration tests for the hardware-aware `FAISSVectorStore`.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pytest
import numpy as np

# --- Configuration and Dependency Checks ---

# First, check if the faiss library can be imported at all.
try:
    import faiss
    FAISS_INSTALLED = True
except ImportError:
    faiss = None  # Ensure the name 'faiss' exists even if import fails
    FAISS_INSTALLED = False

from chain.core.types import Document
from chain.vectors import FAISSVectorStore
from chain.embeddings.base import BaseEmbeddings

# --- Mocking Dependencies ---

class MockEmbeddings(BaseEmbeddings):
    """A mock embedding class for deterministic testing."""
    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        seed = hash(text) % (2**32)
        rng = np.random.RandomState(seed)
        return rng.rand(self.dimension).astype(np.float32).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


# --- Test Fixtures and Parametrization ---

# Second, perform the check for GPU capabilities, but only if FAISS was installed.
if FAISS_INSTALLED:
    try:
        # This check is safe now because we know 'faiss' is not None.
        FAISS_GPU_AVAILABLE = hasattr(faiss, 'StandardGpuResources') and faiss.get_num_gpus() > 0 # type: ignore
    except Exception:
        # Handle any other potential errors during GPU detection
        FAISS_GPU_AVAILABLE = False
else:
    FAISS_GPU_AVAILABLE = False

# A pytest marker to skip tests if FAISS is not installed at all.
requires_faiss = pytest.mark.skipif(not FAISS_INSTALLED, reason="FAISS library is not installed.")

# A list of devices to test.
devices_to_test = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not FAISS_GPU_AVAILABLE, reason="faiss-gpu not installed or no CUDA devices found")
    )
]

# --- Test Functions ---

@requires_faiss
@pytest.mark.parametrize("device", devices_to_test)
def test_add_and_search_on_device(device):
    """
    Tests the fundamental add/search workflow on a specified device (CPU or GPU).
    """
    embeddings = MockEmbeddings()
    vector_store = FAISSVectorStore(embeddings=embeddings, device=device)
    docs = [Document(page_content="The sun is bright.")]
    
    vector_store.add_documents(docs)
    results = vector_store.similarity_search(query="The sun is bright.", k=1)
    
    assert len(results) == 1
    assert results[0][0].page_content == "The sun is bright."
    assert results[0][1] == 0.0

@requires_faiss
@pytest.mark.parametrize("device", devices_to_test)
def test_persistence_save_and_load(tmp_path, device):
    """
    Tests the full save/load lifecycle on a specified device.
    """
    folder_path = str(tmp_path)
    embeddings = MockEmbeddings()
    
    original_store = FAISSVectorStore.from_documents(
        documents=[Document(page_content="Test persistence.")],
        embeddings=embeddings,
        device=device
    )
    original_store.save_local(folder_path)
    
    assert os.path.exists(os.path.join(folder_path, "index.faiss"))
    
    loaded_store = FAISSVectorStore.load_local(folder_path, embeddings, device=device)
    
    results = loaded_store.similarity_search(query="Test persistence.", k=1)
    assert len(results) == 1
    assert results[0][0].page_content == "Test persistence."

@requires_faiss
def test_search_on_empty_store_returns_empty_list():
    """
    Tests that searching an empty store returns an empty list gracefully.
    """
    vector_store = FAISSVectorStore(embeddings=MockEmbeddings())
    results = vector_store.similarity_search(query="anything")
    assert results == []