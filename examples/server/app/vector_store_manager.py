# File: app/vector_store_manager.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from chain.vectors import FAISSVectorStore
from chain.embeddings import LocalEmbeddings

class VectorStoreManager:
    _instance = None
    _vector_store = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreManager, cls).__new__(cls)
            # Initialize the vector store once
            embeddings = LocalEmbeddings()
            cls._vector_store = FAISSVectorStore(embeddings=embeddings)
        return cls._instance

    def get_vector_store(self):
        return self._vector_store

# Create a singleton instance
vector_store_manager = VectorStoreManager()