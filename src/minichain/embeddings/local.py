# src/minichain/embeddings/local.py
"""
Embeddings implementation for local models served via an
OpenAI-compatible API (e.g., LM Studio, Ollama).
"""

from typing import List
from openai import OpenAI

from .base import BaseEmbeddings

class LocalEmbeddings(BaseEmbeddings):
    """
    Connects to a local embedding model served via an OpenAI-compatible API endpoint.
    Ideal for use with LM Studio's embedding models like Nomic.
    """
    
    def __init__(self, 
                 model_name: str = "nomic-ai/nomic-embed-text-v1.5",
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "not-needed", # API key is often not required
                 **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Initialize the OpenAI client to point to the local server
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the local model."""
        
        # The 'model' parameter tells LM Studio which model to use if multiple are loaded.
        # It should match the model name/alias shown in the LM Studio UI.
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the local model."""
        
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding