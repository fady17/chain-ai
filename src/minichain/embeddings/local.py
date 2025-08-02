# src/minichain/embeddings/local.py
"""
Implementation for local embedding models served via an OpenAI-compatible API.
"""
from openai import OpenAI
from .openai import OpenAILikeEmbeddings # Correctly inherit from our robust base class

class LocalEmbeddings(OpenAILikeEmbeddings):
    """
    Connects to a local embedding model (e.g., from LM Studio, Ollama)
    that provides an OpenAI-compatible API endpoint.

    This class inherits its core embedding logic from `OpenAILikeEmbeddings`
    and is only responsible for configuring the `OpenAI` client to point
    to a local server.
    """
    def __init__(self, 
                 model_name: str = "nomic-ai/nomic-embed-text-v1.5",
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "not-needed"):
        """
        Initializes the LocalEmbeddings client.

        Args:
            model_name (str): The model identifier expected by the local server.
            base_url (str): The base URL of the local server API.
            api_key (str): The API key (often unused for local servers).
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # This attribute is used by the OpenAILikeEmbeddings base class.
        self.model_name = model_name
# # src/minichain/embeddings/local.py
# """
# Embeddings implementation for local models served via an
# OpenAI-compatible API (e.g., LM Studio, Ollama).
# """

# from typing import List
# from openai import OpenAI

# from .base import BaseEmbeddings

# class LocalEmbeddings(BaseEmbeddings):
#     """
#     Connects to a local embedding model served via an OpenAI-compatible API endpoint.
#     Ideal for use with LM Studio's embedding models like Nomic.
#     """
    
#     def __init__(self, 
#                  model_name: str = "nomic-ai/nomic-embed-text-v1.5",
#                  base_url: str = "http://localhost:1234/v1",
#                  api_key: str = "not-needed", # API key is often not required
#                  **kwargs):
#         super().__init__(model_name, **kwargs)
        
#         # Initialize the OpenAI client to point to the local server
#         self.client = OpenAI(base_url=base_url, api_key=api_key)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Embed a list of documents using the local model."""
        
#         # The 'model' parameter tells LM Studio which model to use if multiple are loaded.
#         # It should match the model name/alias shown in the LM Studio UI.
#         response = self.client.embeddings.create(
#             input=texts,
#             model=self.model_name
#         )
#         return [item.embedding for item in response.data]

#     def embed_query(self, text: str) -> List[float]:
#         """Embed a single query using the local model."""
        
#         response = self.client.embeddings.create(
#             input=[text],
#             model=self.model_name
#         )
#         return response.data[0].embedding