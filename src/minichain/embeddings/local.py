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
