# src/minichain/chat_models/local.py
"""
Implementation for local chat models served via an OpenAI-compatible API.
"""
from typing import Any, Dict
from openai import OpenAI
from .openai import OpenAILikeChatModel # Inherit from our new base class

class LocalChatModel(OpenAILikeChatModel):
    """
    Connects to a local chat model (e.g., from LM Studio, Ollama)
    that provides an OpenAI-compatible API endpoint.
    """
    def __init__(self, 
                 model_name: str = "local-model/gguf-model",
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "not-needed",
                 temperature: float = 0.7, 
                 max_tokens: int | None = None,
                 **kwargs: Any):
        """
        Initializes the LocalChatModel client.

        Args:
            model_name (str): The model identifier expected by the local server.
            base_url (str): The base URL of the local server API.
            api_key (str): The API key (often unused for local servers).
            temperature (float): The sampling temperature to use.
            max_tokens (int | None): The maximum number of tokens to generate.
            **kwargs: Additional parameters to pass to the API.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        # These attributes are used by the base class for the API call.
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
