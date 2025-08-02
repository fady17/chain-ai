# src/minichain/chat_models/azure.py
"""
Implementation for Azure OpenAI chat models.
"""
import os
from typing import Any, Dict
from openai import AzureOpenAI
from .openai import OpenAILikeChatModel # Inherit from our new base class

class AzureOpenAIChatModel(OpenAILikeChatModel):
    """
    Connects to an Azure OpenAI deployment to generate chat completions.
    """
    def __init__(self, 
                 deployment_name: str,
                 temperature: float = 0.7, 
                 max_tokens: int | None = None,
                 **kwargs: Any):
        """
        Initializes the AzureOpenAIChatModel client.

        Args:
            deployment_name (str): The name of your deployed chat model in Azure.
            temperature (float): The sampling temperature to use.
            max_tokens (int | None): The maximum number of tokens to generate.
            **kwargs: Additional parameters to pass to the API.
        """
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY") 
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not azure_endpoint or not api_key:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables must be set."
            )
        
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
        )
        # These attributes are used by the base class for the API call.
        self.model_name = deployment_name # For Azure, this is the deployment name.
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
