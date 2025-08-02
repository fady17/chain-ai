# mini_langchain/chat_models/implementations.py
"""
Azure OpenAI chat model implementation (focused on Azure only)
"""

import os
from typing import Union, List
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseChatModel
from ..core.types import BaseMessage


class AzureOpenAIChatModel(BaseChatModel):
    """Azure OpenAI chat model implementation"""
    
    def __init__(self, 
                 deployment_name: str,
                 model_name: str = "gpt-4",
                 temperature: float = 0.0, 
                 max_tokens: int = None, # type: ignore
                 **kwargs):
        super().__init__(model_name, temperature, **kwargs)
        
        # Required Azure OpenAI parameters
        self.deployment_name = deployment_name
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY") 
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        self.max_tokens = max_tokens
        
        if not all([self.azure_endpoint, self.api_key]):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables required"
            )
        
        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint, # type: ignore
            api_key=self.api_key,
        )
    
    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """Generate response using Azure OpenAI API"""
        
        # Handle string input - create proper message format
        if isinstance(input_data, str):
            messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": input_data}
            ]
        else:
            # Handle message list - convert using our base method
            messages = self._messages_to_format(input_data)
        
        # Prepare parameters for API call
        completion_params = {
            "model": self.deployment_name,  # Azure uses deployment name
            "messages": messages,
            "temperature": self.temperature,
        }
        
        # Add max_tokens if specified
        if self.max_tokens:
            completion_params["max_tokens"] = self.max_tokens
            
        # Add any additional kwargs
        completion_params.update(self.kwargs)
        
        response = self.client.chat.completions.create(**completion_params)
        
        return response.choices[0].message.content