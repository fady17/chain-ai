# src/minichain/chat_models/local.py
"""
Chat model implementation for local models served via an
OpenAI-compatible API (e.g., LM Studio, Ollama).
"""

from typing import Union, List
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseChatModel
from ..core.types import BaseMessage

class LocalChatModel(BaseChatModel):
    """
    Connects to a local LLM served via an OpenAI-compatible API endpoint.
    Ideal for use with LM Studio, Ollama, etc.
    """
    
    def __init__(self, 
                 model_name: str = "local-model/qwen2-7b-instruct",
                 base_url: str = "http://localhost:1234/v1",
                 api_key: str = "not-needed", # API key is often not required for local servers
                 temperature: float = 0.7, 
                 **kwargs):
        super().__init__(model_name, temperature, **kwargs)
        
        # Initialize the OpenAI client to point to the local server
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """Generate a response from the local model."""
        
        if isinstance(input_data, str):
            messages: List[ChatCompletionMessageParam] = [
                {"role": "user", "content": input_data}
            ]
        else:
            messages = self._messages_to_format(input_data)
        
        # In LM Studio, the 'model' parameter is often ignored as you
        # select the model in the UI. We pass it for compatibility.
        completion_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            **self.kwargs
        }
        
        response = self.client.chat.completions.create(**completion_params)
        
        return response.choices[0].message.content