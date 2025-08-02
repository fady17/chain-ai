import requests
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests
import json
import os

from minichain.core.types import AIMessage, BaseMessage, HumanMessage, SystemMessage


class BaseChatModel(ABC):
    """Abstract base class for all chat models"""
    
    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
    
    @abstractmethod
    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """Generate response from input"""
        pass
    
    def _messages_to_format(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert our message objects to API format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
        return formatted


class OpenAIChatModel(BaseChatModel):
    """OpenAI GPT chat model implementation"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0, **kwargs):
        super().__init__(model_name, temperature, **kwargs)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """Generate response using OpenAI API"""
        
        # Handle string input
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            # Handle message list
            messages = self._messages_to_format(input_data)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


class OpenRouterChatModel(BaseChatModel):
    """OpenRouter chat model implementation (alternative provider)"""
    
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", temperature: float = 0.0, **kwargs):
        super().__init__(model_name, temperature, **kwargs)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """Generate response using OpenRouter API"""
        
        # Handle string input
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            # Handle message list
            messages = self._messages_to_format(input_data)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # Required for OpenRouter
            "X-Title": "Mini LangChain"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]

