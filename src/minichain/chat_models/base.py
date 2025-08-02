# mini_langchain/chat_models/base.py
"""
Base chat model interface
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
from openai.types.chat import ChatCompletionMessageParam
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage


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
    
    def _messages_to_format(self, messages: List[BaseMessage]) -> List[ChatCompletionMessageParam]:
        """Convert our message objects to proper OpenAI API format"""
        formatted: List[ChatCompletionMessageParam] = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
        
        return formatted