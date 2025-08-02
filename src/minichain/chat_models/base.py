# src/minichain/chat_models/base.py
"""
Defines the abstract base class for all chat models in the Mini-Chain framework.
This ensures a consistent `.invoke()` interface across all model providers.
"""
from abc import ABC, abstractmethod
from typing import Union, List
from ..core.types import BaseMessage

class BaseChatModel(ABC):
    """Abstract base class for all chat models."""
    
    @abstractmethod
    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """
        Generates a string response from the chat model.

        Args:
            input_data: Either a single string (for a simple prompt) or a
                list of `BaseMessage` objects (for a conversation).

        Returns:
            The string content of the AI's response.
        """
        pass
