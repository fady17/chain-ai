# src/minichain/chat_models/base.py
"""
Defines abstract base classes and configuration models for chat models.
"""
from abc import ABC, abstractmethod
from typing import Union, List, Iterator
from pydantic import BaseModel, Field
from ..core.types import BaseMessage

# --- Configuration Models ---

class ChatModelConfig(BaseModel):
    """Base Pydantic model for chat model configurations."""
    provider: str = Field(description="The name of the chat model provider.")
    temperature: float = 0.7
    max_tokens: int | None = None

class LocalChatConfig(ChatModelConfig):
    """Configuration for a local, OpenAI-compatible chat model."""
    provider: str = "local"
    model_name: str = "local-model/gguf-model"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"

class AzureChatConfig(ChatModelConfig):
    """Configuration for an Azure OpenAI chat model."""
    provider: str = "azure"
    deployment_name: str
    api_key: str | None = None      # Can be loaded from env
    endpoint: str | None = None     # Can be loaded from env
    api_version: str = "2024-02-01"

# --- Service Interface ---

class BaseChatModel(ABC):
    """Abstract base class for all chat models."""
    def __init__(self, config: ChatModelConfig):
        self.config = config

    @abstractmethod
    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """Generates a complete string response (blocking)."""
        pass
    
    @abstractmethod
    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        """Generates a response as a stream of text chunks."""
        pass
