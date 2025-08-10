# src/chain/chat_models/base.py
from abc import ABC, abstractmethod
from typing import Union, List, Iterator, Optional, Any
from pydantic import BaseModel, Field
from ..core.types import BaseMessage, ChatResult

class ChatModelConfig(BaseModel):
    provider: str = Field(description="The name of the chat model provider.")
    temperature: float = 0.7
    max_tokens: Union[int, None] = None
    system_prompt: Optional[str] = None
    model: str

class LocalChatConfig(ChatModelConfig):
    provider: str = "local"
    model: str = "local-model/gguf-model"
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "not-needed"

class OpenRouterConfig(ChatModelConfig):
    provider: str = "openrouter"
    site_url: Optional[str] = None
    site_name: Optional[str] = None

class AzureChatConfig(ChatModelConfig):
    provider: str = "azure"
    endpoint: Optional[str] = None
    api_version: str = "2024-02-01"

class BaseChatModel(ABC):
    """Abstract base class for all chat models."""
    def __init__(self, config: ChatModelConfig, **kwargs: Any):
        self.config = config
        self.api_kwargs = kwargs

    @abstractmethod
    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        pass

    @abstractmethod
    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        pass

    def invoke(self, input_data: Any, **kwargs: Any) -> Any:
        if hasattr(input_data, 'to_string'):
            input_data = input_data.to_string()
        result_obj = self.generate(input_data)
        return result_obj.content
