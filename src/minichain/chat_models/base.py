# src/minichain/chat_models/base.py
from abc import ABC, abstractmethod
from typing import Union, List, Iterator, Optional, Any

# Assuming ChatResult is defined in core/types
from ..core.types import BaseMessage, ChatResult
from pydantic import BaseModel, Field

class ChatModelConfig(BaseModel):
    provider: str = Field(description="The name of the chat model provider.")
    temperature: float = 0.7
    max_tokens: Union[int, None] = None
    system_prompt: Optional[str] = None
    model: str # Making model required for all configs is cleaner

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
    # 'model' field is used for the deployment_name
    endpoint: Optional[str] = None
    api_version: str = "2024-02-01"

class BaseChatModel(ABC):
    """Abstract base class for all chat models."""
    def __init__(self, config: ChatModelConfig, **kwargs: Any):
        self.config = config
        self.api_kwargs = kwargs

    @abstractmethod
    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        """Generates a rich, structured response with metadata (blocking)."""
        pass

    @abstractmethod
    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        """Generates a response as a stream of text chunks."""
        pass

    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """A convenience method that returns only the string content."""
        return self.generate(input_data).content

# # src/minichain/chat_models/base.py
# """
# Defines abstract base classes and configuration models for chat models.
# """
# from abc import ABC, abstractmethod
# from typing import Optional, Union, List, Iterator
# from pydantic import BaseModel, Field
# from ..core.types import BaseMessage, ChatResult

# # --- Configuration Models ---

# class ChatModelConfig(BaseModel):
#     """Base Pydantic model for chat model configurations."""
#     provider: str = Field(description="The name of the chat model provider.")
#     temperature: float = 0.7
#     max_tokens: Union[int, None] = None
#     system_prompt: Optional[str] = None
#     force_stream_for_invoke: bool = False

# class LocalChatConfig(ChatModelConfig):
#     """Configuration for a local, OpenAI-compatible chat model."""
#     provider: str = "local"
#     model: str = "local-model/gguf-model"
#     base_url: str = "http://localhost:1234/v1"
#     api_key: str = "not-needed"

# class AzureChatConfig(ChatModelConfig):
#     """Configuration for an Azure OpenAI chat model."""
#     provider: str = "azure"
#     deployment_name: str
#     api_key: Union[str, None] = None     # Can be loaded from env
#     endpoint: Union[str, None] = None     # Can be loaded from env
#     api_version: str = "2024-02-01"

# class OpenRouterConfig(ChatModelConfig):
#     """Configuration for an OpenRouter chat model."""
#     provider: str = "openrouter"
#     model: str
#     api_key: Union[str, None] = None     # Can be loaded from env
#     site_url: Union[str, None] = None    # Optional: For HTTP-Referer header
#     site_name: Union[str, None] = None   # Optional: For X-Title header

# # --- Service Interface ---

# class BaseChatModel(ABC):
#     """Abstract base class for all chat models."""
#     def __init__(self, config: BaseModel): #ChatModelConfig
#         self.config = config

#     @abstractmethod
#     def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
#         """Generates a complete string response (blocking)."""
#         pass

#     @abstractmethod
#     def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
#         """
#         Generates a rich, structured response with metadata (blocking).
        
#         This is the preferred method for accessing token usage, finish reasons, etc.
#         """
#         pass
    
#     @abstractmethod
#     def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
#         """Generates a response as a stream of text chunks."""
#         pass
