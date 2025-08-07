# src/minichain/reasoning_models/base.py
"""
Defines the interface for models that use OpenAI's dedicated /v1/responses API.
"""
from abc import ABC, abstractmethod
from typing import Union, List, Iterator, Optional
from pydantic import BaseModel, Field

from ..core.types import BaseMessage, ChatResult

class ReasoningModelConfig(BaseModel):
    """Universal configuration for any model using a 'reasoning' API."""
    model: str
    reasoning_effort: str = Field("medium", pattern="^(low|medium|high)$")
    max_output_tokens: Optional[int] = None


class BaseReasoningModel(ABC):
    """Abstract base class for models using the Responses API."""
    def __init__(self, config: ReasoningModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        """Generates a rich, structured response using the Responses API."""
        pass

    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """A convenience method that returns only the string content."""
        return self.generate(input_data).content
# # src/minichain/reasoning_models/base.py
# """
# Defines abstract base classes and configuration models for reasoning-specific models.
# """

# from abc import ABC, abstractmethod
# from typing import Union, List, Optional, Iterator
# from pydantic import BaseModel, Field

# from ..core.types import BaseMessage, ChatResult

# class ReasoningModelConfig(BaseModel):
#     """Base Pydantic model for all reasoning model configurations."""
#     provider: str
#     model: str
#     reasoning_effort: str = Field(
#         "medium",  # Correctly sets a default value
#         description="Specifies the reasoning effort: 'low', 'medium', or 'high'.",
#         pattern="^(low|medium|high)$"
#     )
#     max_output_tokens: Optional[int] = None

# class OpenAIReasoningConfig(ReasoningModelConfig):
#     """Configuration for an official OpenAI Reasoning Model."""
#     provider: str = "openai_reasoning"
#     # No API key field needed; handled internally.

# class OpenRouterReasoningConfig(ReasoningModelConfig):
#     """Configuration for a Reasoning Model accessed via OpenRouter."""
#     provider: str = "openrouter_reasoning"
#     site_url: Optional[str] = None
#     site_name: Optional[str] = None

# class AzureReasoningConfig(ReasoningModelConfig):
#     """(Future-proofing) Configuration for an Azure Reasoning Model."""
#     provider: str = "azure_reasoning"
#     deployment_name: str
#     endpoint: Optional[str] = None
#     api_version: str = "2024-05-01-preview"

# class LocalReasoningConfig(ReasoningModelConfig):
#     """Configuration for a local, OpenAI-compatible reasoning model."""
#     provider: str = "local_reasoning"
#     model: str = "local-reasoning-model/gguf"
#     base_url: str = "http://localhost:1234/v1"
#     api_key: str = "not-needed" # The only config that needs this field.

# class BaseReasoningModel(ABC):
#     """Abstract base class for all reasoning models."""
#     def __init__(self, config: ReasoningModelConfig):
#         self.config = config

#     @abstractmethod
#     def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
#         """Generates a rich, structured response with metadata (blocking)."""
#         pass

#     @abstractmethod
#     def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
#         """Generates a response as a stream of text chunks."""
#         pass

#     def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
#         """A convenience method that returns only the string content."""
#         return self.generate(input_data).content