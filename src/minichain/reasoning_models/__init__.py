# src/minichain/reasoning_models/__init__.py
"""
This module provides classes for interacting with OpenAI's official reasoning models
that use the dedicated /v1/responses API endpoint.
"""
from dotenv import load_dotenv
load_dotenv()

from .base import BaseReasoningModel, ReasoningModelConfig
from .openai import OpenAIReasoningModel

__all__ = [
    "BaseReasoningModel",
    "ReasoningModelConfig",
    "OpenAIReasoningModel",
]
# # src/minichain/reasoning_models/__init__.py
# """
# This module provides classes for interacting with reasoning-specific models.

# These models are distinct from standard chat models as they often have different
# API endpoints or require special parameters (like the 'reasoning' object)
# to unlock their advanced capabilities.
# """
# from dotenv import load_dotenv
# load_dotenv()
# from .base import (
#     BaseReasoningModel,
#     ReasoningModelConfig,
#     OpenAIReasoningConfig,
#     OpenRouterReasoningConfig,
#     AzureReasoningConfig,
#     LocalReasoningConfig,
# )
# from .openai_like import OpenAILikeReasoningModel
# from .openai import OpenAIReasoningModel
# from .openrouter import OpenRouterReasoningModel
# from .azure import AzureReasoningModel
# from .local import LocalReasoningModel
# from .run import run_reasoning_chat
# __all__ = [
#     "BaseReasoningModel",
#     "ReasoningModelConfig",
#     "OpenAIReasoningConfig",
#     "OpenRouterReasoningConfig",
#     "AzureReasoningConfig",
#     "LocalReasoningConfig",
#     "OpenAILikeReasoningModel",
#     "OpenAIReasoningModel",
#     "OpenRouterReasoningModel",
#     "AzureReasoningModel",
#     "LocalReasoningModel",
#     "run_reasoning_chat",
# ]