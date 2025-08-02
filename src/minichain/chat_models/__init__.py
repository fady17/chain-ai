# src/minichain/chat_models/__init__.py
"""
This module provides classes for interacting with chat-based language models,
supporting both cloud-based and local providers.

The key components exposed are:
    - BaseChatModel: The abstract interface for all chat models.
    - AzureOpenAIChatModel: For generating chat completions using Azure OpenAI.
    - LocalChatModel: For generating chat completions using a local, OpenAI-compatible
      server like LM Studio.
"""
from .base import BaseChatModel
from .azure import AzureOpenAIChatModel
from .local import LocalChatModel

__all__ = [
    "BaseChatModel",
    "AzureOpenAIChatModel",
    "LocalChatModel",
]