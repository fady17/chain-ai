# src/minichain/chat_models/__init__.py
"""
This module provides classes for interacting with chat-based language models.
"""
from dotenv import load_dotenv
load_dotenv()
from .base import BaseChatModel, LocalChatConfig, AzureChatConfig, OpenRouterConfig 
from .azure import AzureOpenAIChatModel
from .openrouter import OpenRouterChatModel
from .local import LocalChatModel
from .run import run_chat

__all__ = [
    "BaseChatModel",
    "LocalChatConfig",
    "AzureChatConfig",
    "OpenRouterConfig",
    "AzureOpenAIChatModel",
    "LocalChatModel",
    "OpenRouterChatModel",
    "run_chat",
]