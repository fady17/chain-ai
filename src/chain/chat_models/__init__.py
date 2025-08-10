# src/chain/chat_models/__init__.py
from dotenv import load_dotenv
load_dotenv()

from .base import BaseChatModel, ChatModelConfig, LocalChatConfig, AzureChatConfig, OpenRouterConfig
from .azure import AzureOpenAIChatModel
from .local import LocalChatModel
from .openrouter import OpenRouterChatModel
from .run import run_chat

__all__ = [
    "BaseChatModel",
    "ChatModelConfig",
    "LocalChatConfig",
    "AzureChatConfig",
    "OpenRouterConfig",
    "AzureOpenAIChatModel",
    "LocalChatModel",
    "OpenRouterChatModel",
    "run_chat",
]
