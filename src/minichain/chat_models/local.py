# src/minichain/chat_models/local.py
from typing import Any
from openai import OpenAI
from .openai import OpenAILikeChatModel
from .base import LocalChatConfig

class LocalChatModel(OpenAILikeChatModel):
    def __init__(self, config: LocalChatConfig, **kwargs: Any):
        super().__init__(config=config)
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.kwargs = kwargs
