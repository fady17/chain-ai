# src/minichain/chat_models/openrouter.py
import os
from typing import Any, Dict
from openai import OpenAI
from .openai import OpenAILikeChatModel
from .base import OpenRouterConfig

class OpenRouterChatModel(OpenAILikeChatModel):
    def __init__(self, config: OpenRouterConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        extra_headers: Dict[str, Any] = {}
        if config.site_url:
            extra_headers["HTTP-Referer"] = config.site_url
        if config.site_name:
            extra_headers["X-Title"] = config.site_name
        
        if extra_headers:
            self.api_kwargs.setdefault('extra_headers', {}).update(extra_headers)
