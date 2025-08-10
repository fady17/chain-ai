# src/chain/chat_models/azure.py
"""
Provides a self-contained interface for interacting with Azure OpenAI chat models.
"""

import os
from typing import Any, Dict, Union, List, Iterator

# The official openai library is used, specifically the AzureOpenAI client
from openai import AzureOpenAI

from .base import BaseChatModel, AzureChatConfig
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatResult, TokenUsage

class AzureOpenAIChatModel(BaseChatModel):
    """
    A self-contained chat model for the Azure OpenAI service.

    This class handles the specific authentication and endpoint requirements for Azure,
    inheriting only from the abstract BaseChatModel to ensure a clean implementation.
    """
    def __init__(self, config: AzureChatConfig, **kwargs: Any):
        """
        Initializes the AzureOpenAIChatModel.

        Args:
            config: An AzureChatConfig object with deployment details.
            **kwargs: Extra keyword arguments to pass to the API.
        """
        super().__init__(config=config, **kwargs)
        self.config: AzureChatConfig = config  # For better type hinting
        
        # Azure requires endpoint and api_key, which can be in the config or env vars
        endpoint = config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not endpoint or not api_key:
            raise ValueError(
                "Azure endpoint and API key must be provided in the config "
                "or set as AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
                "environment variables."
            )
        
        # Use the dedicated AzureOpenAI client
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        
        # In Azure, the 'model' is the deployment name
        self.model_name = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.system_prompt = getattr(config, 'system_prompt', None)

    def _prepare_messages(self, input_data: Union[str, List[BaseMessage]]) -> List[Dict[str, str]]:
        """Prepares the message list for the API call, including the system prompt."""
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = [
                {
                    "role": "system" if isinstance(msg, SystemMessage) else 
                            "assistant" if isinstance(msg, AIMessage) else "user",
                    "content": msg.content
                }
                for msg in input_data
            ]
        
        # Prepend system prompt if it exists and is not already in the messages
        has_system_message = any(msg["role"] == "system" for msg in messages)
        if not has_system_message and self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
            
        return messages

    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        """Generates a rich, structured response from the Azure deployment."""
        messages = self._prepare_messages(input_data)
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            **self.api_kwargs
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        completion = self.client.chat.completions.create(**params)

        usage = completion.usage
        token_usage = TokenUsage(
            completion_tokens=usage.completion_tokens if usage else None,
            prompt_tokens=usage.prompt_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
        
        content = completion.choices[0].message.content or ""
        
        return ChatResult(
            content=content,
            model_name=completion.model,
            token_usage=token_usage,
            finish_reason=completion.choices[0].finish_reason,
            raw=completion,
        )

    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        """Streams a response from the Azure deployment."""
        messages = self._prepare_messages(input_data)
        
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            **self.api_kwargs
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        stream = self.client.chat.completions.create(**params)
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
