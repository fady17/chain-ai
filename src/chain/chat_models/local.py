# src/chain/chat_models/local.py
from typing import Any, Dict, Union, List, Iterator
from openai import OpenAI
from .base import BaseChatModel, LocalChatConfig
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatResult, TokenUsage

class LocalChatModel(BaseChatModel):
    """A self-contained chat model for local, OpenAI-compatible servers."""
    def __init__(self, config: LocalChatConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.model_name = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.system_prompt = getattr(config, 'system_prompt', None)

    def _prepare_messages(self, input_data: Union[str, List[BaseMessage]]) -> List[Dict[str, str]]:
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = [{"role": "system" if isinstance(msg, SystemMessage) else 
                         "assistant" if isinstance(msg, AIMessage) else "user", 
                         "content": msg.content} for msg in input_data]
        if not any(msg["role"] == "system" for msg in messages) and self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        messages = self._prepare_messages(input_data)
        params = {"model": self.model_name, "messages": messages, "temperature": self.temperature, **self.api_kwargs}
        if self.max_tokens:
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
            content=content, model_name=completion.model, token_usage=token_usage,
            finish_reason=completion.choices[0].finish_reason, raw=completion,
        )

    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        messages = self._prepare_messages(input_data)
        params = {"model": self.model_name, "messages": messages, "temperature": self.temperature, "stream": True, **self.api_kwargs}
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

