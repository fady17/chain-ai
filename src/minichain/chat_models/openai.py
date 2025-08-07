# src/minichain/chat_models/openai_like.py
from typing import Union, List, Dict, Any, Iterator
from openai import OpenAI

from .base import BaseChatModel, ChatModelConfig
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatResult, TokenUsage

class OpenAILikeChatModel(BaseChatModel):
    """
    A CONCRETE implementation for OpenAI-compatible chat APIs.
    This class fulfills the abstract methods of BaseChatModel.
    """
    client: OpenAI
    model_name: str
    temperature: float
    max_tokens: Union[int, None]
    system_prompt: Union[str, None]

    def __init__(self, config: ChatModelConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        self.model_name = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.system_prompt = getattr(config, 'system_prompt', None)
        # This will be overridden by subclasses
        self.client = None # type: ignore

    def _prepare_messages(self, input_data: Union[str, List[BaseMessage]]) -> List[Dict[str, str]]:
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = [
                {"role": "system" if isinstance(msg, SystemMessage) else 
                         "assistant" if isinstance(msg, AIMessage) else "user", 
                 "content": msg.content} 
                for msg in input_data
            ]
        
        has_system_message = any(msg["role"] == "system" for msg in messages)
        if not has_system_message and self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    # --- IMPLEMENTING THE ABSTRACT METHODS ---
    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        messages = self._prepare_messages(input_data)
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            **self.api_kwargs,
        }
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        completion = self.client.chat.completions.create(**params)

        usage = completion.usage
        token_usage = TokenUsage(
            completion_tokens=usage.completion_tokens if usage else None,
            prompt_tokens=usage.prompt_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
        return ChatResult(
            content=completion.choices[0].message.content or "",
            model_name=completion.model,
            token_usage=token_usage,
            finish_reason=completion.choices[0].finish_reason,
            raw=completion,
        )

    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        messages = self._prepare_messages(input_data)
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            **self.api_kwargs,
        }
        if self.max_tokens:
            params["max_tokens"] = self.max_tokens

        stream = self.client.chat.completions.create(**params)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
