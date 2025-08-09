# src/minichain/chat_models/openrouter.py
import os
import json
from typing import Any, Dict, Union, List, Iterator
import requests
from .base import BaseChatModel, OpenRouterConfig
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatResult, TokenUsage

class OpenRouterChatModel(BaseChatModel):
    """A self-contained chat model for the OpenRouter API, supporting special parameters like 'reasoning'."""
    def __init__(self, config: OpenRouterConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        self.config: OpenRouterConfig = config # For type hinting
        self.model_name = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.system_prompt = getattr(config, 'system_prompt', None)
        self.api_key = os.getenv("OPENAI_API_KEY") # Using your preferred name
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set for OpenRouter.")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

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

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool) -> Dict[str, Any]:
        payload = {"model": self.model_name, "messages": messages, "temperature": self.temperature, "stream": stream, **self.api_kwargs}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.config.site_url: headers["HTTP-Referer"] = self.config.site_url
        if self.config.site_name: headers["X-Title"] = self.config.site_name
        return headers

    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        messages = self._prepare_messages(input_data)
        payload = self._build_payload(messages, stream=False)
        headers = self._build_headers()
        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        usage_data = data.get("usage", {})
        token_usage = TokenUsage(
            completion_tokens=usage_data.get("completion_tokens"),
            prompt_tokens=usage_data.get("prompt_tokens"),
            total_tokens=usage_data.get("total_tokens"),
        )
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        return ChatResult(
            content=content, model_name=data.get("model", self.model_name),
            token_usage=token_usage, finish_reason=choice.get("finish_reason"), raw=data,
        )

    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        messages = self._prepare_messages(input_data)
        payload = self._build_payload(messages, stream=True)
        headers = self._build_headers()
        with requests.post(self.base_url, headers=headers, data=json.dumps(payload), stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "): line_str = line_str[6:]
                    if line_str == "[DONE]": break
                    try:
                        chunk_data = json.loads(line_str)
                        delta = chunk_data["choices"][0]["delta"]
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue
