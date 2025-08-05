# src/minichain/chat_models/openai_like.py
"""
Provides a base class for chat models that use an OpenAI-compatible API.
This centralizes logic for both blocking (`invoke`) and streaming (`stream`)
API calls, reducing code duplication.
"""
from typing import Union, List, Dict, Any, Iterator, cast
from openai import OpenAI, AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseChatModel
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage

class OpenAILikeChatModel(BaseChatModel):
    """
    A base class that handles core logic for OpenAI-compatible chat APIs.
    """
    client: OpenAI | AzureOpenAI
    model_name: str
    temperature: float = 0.7
    max_tokens: int | None = None
    kwargs: Dict[str, Any]

    def _messages_to_openai_format(self, messages: List[BaseMessage]) -> List[ChatCompletionMessageParam]:
        """Converts Mini-Chain Message objects to the OpenAI API dictionary format."""
        openai_messages: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            else: # HumanMessage or other
                openai_messages.append({"role": "user", "content": msg.content})
        return cast(List[ChatCompletionMessageParam], openai_messages)

    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """Handles a standard, blocking request to the chat completions endpoint."""
        chunks = list(self.stream(input_data))
        return "".join(chunks)

    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        """Handles a streaming request to the chat completions endpoint."""
        messages: List[ChatCompletionMessageParam]
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = self._messages_to_openai_format(input_data)
        
        completion_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
            **self.kwargs,
        }
        
        if self.max_tokens is not None:
            completion_params["max_tokens"] = self.max_tokens
            
        stream = self.client.chat.completions.create(**completion_params)
        
        # --- FIX: A robust loop to handle all stream events ---
        for chunk in stream:
            # Check if the choices list is not empty. Some chunks are for
            # metadata and have an empty choices list.
            if chunk.choices:
                content = chunk.choices[0].delta.content
                # The content can also be None in some chunks, so check for that too.
                if content is not None:
                    yield content
        # --- END FIX ---