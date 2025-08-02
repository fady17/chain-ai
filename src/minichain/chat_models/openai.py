# src/minichain/chat_models/openai.py
"""
Provides a base class for chat models that use an OpenAI-compatible API.
This centralizes logic for message formatting and API calls, reducing code
duplication between implementations like Azure and local servers.
"""
from typing import Union, List, Dict, Any, cast
from openai import OpenAI, AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseChatModel
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage

class OpenAILikeChatModel(BaseChatModel):
    """
    A base class that handles the core logic for chat completion using an
    API that follows the OpenAI SDK's conventions.
    """
    client: OpenAI | AzureOpenAI
    model_name: str
    temperature: float = 0.7
    max_tokens: int | None = None
    kwargs: Dict[str, Any]

    def _messages_to_openai_format(self, messages: List[BaseMessage]) -> List[ChatCompletionMessageParam]:
        """Converts our Pydantic Message objects to the dictionary format required by the OpenAI API."""
        openai_messages: List[ChatCompletionMessageParam] = []
        for msg in messages:
            # --- FIX: Construct the dictionary directly to help type inference ---
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            # Default to 'user' for HumanMessage or any other BaseMessage subclass
            else:
                openai_messages.append({"role": "user", "content": msg.content})
        
        # This explicit cast can also satisfy stricter type checkers if needed,
        # but the inline construction is often sufficient.
        return cast(List[ChatCompletionMessageParam], openai_messages)


    def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
        """
        Handles the logic for preparing and sending a request to an
        OpenAI-compatible chat completions endpoint.
        """
        messages: List[ChatCompletionMessageParam]
        if isinstance(input_data, str):
            messages = [{"role": "user", "content": input_data}]
        else:
            messages = self._messages_to_openai_format(input_data)
        
        # Assemble the parameters for the API call
        completion_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            **self.kwargs,
        }
        
        # Only add max_tokens if it has been set
        if self.max_tokens is not None:
            completion_params["max_tokens"] = self.max_tokens
            
        response = self.client.chat.completions.create(**completion_params)
        
        # Extract and return the response content
        return response.choices[0].message.content or ""