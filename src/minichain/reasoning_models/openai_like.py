# # src/minichain/reasoning_models/openai_like.py
# """
# Provides a common base class for reasoning models that use an OpenAI-compatible
# Chat Completions endpoint. Modified to remove unsupported 'reasoning' parameter.
# """

# from typing import Union, List, Dict, Any, Iterator
# from openai import OpenAI

# from .base import BaseReasoningModel, ReasoningModelConfig
# from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatResult, TokenUsage

# class OpenAILikeReasoningModel(BaseReasoningModel):
#     """A base class for OpenAI-compatible reasoning APIs."""
#     client: OpenAI
#     config: ReasoningModelConfig
#     api_kwargs: Dict[str, Any]

#     def __init__(self, config: ReasoningModelConfig):
#         super().__init__(config=config)
#         self.api_kwargs = {}

#     def _prepare_messages(self, input_data: Union[str, List[BaseMessage]]) -> List[Dict[str, str]]:
#         if isinstance(input_data, str):
#             return [{"role": "user", "content": input_data}]
#         messages: List[Dict[str, str]] = []
#         for msg in input_data:
#             role = "system" if isinstance(msg, SystemMessage) else \
#                    "assistant" if isinstance(msg, AIMessage) else "user"
#             messages.append({"role": role, "content": msg.content})
#         return messages

#     def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
#         """Generates a response using chat.completions.create without reasoning parameter."""
#         messages = self._prepare_messages(input_data)
        
#         # Build system message with reasoning effort hint instead of using reasoning parameter
#         if messages and self.config.reasoning_effort != "medium":
#             system_msg = f"Please think carefully and provide a {self.config.reasoning_effort}-effort response."
#             if messages[0]["role"] == "system":
#                 messages[0]["content"] = f"{system_msg}\n\n{messages[0]['content']}"
#             else:
#                 messages.insert(0, {"role": "system", "content": system_msg})
        
#         params: Dict[str, Any] = {
#             "model": self.config.model,
#             "messages": messages,
#             **self.api_kwargs,
#         }
#         if self.config.max_output_tokens:
#             params["max_tokens"] = self.config.max_output_tokens

#         completion = self.client.chat.completions.create(**params)

#         usage = completion.usage
#         token_usage = TokenUsage(
#             completion_tokens=usage.completion_tokens if usage else None,
#             prompt_tokens=usage.prompt_tokens if usage else None,
#             total_tokens=usage.total_tokens if usage else None,
#         )
#         return ChatResult(
#             content=completion.choices[0].message.content or "",
#             model_name=completion.model,
#             token_usage=token_usage,
#             finish_reason=completion.choices[0].finish_reason,
#             raw=completion,
#         )

#     def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
#         """Provides a streaming response via the chat completions endpoint."""
#         messages = self._prepare_messages(input_data)
        
#         # Build system message with reasoning effort hint
#         if messages and self.config.reasoning_effort != "medium":
#             system_msg = f"Please think carefully and provide a {self.config.reasoning_effort}-effort response."
#             if messages[0]["role"] == "system":
#                 messages[0]["content"] = f"{system_msg}\n\n{messages[0]['content']}"
#             else:
#                 messages.insert(0, {"role": "system", "content": system_msg})
        
#         params: Dict[str, Any] = {
#             "model": self.config.model,
#             "messages": messages,
#             "stream": True,
#             **self.api_kwargs,
#         }
#         if self.config.max_output_tokens:
#             params["max_tokens"] = self.config.max_output_tokens

#         stream = self.client.chat.completions.create(**params)
#         for chunk in stream:
#             if chunk.choices and chunk.choices[0].delta.content:
#                 yield chunk.choices[0].delta.content