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
# # src/minichain/chat_models/openai.py

# import re
# from typing import Union, List, Dict, Any, Iterator, cast
# from openai import OpenAI, AzureOpenAI
# from openai.types.chat import ChatCompletionMessageParam

# from .base import BaseChatModel, ChatModelConfig
# from ..core.types import (
#     BaseMessage, SystemMessage, HumanMessage, AIMessage, 
#     ChatResult, TokenUsage
# )

# class OpenAILikeChatModel(BaseChatModel):
#     """
#     A robust base class for chat models that use an OpenAI-compatible API.
#     It defaults to fast, non-streaming calls and provides an optional,
#     explicit mechanism to handle "thinky" models that require stream processing.
#     """
#     client: Union[OpenAI, AzureOpenAI]
#     model_name: str
#     temperature: float
#     max_tokens: Union[int, None]
#     system_prompt: str | None
#     force_stream_for_invoke: bool
#     kwargs: Dict[str, Any]

#     def __init__(self, config: ChatModelConfig, **kwargs: Any):
#         super().__init__(config=config)
#         self.system_prompt = getattr(config, 'system_prompt', None)
#         # Add the new flag, defaulting to False (the fast path)
#         self.force_stream_for_invoke = getattr(config, 'force_stream_for_invoke', False)
        
#         # We need to initialize the attributes that are defined in the class
#         self.model_name = "" # Will be set by subclasses
#         self.temperature = config.temperature
#         self.max_tokens = config.max_tokens
#         self.kwargs = kwargs
        
#     def _prepare_messages(self, input_data: Union[str, List[BaseMessage]]) -> List[ChatCompletionMessageParam]:
#         """
#         Prepares the final list of messages for the API call,
#         intelligently adding the system prompt if needed.
#         """
#         messages: List[Dict[str, str]]
        
#         if isinstance(input_data, str):
#             messages = [{"role": "user", "content": input_data}]
#         else:
#             messages = []
#             for msg in input_data:
#                 if isinstance(msg, SystemMessage):
#                     messages.append({"role": "system", "content": msg.content})
#                 elif isinstance(msg, AIMessage):
#                     messages.append({"role": "assistant", "content": msg.content})
#                 else: # HumanMessage
#                     messages.append({"role": "user", "content": msg.content})

#         has_system_message = any(msg["role"] == "system" for msg in messages)

#         if not has_system_message and self.system_prompt:
#             messages.insert(0, {"role": "system", "content": self.system_prompt})
        
#         return cast(List[ChatCompletionMessageParam], messages)

#     @staticmethod
#     def _clean_streamed_text(text: str) -> str:
#         """Removes <think> blocks from a streamed response."""
#         return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

#     # --- PRIMARY METHOD: generate() ---
#     def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
#         """
#         Generates a rich, structured response.
        
#         Defaults to a fast non-streaming call. If `force_stream_for_invoke` is
#         True in the config, it uses a streaming call and cleans the output,
#         which is more reliable for "thinky" models.
#         """
#         if self.force_stream_for_invoke:
#             return self._generate_via_stream(input_data)
#         else:
#             return self._generate_via_api(input_data)

#     def _generate_via_api(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
#         """The default, fast path using a non-streaming API call."""
#         messages = self._prepare_messages(input_data)
        
#         completion_params = {
#             "model": self.model_name,
#             "messages": messages,
#             "temperature": self.temperature,
#             "stream": False,
#             **self.kwargs,
#         }
#         if self.max_tokens is not None:
#             completion_params["max_tokens"] = self.max_tokens
            
#         completion = self.client.chat.completions.create(**completion_params)

#         usage_data = completion.usage
#         token_usage = TokenUsage(
#             completion_tokens=usage_data.completion_tokens if usage_data else None,
#             prompt_tokens=usage_data.prompt_tokens if usage_data else None,
#             total_tokens=usage_data.total_tokens if usage_data else None,
#         )

#         return ChatResult(
#             content=completion.choices[0].message.content or "",
#             model_name=completion.model,
#             token_usage=token_usage,
#             finish_reason=completion.choices[0].finish_reason,
#             raw=completion,
#         )

#     def _generate_via_stream(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
#         """The fallback path for "thinky" models."""
#         full_content = "".join(self.stream(input_data))
#         cleaned_content = self._clean_streamed_text(full_content)

#         # Metadata is limited when streaming this way
#         return ChatResult(
#             content=cleaned_content,
#             model_name=self.model_name,
#             raw=full_content,
            
#         ) # type: ignore

#     # --- CONVENIENCE WRAPPER: invoke() ---
#     def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
#         """Generates a clean, complete string response."""
#         return self.generate(input_data).content

#     # --- UI METHOD: stream() ---
#     def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
#         """Generates a response as a raw, real-time stream of text chunks."""
#         messages = self._prepare_messages(input_data)
        
#         completion_params = {
#             "model": self.model_name,
#             "messages": messages,
#             "temperature": self.temperature,
#             "stream": True,
#             **self.kwargs,
#         }
#         if self.max_tokens is not None:
#             completion_params["max_tokens"] = self.max_tokens
            
#         response_stream = self.client.chat.completions.create(**completion_params)
        
#         for chunk in response_stream:
#             if chunk.choices and chunk.choices[0].delta.content is not None:
#                 yield chunk.choices[0].delta.content
# # # src/minichain/chat_models/openai.py
# # """
# # Provides a base class for chat models that use an OpenAI-compatible API.
# # This centralizes logic for both blocking (`invoke`) and streaming (`stream`)
# # API calls, reducing code duplication.
# # """
# # import re
# # from typing import Union, List, Dict, Any, Iterator, cast
# # from openai import OpenAI, AzureOpenAI
# # from openai.types.chat import ChatCompletionMessageParam, ChatCompletionChunk

# # from .base import BaseChatModel
# # from ..core.types import BaseMessage, ChatResult, SystemMessage, HumanMessage, AIMessage, TokenUsage

# # class OpenAILikeChatModel(BaseChatModel):
# #     """
# #     A base class that handles core logic for OpenAI-compatible chat APIs.
# #     """
# #     # client: OpenAI | AzureOpenAI
# #     client: Union[OpenAI,AzureOpenAI]
# #     model_name: str
# #     temperature: float = 0.7
# #     max_tokens: Union[int, None] = None
# #     kwargs: Dict[str, Any]
# #     system_prompt: str | None

# #     def _messages_to_openai_format(self, messages: List[BaseMessage]) -> List[ChatCompletionMessageParam]:
# #         """Converts Mini-Chain Message objects to the OpenAI API dictionary format."""
# #         openai_messages: List[Dict[str, str]] = []
# #         for msg in messages:
# #             if isinstance(msg, SystemMessage):
# #                 openai_messages.append({"role": "system", "content": msg.content})
# #             elif isinstance(msg, AIMessage):
# #                 openai_messages.append({"role": "assistant", "content": msg.content})
# #             else: # HumanMessage or other
# #                 openai_messages.append({"role": "user", "content": msg.content})
# #         return cast(List[ChatCompletionMessageParam], openai_messages)
# #     @staticmethod
# #     def _clean_response(text: str) -> str:
# #         """
# #         Removes preliminary "thinking" or "reasoning" blocks from the model's output.
# #         Handles both <think> tags and untagged reasoning paragraphs.
# #         """
# #         # First, remove any formal <think> blocks
# #         cleaned_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        
# #         # Second, heuristically check if the text *starts* with a reasoning block
# #         # that looks like the "Okay, the user is asking..." pattern.
# #         # This is more brittle but necessary for models that don't use tags.
# #         # We look for a paragraph that ends right before the "real" answer. A common
# #         # pattern is that the real answer starts with a direct address or statement.
# #         # This is complex to generalize, so we can start with a simpler rule:
# #         # If the text contains a well-known start to the answer, we can split on it.
# #         # For now, let's keep it simple and just rely on the <think> tag removal,
# #         # but this is where more advanced logic would go.
# #         # A more robust solution for untagged reasoning might require prompt engineering.
        
# #         return cleaned_text
# #     def _clean_raw_output(self, text: str) -> str:
# #         """
# #         Removes preliminary "thinking" blocks from the model's raw output.
# #         This function is designed to be robust and handle the <think>...</think> pattern.
# #         """
# #         # The regex is correct. DOTALL is crucial for multi-line <think> blocks.
# #         # The \s* at the end handles newlines and spaces between the block and the answer.
# #         return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


# #    # --- REVISED generate() METHOD ---
# #     def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
# #         """
# #         Generates a rich, structured response by consuming a stream for robustness.
# #         This is the primary method for getting a complete and cleaned response
# #         with full metadata.
# #         """
# #         # 1. Get the raw, potentially "unclean" stream of text
# #         full_unclean_content = "".join(self.stream(input_data))

# #         # 2. Clean the fully assembled response
# #         cleaned_content = self._clean_raw_output(full_unclean_content)

# #         # 3. Create the result object. 
# #         # For now, we are prioritizing the correct content. Metadata like
# #         # token usage from a stream is complex and often not provided by all APIs.
# #         # We will return the original unclean content in the 'raw' field for debugging.
# #         return ChatResult(
# #             content=cleaned_content,
# #             model_name=self.model_name,
# #             # Placeholder for token usage and finish reason, as they are not
# #             # reliably available from a simple stream consumption.
# #             token_usage=TokenUsage(), 
# #             finish_reason=None, 
# #             raw=full_unclean_content,
# #         )
# #     def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
# #         """
# #         Generates a clean, complete string response.
# #         This is a convenience wrapper around the `generate` method.
# #         """
# #         return self.generate(input_data).content

# #      # --- stream() is the fundamental building block ---
# #     def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
# #         """
# #         Generates a response as a raw, real-time stream of text chunks.
# #         This may include "thinking" blocks and is ideal for UI applications.
# #         """
# #         if isinstance(input_data, str):
# #             messages = [{"role": "user", "content": input_data}]
# #         else:
# #             messages = self._messages_to_openai_format(input_data)
        
# #         completion_params = {
# #             "model": self.model_name,
# #             "messages": messages,
# #             "temperature": self.temperature,
# #             "stream": True,
# #             **self.kwargs,
# #         }
        
# #         if self.max_tokens is not None:
# #             completion_params["max_tokens"] = self.max_tokens
            
# #         response_stream = self.client.chat.completions.create(**completion_params)
        
# #         for chunk in response_stream:
# #             if chunk.choices and chunk.choices[0].delta.content is not None:
# #                 yield chunk.choices[0].delta.content

# #     # def invoke(self, input_data: Union[str, List[BaseMessage]]) -> str:
# #     #     """Handles a standard, blocking request to the chat completions endpoint."""
# #     #     chunks = list(self.stream(input_data))
# #     #     # return "".join(chunks)
# #     #     full_response = "".join(chunks)
# #     # # Add post-processing to remove the think block
# #     #     clean_response = re.sub(r'<think>.*?</think>\s*', '', full_response, flags=re.DOTALL).strip()
# #     #     return clean_response
        
# #     # def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
# #     #     """Handles a non-streaming request to return a structured ChatResult."""
# #     #     messages: List[ChatCompletionMessageParam]
# #     #     if isinstance(input_data, str):
# #     #         messages = [{"role": "user", "content": input_data}]
# #     #     else:
# #     #         messages = self._messages_to_openai_format(input_data)
        
# #     #     completion_params = {
# #     #         "model": self.model_name,
# #     #         "messages": messages,
# #     #         "temperature": self.temperature,
# #     #         "stream": False,  # Ensure stream is False to get metadata
# #     #         **self.kwargs,
# #     #     }
        
# #     #     if self.max_tokens is not None:
# #     #         completion_params["max_tokens"] = self.max_tokens
            
# #     #     completion = self.client.chat.completions.create(**completion_params)

# #     #     usage_data = completion.usage
# #     #     token_usage = TokenUsage(
# #     #         completion_tokens=usage_data.completion_tokens if usage_data else None,
# #     #         prompt_tokens=usage_data.prompt_tokens if usage_data else None,
# #     #         total_tokens=usage_data.total_tokens if usage_data else None,
# #     #     )

# #     #     return ChatResult(
# #     #         content=completion.choices[0].message.content or "",
# #     #         model_name=completion.model,
# #     #         token_usage=token_usage,
# #     #         finish_reason=completion.choices[0].finish_reason,
# #     #         raw=completion,
# #     #     )

# #     # def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
# #     #     """Handles a streaming request to the chat completions endpoint."""
# #     #     messages: List[ChatCompletionMessageParam]
# #     #     if isinstance(input_data, str):
# #     #         messages = [{"role": "user", "content": input_data}]
# #     #     else:
# #     #         messages = self._messages_to_openai_format(input_data)
        
# #     #     completion_params = {
# #     #         "model": self.model_name,
# #     #         "messages": messages,
# #     #         "temperature": self.temperature,
# #     #         "stream": True,
# #     #         **self.kwargs,
# #     #     }
        
# #     #     if self.max_tokens is not None:
# #     #         completion_params["max_tokens"] = self.max_tokens
            
# #     #     stream = self.client.chat.completions.create(**completion_params)
        
# #     #     # ---  loop to handle all stream events ---
# #     #     for chunk in stream:
# #     #         # Check if the choices list is not empty. Some chunks are for
# #     #         # metadata and have an empty choices list.
# #     #         if chunk.choices:
# #     #             content = chunk.choices[0].delta.content
# #     #             # The content can also be None in some chunks, so check for that too.
# #     #             if content is not None:
# #     #                 yield content
  