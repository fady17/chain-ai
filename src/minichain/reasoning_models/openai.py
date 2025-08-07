# src/minichain/reasoning_models/openai.py
"""
Provides an interface to OpenAI's official reasoning models, using the
dedicated /v1/responses endpoint.
"""
import os
from typing import Union, List, Dict, Iterator
from openai import OpenAI
from openai.types.responses import Response

from .base import BaseReasoningModel, ReasoningModelConfig
from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatResult, TokenUsage

class OpenAIReasoningModel(BaseReasoningModel):
    """An official OpenAI reasoning model that correctly uses client.responses.create()."""
    client: OpenAI
    config: ReasoningModelConfig

    def __init__(self, config: ReasoningModelConfig):
        super().__init__(config=config)
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _prepare_input(self, input_data: Union[str, List[BaseMessage]]) -> List[Dict[str, str]]:
        if isinstance(input_data, str):
            return [{"role": "user", "content": input_data}]
        messages: List[Dict[str, str]] = []
        for msg in input_data:
            role = "developer" if isinstance(msg, SystemMessage) else \
                   "assistant" if isinstance(msg, AIMessage) else "user"
            messages.append({"role": role, "content": msg.content})
        return messages

    def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
        """Generates a response using the client.responses.create() method."""
        api_input = self._prepare_input(input_data)
        params = {
            "model": self.config.model,
            "input": api_input,
            "reasoning": {"effort": self.config.reasoning_effort},
        }
        if self.config.max_output_tokens:
            params["max_output_tokens"] = self.config.max_output_tokens

        response: Response = self.client.responses.create(**params)

        usage = response.usage
        token_usage = TokenUsage(
            completion_tokens=usage.output_tokens if usage else None,
            prompt_tokens=usage.input_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
        )
        finish_reason = response.status
        if response.status == "incomplete" and response.incomplete_details:
            finish_reason = response.incomplete_details.reason

        return ChatResult(
            content=response.output_text or "",
            model_name=response.model,
            token_usage=token_usage,
            finish_reason=finish_reason,
            raw=response,
        )

    def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
        """The /v1/responses endpoint is blocking; this conforms to the interface by yielding the single result."""
        result = self.generate(input_data)
        yield result.content
# # src/minichain/reasoning_models/openai.py
# """
# Provides an interface to OpenAI's official reasoning models, using the
# dedicated /v1/responses endpoint.
# """

# import os
# from typing import Union, List, Dict, Iterator
# from openai import OpenAI
# from openai.types.responses import Response

# from .base import BaseReasoningModel, OpenAIReasoningConfig
# from ..core.types import BaseMessage, SystemMessage, HumanMessage, AIMessage, ChatResult, TokenUsage

# class OpenAIReasoningModel(BaseReasoningModel):
#     """An official OpenAI reasoning model that uses client.responses.create()."""
#     client: OpenAI
#     config: OpenAIReasoningConfig

#     def __init__(self, config: OpenAIReasoningConfig):
#         super().__init__(config=config)
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OPENAI_API_KEY environment variable not set.")
#         self.client = OpenAI(api_key=api_key)

#     def _prepare_input(self, input_data: Union[str, List[BaseMessage]]) -> List[Dict[str, str]]:
#         if isinstance(input_data, str):
#             return [{"role": "user", "content": input_data}]
#         messages: List[Dict[str, str]] = []
#         for msg in input_data:
#             role = "developer" if isinstance(msg, SystemMessage) else \
#                    "assistant" if isinstance(msg, AIMessage) else "user"
#             messages.append({"role": role, "content": msg.content})
#         return messages

#     def generate(self, input_data: Union[str, List[BaseMessage]]) -> ChatResult:
#         """Generates a response using the client.responses.create() method."""
#         api_input = self._prepare_input(input_data)
#         params = {
#             "model": self.config.model,
#             "input": api_input,
#             "reasoning": {"effort": self.config.reasoning_effort},
#         }
#         if self.config.max_output_tokens:
#             params["max_output_tokens"] = self.config.max_output_tokens

#         response: Response = self.client.responses.create(**params)

#         usage = response.usage
#         token_usage = TokenUsage(
#             completion_tokens=usage.output_tokens if usage else None,
#             prompt_tokens=usage.input_tokens if usage else None,
#             total_tokens=usage.total_tokens if usage else None,
#         )
#         finish_reason = response.status
#         if response.status == "incomplete" and response.incomplete_details:
#             finish_reason = response.incomplete_details.reason

#         return ChatResult(
#             content=response.output_text or "",
#             model_name=response.model,
#             token_usage=token_usage,
#             finish_reason=finish_reason,
#             raw=response,
#         )

#     def stream(self, input_data: Union[str, List[BaseMessage]]) -> Iterator[str]:
#         """The /v1/responses endpoint is blocking; this conforms to the interface by yielding the single result."""
#         result = self.generate(input_data)
#         yield result.content