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
# import os
# from typing import Any
# from openai import OpenAI
# from .openai import OpenAILikeChatModel
# from .base import OpenRouterConfig

# class OpenRouterChatModel(OpenAILikeChatModel):
#     def __init__(self, config: OpenRouterConfig, **kwargs: Any):
#         super().__init__(config=config, **kwargs) # Pass kwargs to parent
#         api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
#         if not api_key:
#             raise ValueError("OpenRouter API key must be provided...")
        
#         self.client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=api_key,
#         )
#         self.model_name = config.model_name
        
#         # Handle extra headers
#         extra_headers = {}
#         if config.site_url:
#             extra_headers["HTTP-Referer"] = config.site_url
#         if config.site_name:
#             extra_headers["X-Title"] = config.site_name
        
#         if extra_headers:
#             self.kwargs["extra_headers"] = {**self.kwargs.get("extra_headers", {}), **extra_headers}
# # import os
# # from typing import Any
# # from openai import OpenAI
# # from .openai import OpenAILikeChatModel
# # from .base import OpenRouterConfig

# # class OpenRouterChatModel(OpenAILikeChatModel):
# #     """
# #     A chat model that uses the OpenRouter API.
    
# #     It is compatible with the OpenAI API, but requires a specific base URL,
# #     an OpenRouter API key, and allows for custom headers for site identification.
# #     """
# #     def __init__(self, config: OpenRouterConfig, **kwargs: Any):
# #         super().__init__(config=config)
        
# #         # Allow overriding env var with a direct config value.
# #         # It's good practice to check for an OPENROUTER_API_KEY env var.
# #         api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")

# #         if not api_key:
# #             raise ValueError(
# #                 "OpenRouter API key must be provided in the config or "
# #                 "set as the OPENROUTER_API_KEY environment variable."
# #             )
        
# #         self.client = OpenAI(
# #             base_url="https://openrouter.ai/api/v1",
# #             api_key=api_key,
# #         )
# #         self.model_name = config.model_name
# #         self.temperature = config.temperature
# #         self.max_tokens = config.max_tokens

# #         # Handle extra headers for OpenRouter rankings.
# #         # We store them in self.kwargs so OpenAILikeChatModel passes them automatically.
# #         self.kwargs = kwargs.copy() 
# #         extra_headers = {}
# #         if config.site_url:
# #             extra_headers["HTTP-Referer"] = config.site_url
# #         if config.site_name:
# #             extra_headers["X-Title"] = config.site_name
        
# #         if extra_headers:
# #             # Merge with any headers the user might have passed in kwargs.
# #             # The headers from the config will take precedence.
# #             existing_headers = self.kwargs.get("extra_headers", {})
# #             self.kwargs["extra_headers"] = {**existing_headers, **extra_headers}