# # src/minichain/reasoning_models/openrouter.py
# """
# Provides an interface to reasoning models available through the OpenRouter gateway.
# """
# import os
# from openai import OpenAI
# from typing import Dict, Any

# from .base import OpenRouterReasoningConfig
# from .openai_like import OpenAILikeReasoningModel

# class OpenRouterReasoningModel(OpenAILikeReasoningModel):
#     """A reasoning model accessed via the OpenRouter gateway."""
#     config: OpenRouterReasoningConfig # type: ignore

#     def __init__(self, config: OpenRouterReasoningConfig):
#         super().__init__(config=config)
#         api_key = os.getenv("OPENROUTER_API_KEY")
#         if not api_key:
#             raise ValueError("OPENROUTER_API_KEY environment variable not set.")

#         self.client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=api_key
#         )
        
#         extra_headers: Dict[str, Any] = {}
#         if config.site_url:
#             extra_headers["HTTP-Referer"] = config.site_url
#         if config.site_name:
#             extra_headers["X-Title"] = config.site_name
        
#         if extra_headers:
#             self.api_kwargs['extra_headers'] = extra_headers