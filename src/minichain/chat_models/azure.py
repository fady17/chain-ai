# src/minichain/chat_models/azure.py
import os
from typing import Any
from openai import AzureOpenAI
from .openai import OpenAILikeChatModel
from .base import AzureChatConfig

class AzureOpenAIChatModel(OpenAILikeChatModel):
    def __init__(self, config: AzureChatConfig, **kwargs: Any):
        super().__init__(config=config, **kwargs)
        
        endpoint = config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        if not endpoint or not api_key:
            raise ValueError("Azure endpoint and API key must be provided or set as env variables.")
        
        self.client = AzureOpenAI(
            api_version=config.api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        # For Azure, the 'model' in the config is used as the deployment name
        # The base class already sets self.model_name = config.model
# # src/minichain/chat_models/azure.py
# import os
# from typing import Any
# from openai import AzureOpenAI
# from .openai import OpenAILikeChatModel
# from .base import AzureChatConfig

# class AzureOpenAIChatModel(OpenAILikeChatModel):
#     def __init__(self, config: AzureChatConfig, **kwargs: Any):
#         super().__init__(config=config)
        
#         # Allow overriding env vars with direct config values
#         endpoint = config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
#         api_key = config.api_key or os.getenv("AZURE_OPENAI_API_KEY")

#         if not endpoint or not api_key:
#             raise ValueError("Azure endpoint and API key must be provided or set as environment variables.")
        
#         self.client = AzureOpenAI(
#             api_version=config.api_version,
#             azure_endpoint=endpoint,
#             api_key=api_key,
#         )
#         self.model_name = config.deployment_name
#         self.temperature = config.temperature
#         self.max_tokens = config.max_tokens
#         self.kwargs = kwargs
