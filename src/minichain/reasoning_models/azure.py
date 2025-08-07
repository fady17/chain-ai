# # src/minichain/reasoning_models/azure.py
# """
# (Future-proofing) Provides an interface to Azure's reasoning models.
# """

# import os
# from openai import AzureOpenAI

# from .base import AzureReasoningConfig
# from .openai_like import OpenAILikeReasoningModel

# class AzureReasoningModel(OpenAILikeReasoningModel):
#     """
#     A placeholder for an Azure reasoning model.
#     This class assumes Azure will eventually support reasoning models
#     via an OpenAI-compatible API.
#     """
#     config: AzureReasoningConfig # type: ignore

#     def __init__(self, config: AzureReasoningConfig):
#         super().__init__(config=config)
#                 # --- Future implementation would look like this ---
#         endpoint = config.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
#         api_key = os.getenv("AZURE_OPENAI_API_KEY")

#         if not endpoint or not api_key:
#             raise ValueError("Azure endpoint and API key must be provided or set as environment variables.")
        
#         self.client = AzureOpenAI(
#             api_version=config.api_version,
#             azure_endpoint=endpoint,
#             api_key=api_key,
#         )
#         self.config.model = config.deployment_name # Azure uses deployment name as model