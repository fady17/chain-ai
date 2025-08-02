# src/minichain/embeddings/azure.py
"""
Implementation for Azure OpenAI embedding models.
"""
import os
from openai import AzureOpenAI
from minichain.embeddings.openai import OpenAILikeEmbeddings # Inherit from our new base class

class AzureOpenAIEmbeddings(OpenAILikeEmbeddings):
    """
    Connects to an Azure OpenAI deployment to generate text embeddings.
    """
    def __init__(self, deployment_name: str):
        """
        Initializes the AzureOpenAIEmbeddings client.

        Args:
            deployment_name (str): The name of your deployed embedding model in Azure.
        """
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS")
        api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_EMBEDDINGS_VERSION", "2024-02-01")

        if not azure_endpoint or not api_key:
            raise ValueError(
            "AZURE_OPENAI_ENDPOINT_EMBEDDINGS and AZURE_OPENAI_EMBEDDINGS_API_KEY "
            "environment variables must be set."
        )

        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
        )
        # The 'model_name' attribute is used by the base class for the API call.
        # For Azure, this is the deployment name.
        self.model_name = deployment_name
