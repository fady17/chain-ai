# src/minichain/embeddings/azure.py
"""
Azure OpenAI embeddings implementation.
"""

import os
from typing import List
from openai import AzureOpenAI

from .base import BaseEmbeddings

class AzureOpenAIEmbeddings(BaseEmbeddings):
    """Azure OpenAI embeddings implementation"""

    def __init__(self, 
                 deployment_name: str, 
                 model_name: str = "text-embedding-3-small", 
                 **kwargs):
        super().__init__(model_name, **kwargs)

        self.deployment_name = deployment_name
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDINGS")
        self.api_key = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_KEY")
        self.api_version = os.getenv("AZURE_OPENAI_API_EMBEDDINGS_VERSION", "2024-12-01-preview")

        if not all([self.azure_endpoint, self.api_key]):
             raise ValueError(
                "AZURE_OPENAI_ENDPOINT_EMBEDDINGS and AZURE_OPENAI_EMBEDDINGS_API_KEY "
                "environment variables required"
            )

        self.client = AzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint, # type: ignore
            api_key=self.api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Azure OpenAI API"""
        response = self.client.embeddings.create(
            input=texts,
            model=self.deployment_name, # Azure uses deployment name for the model
            **self.kwargs
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Azure OpenAI API"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.deployment_name,
            **self.kwargs
        )
        return response.data[0].embedding