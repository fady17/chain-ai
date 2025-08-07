
"""
Provides a base class for embedding models that use an OpenAI-compatible API.
This reduces code duplication between different provider implementations (e.g., Azure, local).
"""
from typing import List, Union
from openai import OpenAI, AzureOpenAI
from minichain.embeddings.base import BaseEmbeddings

class OpenAILikeEmbeddings(BaseEmbeddings):
    """
    A base class that handles the core logic for embedding text using an
    API that follows the OpenAI SDK's conventions.
    """
    client: Union[OpenAI, AzureOpenAI]
    model_name: str # The model/deployment name to be passed to the API

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents. Handles empty input gracefully.
        """
        if not texts:
            return []
        
        # Replace empty strings with a single space, as some APIs fail on empty input
        processed_texts = [text.strip() or " " for text in texts]
        
        response = self.client.embeddings.create(
            input=processed_texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query."""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model_name
        )
        return response.data[0].embedding
        # We can reuse the batch method for a single item for consistency.
        # This also benefits from the empty string handling.
        # embeddings = self.embed_documents([text])
        # return embeddings[0]