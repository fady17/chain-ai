# src/minichain/memory/azure_ai_search.py
"""
A cloud-based vector store implementation using Azure AI Search.
"""
import os
from typing import List, Any, Tuple
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import ResourceExistsError
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex, SearchField, SearchFieldDataType,
        VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile
    )
    from azure.search.documents.models import VectorizedQuery
except ImportError:
    raise ImportError(
        "Azure Search dependencies not found. Please run `pip install minichain-ai[azure]` "
        "to use AzureAISearchVectorStore."
    )

from ..core.types import Document
from ..embeddings.base import BaseEmbeddings
from .base import BaseVectorStore

class AzureAISearchVectorStore(BaseVectorStore):
    """
    A vector store that uses Azure AI Search for scalable, cloud-based
    similarity searches.
    """
    def __init__(self, embeddings: BaseEmbeddings, index_name: str, **kwargs: Any):
        super().__init__(embeddings=embeddings, **kwargs)

        endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        admin_key = os.getenv("AZURE_AI_SEARCH_ADMIN_KEY")

        if not endpoint or not admin_key:
            raise ValueError(
                "AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_ADMIN_KEY must be set in environment."
            )

        self.index_name = index_name
        self.credential = AzureKeyCredential(admin_key)
        self.index_client = SearchIndexClient(endpoint, self.credential)
        self.search_client = SearchClient(endpoint, index_name, self.credential)
        
        self._ensure_index_exists()

    def _get_embedding_dimension(self) -> int:
        """Helper to determine the embedding dimension."""
        return len(self.embedding_function.embed_query("test"))

    def _ensure_index_exists(self):
        """Creates the search index if it does not already exist."""
        try:
            self.index_client.get_index(self.index_name)
        except Exception: # Index does not exist
            dimension = self._get_embedding_dimension()
            self._create_index(dimension)

    def _create_index(self, dimension: int):
        """Defines and creates the Azure AI Search index schema."""
        fields = [
            SearchField(name="id", type=SearchFieldDataType.String, key=True),
            SearchField(name="page_content", type=SearchFieldDataType.String, searchable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimension,
                vector_search_profile_name="default-hnsw-profile",
            ),
            SearchField(name="source", type=SearchFieldDataType.String, filterable=True, sortable=True),
        ]
        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="default-hnsw-algo")],
            profiles=[VectorSearchProfile(name="default-hnsw-profile", algorithm_configuration_name="default-hnsw-algo")]
        )
        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        
        try:
            self.index_client.create_index(index)
        except ResourceExistsError:
            # Handles a rare race condition where another process creates the index
            # between our check and our creation attempt.
            pass

    def add_documents(self, documents: List[Document]) -> None:
        """Embeds documents and uploads them to the Azure AI Search index."""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        vectors = self.embedding_function.embed_documents(texts)
        
        # A simple hashing approach for document IDs. In a production system,
        # a more robust unique ID (like a UUID) would be preferable.
        docs_to_upload = [
            {
                "id": str(hash(doc.page_content)),
                "page_content": doc.page_content,
                "content_vector": vectors[i],
                "source": doc.metadata.get("source", "")
            }
            for i, doc in enumerate(documents)
        ]
        
        self.search_client.upload_documents(docs_to_upload)

    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Performs a vector similarity search. For Azure AI Search, a higher
        score indicates higher similarity.
        """
        query_vector = self.embedding_function.embed_query(query)
        vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="content_vector")
        
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["page_content", "source"],
        )
        
        # Modify the return type to List[Tuple[Document, float]]
        return [
            (
                Document(
                    page_content=res["page_content"],
                    metadata={"source": res.get("source")} # Score is no longer in metadata
                ),
                float(res["@search.score"])
            )
            for res in results
        ]
