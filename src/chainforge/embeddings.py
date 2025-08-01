"""
Embeddings Layer for the ChainForge Framework.

This module provides the core abstraction (`BaseEmbeddings`) and concrete
implementations for converting text into dense vector representations. Each
component is designed for resilience and independent configuration.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional, Union

import openai
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

from chainforge.config import Settings
from chainforge.core import Runnable

# Standard library logging for operational visibility.
logger = logging.getLogger(__name__)

# --- Type Definitions for Clarity ---
# Defines the possible inputs to an embedding model.
EmbeddingInput = Union[str, List[str]]
# Defines the possible outputs of an embedding model.
EmbeddingOutput = Union[List[float], List[List[float]]]


class BaseEmbeddings(Runnable[EmbeddingInput, EmbeddingOutput], ABC):
    """
    Abstract base class for all embedding models.

    This class defines a `Runnable` contract for a component that can transform
    either a single string or a list of strings into a vector embedding or a
    list of vector embeddings, respectively. This dual-functionality is a
    common requirement for embedding workflows.
    """

    @abstractmethod
    async def aembed_document(self, text: str, **kwargs) -> List[float]:
        """Asynchronously create an embedding for a single piece of text."""
        raise NotImplementedError

    @abstractmethod
    async def aembed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Asynchronously create embeddings for a list of texts."""
        raise NotImplementedError

    async def ainvoke(self, input: EmbeddingInput, **kwargs) -> EmbeddingOutput:
        """
        Dispatches to the appropriate embedding method based on input data type.

        This method acts as a router, providing a single `Runnable` entry point
        for both single-document and batch-embedding tasks.

        Args:
            input: A `str` for a single document or a `List[str]` for a batch.
            **kwargs: Additional keyword arguments.

        Returns:
            The resulting embedding(s), either `List[float]` or `List[List[float]]`.
        """
        if isinstance(input, str):
            return await self.aembed_document(input, **kwargs)
        elif isinstance(input, list):
            return await self.aembed_documents(input, **kwargs)
        else:
            # This provides a safeguard against incorrect input types.
            raise TypeError("Input for embedding must be a string or a list of strings.")

    async def astream(self, input: EmbeddingInput, **kwargs) -> AsyncIterator[EmbeddingOutput]:
        """
        Yields the single embedding result.

        Embedding is an atomic operation. This method fulfills the `Runnable`
        contract by invoking the model and yielding the complete result as a
        single item in an async iterator.
        """
        result = await self.ainvoke(input, **kwargs)
        yield result


class OpenAIEmbeddings(BaseEmbeddings):
    """
    A resilient, asynchronous client for OpenAI embedding models, with support
    for both standard and Azure endpoints.
    """

    def __init__(
        self,
        settings: Settings,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        """Initializes the OpenAIEmbeddings client."""
        self.settings = settings
        _azure_endpoint = azure_endpoint or self.settings.AZURE_OPENAI_ENDPOINT

        if _azure_endpoint:
            # --- Azure Configuration Path ---
            _api_key = api_key or self.settings.AZURE_OPENAI_API_KEY
            _api_version = api_version or self.settings.OPENAI_API_VERSION
            self.model_name = azure_deployment or self.settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT

            if not all([_api_key, _azure_endpoint, self.model_name]):
                raise ValueError("For Azure, 'api_key', 'azure_endpoint', and 'azure_deployment' must be provided.")

            self.client = openai.AsyncAzureOpenAI(
                api_key=_api_key, azure_endpoint=_azure_endpoint, api_version=_api_version
            )
            logger.info(f"OpenAIEmbeddings initialized for Azure. Endpoint: {_azure_endpoint}, Deployment: {self.model_name}")
        else:
            # --- Standard OpenAI Configuration Path ---
            _api_key = api_key or self.settings.OPENAI_API_KEY
            self.model_name = "text-embedding-3-small"

            if not _api_key:
                raise ValueError("For standard OpenAI, 'api_key' must be provided.")

            self.client = openai.AsyncOpenAI(api_key=_api_key)
            logger.info("OpenAIEmbeddings initialized for standard OpenAI.")

    @retry(
        wait=wait_random_exponential(min=Settings().DEFAULT_RETRY_DELAY, max=60),
        stop=stop_after_attempt(Settings().DEFAULT_RETRY_ATTEMPTS),
        reraise=True
    )
    async def _ainvoke_with_retry(self, **kwargs: Any) -> Any:
        """A private helper to wrap the API call with tenacity's retry logic."""
        return await self.client.embeddings.create(**kwargs)

    async def aembed_document(self, text: str, **kwargs) -> List[float]:
        """Creates an embedding for a single text using the configured API."""
        # This reuses the batch method, which is efficient as the underlying
        # API is optimized for lists, even a list of one.
        embeddings = await self.aembed_documents([text], **kwargs)
        return embeddings[0]

    async def aembed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Creates embeddings for a list of texts using the configured API."""
        if not texts:
            return []
        
        logger.info(f"Requesting embeddings for {len(texts)} documents from deployment '{self.model_name}'...")
        response = await self._ainvoke_with_retry(model=self.model_name, input=texts)
        logger.info("Successfully received embeddings.")
        return [item.embedding for item in response.data]


class HuggingFaceEmbeddings(BaseEmbeddings):
    """
    An async-compatible client for local Hugging Face embedding models.
    """

    def __init__(self, settings: Settings, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the HuggingFaceEmbeddings client."""
        self.settings = settings
        self.model_name = model_name
        logger.info(f"Loading Hugging Face model '{self.model_name}' into memory...")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Hugging Face model loaded successfully.")

    async def aembed_document(self, text: str, **kwargs) -> List[float]:
        """Creates an embedding for a single text using a local model."""
        # The synchronous, CPU-bound `encode` method is deferred to a thread pool.
        embedding = await asyncio.to_thread(self.model.encode, text)
        return embedding.tolist()

    async def aembed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Creates embeddings for a list of texts using a local model."""
        if not texts:
            return []
        
        logger.info(f"Running local embedding for {len(texts)} documents in thread pool...")
        embeddings = await asyncio.to_thread(self.model.encode, texts)
        logger.info("Local embedding complete.")
        return embeddings.tolist()