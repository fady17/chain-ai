"""
Vector Store Layer for the ChainForge Framework.

This module provides the core abstraction (`BaseVectorStore`) and concrete
implementations for storing and efficiently searching dense vector embeddings.
It includes options for local, in-memory search (FAISS) and a client for a
production-grade, dedicated vector database (Qdrant).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, List, Optional

import faiss
import numpy as np
from qdrant_client import AsyncQdrantClient, models

from chainforge.core import Runnable
from chainforge.embeddings import BaseEmbeddings

# Standard library logging for operational visibility.
logger = logging.getLogger(__name__)


class BaseVectorStore(Runnable[str, List[str]], ABC):
    """
    Abstract base class for all vector store implementations.

    A vector store's primary `Runnable` role in a chain is to perform retrieval.
    Therefore, its contract is to take a query string as input and return a
    list of relevant document strings as output. Methods for adding documents
    are provided as separate utility functions.
    """

    def __init__(self, embeddings: BaseEmbeddings):
        """
        Args:
            embeddings: An embedding model instance that will be used to convert
                        texts into vectors for storage and search.
        """
        if not isinstance(embeddings, BaseEmbeddings):
            raise TypeError("A valid BaseEmbeddings instance is required.")
        self.embeddings = embeddings

    @abstractmethod
    async def aadd_documents(self, texts: List[str], **kwargs) -> None:
        """
        Embeds a list of texts and adds them to the vector store.
        This is the primary method for populating the store's knowledge base.
        """
        raise NotImplementedError

    @abstractmethod
    async def asimilarity_search(self, query: str, k: int = 4, **kwargs) -> List[str]:
        """
        Finds the k most similar document texts to a given query string.
        """
        raise NotImplementedError

    async def ainvoke(self, input: str, **kwargs: Any) -> List[str]:
        """
        Performs a similarity search using the input string as the query.

        This method makes the vector store directly chainable, where the output
        of a previous step (if it's a string) can be piped directly as a query.
        """
        if not isinstance(input, str):
            raise TypeError("Vector store input must be a single query string.")
        
        k = kwargs.get('k', 4)
        return await self.asimilarity_search(input, k=k, **kwargs)

    async def astream(self, input: str, **kwargs: Any) -> AsyncIterator[List[str]]:
        """
        Yields the single list of search results.
        
        Vector search is an atomic operation. This method fulfills the `Runnable`
        contract by yielding the complete list of results as a single item.
        """
        result = await self.ainvoke(input, **kwargs)
        yield result


class FAISSVectorStore(BaseVectorStore):
    """
    A high-performance vector store for local, in-memory search using FAISS.
    
    Ideal for rapid prototyping, testing, or applications with moderately-sized
    datasets where persistence is not required.
    """
    def __init__(self, embeddings: BaseEmbeddings):
        super().__init__(embeddings)
        self._texts: List[str] = []
        self._index: faiss.Index | None = None

    async def aadd_documents(self, texts: List[str], **kwargs) -> None:
        if not texts:
            return
            
        embeddings_list = await self.embeddings.aembed_documents(texts, **kwargs)
        vectors = np.array(embeddings_list, dtype=np.float32)

        if self._index is None:
            dimension = vectors.shape[1]
            logger.info(f"Initializing FAISS IndexFlatL2 with dimension {dimension}.")
            self._index = faiss.IndexFlatL2(dimension)
        
        assert self._index is not None, "FAISS index was not initialized."

        # Defer the CPU-bound `add` operation to a thread pool to avoid blocking.
        await asyncio.to_thread(lambda: self._index.add(vectors)) # type: ignore
        self._texts.extend(texts)

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs) -> List[str]:
        assert self._index is not None, "Cannot search on an uninitialized FAISS store."

        query_embedding = await self.embeddings.aembed_document(query, **kwargs)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Defer the CPU-bound `search` operation to a thread pool.
        _, indices = await asyncio.to_thread(
            lambda: self._index.search(query_vector, min(k, len(self._texts))) # type: ignore
        )
        
        return [self._texts[i] for i in indices[0] if i != -1]


class QdrantVectorStore(BaseVectorStore):
    """
    A robust, production-grade vector store client for a Qdrant instance.
    
    This is the recommended choice for production applications requiring data
    persistence, scalability, and advanced filtering capabilities. It connects
    to a Qdrant service which can be self-hosted or a managed cloud instance.
    """
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        collection_name: str,
        url: Optional[str] = "localhost",
        port: Optional[int] = 6333,
        api_key: Optional[str] = None,
        in_memory: bool = False,
    ):
        super().__init__(embeddings)
        self.collection_name = collection_name
        
        # Initialize the async client for Qdrant.
        if in_memory:
            logger.info("Initializing Qdrant client in-memory.")
            self.client = AsyncQdrantClient(":memory:")
        else:
            logger.info(f"Initializing Qdrant client for server at {url}:{port}.")
            self.client = AsyncQdrantClient(url=url, port=port, api_key=api_key)
        
        self._collection_initialized = False

    async def _initialize_collection(self, vector_size: int):
        """Private helper to create the Qdrant collection on first use."""
        if self._collection_initialized:
            return
        
        try:
            # Check if the collection already exists in the Qdrant instance.
            await self.client.get_collection(collection_name=self.collection_name)
            logger.info(f"Connected to existing Qdrant collection: '{self.collection_name}'.")
        except Exception:
            # If it doesn't exist, create it.
            logger.info(f"Creating new Qdrant collection: '{self.collection_name}' with vector size {vector_size}.")
            await self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        self._collection_initialized = True

    async def aadd_documents(self, texts: List[str], **kwargs) -> None:
        if not texts:
            return
            
        vectors = await self.embeddings.aembed_documents(texts, **kwargs)
        
        # Ensure the collection exists and is configured for the correct vector size.
        # This is a lazy initialization done on the first data addition.
        await self._initialize_collection(vector_size=len(vectors[0]))
        
        # In Qdrant, a "Point" consists of a unique ID, its vector, and a
        # "payload" which can store arbitrary metadata, like the original text.
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text}
            ) for text, vector in zip(texts, vectors)
        ]

        # `upsert` is an efficient and idempotent operation to add or update points.
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True # Wait for the operation to be indexed.
        )
        logger.info(f"Upserted {len(points)} points to Qdrant collection '{self.collection_name}'.")

    async def asimilarity_search(self, query: str, k: int = 4, **kwargs) -> List[str]:
        query_vector = await self.embeddings.aembed_document(query, **kwargs)
        
        # The core search operation using the async client.
        search_results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
        )
        
        # The results are `ScoredPoint` objects; we extract the original text
        # from the payload of each point.
        return [result.payload["text"] for result in search_results] # type: ignore