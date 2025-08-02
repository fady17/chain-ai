# src/minichain/text_splitters/base.py
"""
Defines the abstract base class for all text splitters in the Mini-Chain framework.
This ensures a consistent interface for splitting text and creating Document objects.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.types import Document

class BaseTextSplitter(ABC):
    """Abstract base class for text splitters."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Initializes the base splitter with common parameters.

        Args:
            chunk_size: The maximum size of a chunk.
            chunk_overlap: The overlap between consecutive chunks.
        """
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) cannot be larger than chunk size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Abstract method to split a single text into chunks."""
        pass

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """
        Processes a list of texts, splitting each and creating Document objects.
        This is a generic implementation that can be used by most subclasses.
        """
        metadatas = metadatas or [{}] * len(texts)
        if len(metadatas) != len(texts):
            raise ValueError("The number of metadatas must match the number of texts.")
        
        documents = []
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            for j, chunk in enumerate(chunks):
                chunk_metadata = metadatas[i].copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "source_index": i
                })
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Takes a list of existing Document objects and splits them into smaller
        documents, preserving metadata.
        """
        split_docs = []
        for doc_index, document in enumerate(documents):
            chunks = self.split_text(document.page_content)
            for chunk_index, chunk in enumerate(chunks):
                new_metadata = document.metadata.copy()
                new_metadata.update({
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "source_document_index": doc_index
                })
                split_doc = Document(page_content=chunk, metadata=new_metadata)
                split_docs.append(split_doc)
        return split_docs