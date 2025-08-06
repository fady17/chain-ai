"""
Defines the abstract base class for all text splitters in the Mini-Chain framework.
This version is adapted from LangChain's robust implementation to provide
powerful helper methods like _merge_splits.
"""
from abc import ABC, abstractmethod
import copy
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
)
from minichain.core.types import Document

class BaseTextSplitter(ABC):
    """
    Base interface for splitting text into chunks. It also provides the
    core _merge_splits logic used by subclasses.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._strip_whitespace = strip_whitespace

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Abstract method to split text into chunks."""
        pass

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            start_index = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    # Find the start index of the chunk in the original text.
                    try:
                        start_index = text.index(chunk, start_index)
                        metadata["start_index"] = start_index
                        start_index += len(chunk) # Move index for next search
                    except ValueError:
                        # Chunk not found, which can happen with complex splitting.
                        # We'll just not add the start_index in this case.
                        pass
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Splits a list of Documents into smaller documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        return text if text else None

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        """The robust merging logic. This is the core engine."""
        separator_len = self._length_function(separator)
        docs = []
        current_doc: List[str] = []
        total = 0

        for s in splits:
            _len = self._length_function(s)
            if total + _len + (separator_len if current_doc else 0) > self._chunk_size:
                if total > self._chunk_size:
                    # This warning is helpful for debugging separators.
                    print(
                        f"Warning: Created a chunk of size {total}, which is "
                        f"longer than the specified {self._chunk_size}"
                    )
                if current_doc:
                    doc = self._join_docs(current_doc, separator)
                    if doc:
                        docs.append(doc)
                    # Slide the window forward to handle overlap.
                    while total > self._chunk_overlap:
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]

            current_doc.append(s)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        
        doc = self._join_docs(current_doc, separator)
        if doc:
            docs.append(doc)
        
        return docs
    # """Abstract base class for text splitters."""

    # def __init__(self, chunk_size: int, chunk_overlap: int):
    #     """
    #     Initializes the base splitter with common parameters.

    #     Args:
    #         chunk_size: The maximum size of a chunk.
    #         chunk_overlap: The overlap between consecutive chunks.
    #     """
    #     if chunk_overlap > chunk_size:
    #         raise ValueError(
    #             f"Chunk overlap ({chunk_overlap}) cannot be larger than chunk size ({chunk_size})."
    #         )
    #     self.chunk_size = chunk_size
    #     self.chunk_overlap = chunk_overlap

    # @abstractmethod
    # def split_text(self, text: str) -> List[str]:
    #     """Abstract method to split a single text into chunks."""
    #     pass

    # def create_documents(
    #     self, texts: List[str], metadatas: Optional[List[dict]] = None
    # ) -> List[Document]:
    #     """
    #     Processes a list of texts, splitting each and creating Document objects.
    #     This is a generic implementation that can be used by most subclasses.
    #     """
    #     metadatas = metadatas or [{}] * len(texts)
    #     if len(metadatas) != len(texts):
    #         raise ValueError("The number of metadatas must match the number of texts.")
        
    #     documents = []
    #     for i, text in enumerate(texts):
    #         chunks = self.split_text(text)
    #         for j, chunk in enumerate(chunks):
    #             chunk_metadata = metadatas[i].copy()
    #             chunk_metadata.update({
    #                 "chunk_index": j,
    #                 "total_chunks": len(chunks),
    #                 "source_index": i
    #             })
    #             documents.append(Document(page_content=chunk, metadata=chunk_metadata))
    #     return documents

    # def split_documents(self, documents: List[Document]) -> List[Document]:
    #     """
    #     Takes a list of existing Document objects and splits them into smaller
    #     documents, preserving metadata.
    #     """
    #     split_docs = []
    #     for doc_index, document in enumerate(documents):
    #         chunks = self.split_text(document.page_content)
    #         for chunk_index, chunk in enumerate(chunks):
    #             new_metadata = document.metadata.copy()
    #             new_metadata.update({
    #                 "chunk_index": chunk_index,
    #                 "total_chunks": len(chunks),
    #                 "source_document_index": doc_index
    #             })
    #             split_doc = Document(page_content=chunk, metadata=new_metadata)
    #             split_docs.append(split_doc)
    #     return split_docs