# mini_langchain/text_splitters/implementations.py
"""
Text splitter implementations for document chunking
"""

import re
from typing import List, Callable, Optional
from ..core.types import Document


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text using a hierarchy of separators.
    Tries to keep related content together by using semantic separators first.
    """
    
    def __init__(self,
                 chunk_size: int = 4000,
                 chunk_overlap: int = 200,
                 length_function: Callable[[str], int] = len,
                 is_separator_regex: bool = False,
                 separators: Optional[List[str]] = None):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.is_separator_regex = is_separator_regex
        
        # Default separators in order of preference (most semantic first)
        if separators is None:
            self.separators = [
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks  
                " ",       # Word breaks
                ""         # Character breaks (last resort)
            ]
        else:
            self.separators = separators
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        final_chunks = []
        
        # Get the current separator
        separator = separators[0] if separators else ""
        new_separators = separators[1:] if len(separators) > 1 else []
        
        # Split by current separator
        if separator == "":
            # Character-level split (last resort)
            splits = list(text)
        elif self.is_separator_regex:
            splits = re.split(separator, text)
        else:
            splits = text.split(separator)
        
        # Merge splits into chunks
        good_splits = []
        for split in splits:
            if self.length_function(split) < self.chunk_size:
                good_splits.append(split)
            else:
                # Split is too large, need to split further
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                
                # Recursively split the large chunk
                if new_separators:
                    other_info = self._split_text_recursive(split, new_separators)
                    final_chunks.extend(other_info)
                else:
                    # No more separators, force split by character
                    final_chunks.append(split)
        
        # Handle remaining splits
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks respecting size limits and overlap"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            separator_length = self.length_function(separator) if separator else 0
            
            # Check if adding this split would exceed chunk size
            potential_length = current_length + split_length
            if current_chunk:  # Add separator length if not first split
                potential_length += separator_length
            
            if current_chunk and potential_length > self.chunk_size:
                # Create chunk from current splits
                chunk_text = separator.join(current_chunk)
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                current_chunk, current_length = self._create_chunk_with_overlap(
                    current_chunk, separator, split, split_length
                )
            else:
                # Add split to current chunk
                current_chunk.append(split)
                current_length = potential_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks
    
    def _create_chunk_with_overlap(self, current_chunk: List[str], separator: str, 
                                 new_split: str, new_split_length: int) -> tuple:
        """Create new chunk with overlap from previous chunk"""
        if self.chunk_overlap == 0:
            return [new_split], new_split_length
        
        # Calculate overlap
        overlap_chunks = []
        overlap_length = 0
        separator_length = self.length_function(separator) if separator else 0
        
        # Add splits from the end until we reach overlap size
        for split in reversed(current_chunk):
            split_length = self.length_function(split)
            potential_length = overlap_length + split_length
            if overlap_chunks:  # Add separator if not first
                potential_length += separator_length
            
            if potential_length <= self.chunk_overlap:
                overlap_chunks.insert(0, split)
                overlap_length = potential_length
            else:
                break
        
        # Start new chunk with overlap + new split
        new_chunk = overlap_chunks + [new_split]
        new_length = overlap_length + new_split_length
        if overlap_chunks:  # Add separator length
            new_length += separator_length
        
        return new_chunk, new_length
    
    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        """Create Document objects from texts"""
        documents = []
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts")
        
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                # Create metadata for this chunk
                chunk_metadata = metadatas[i].copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "source_index": i
                })
                
                document = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(document)
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split existing Document objects into smaller chunks"""
        split_docs = []
        
        for doc_index, document in enumerate(documents):
            chunks = self.split_text(document.page_content)
            
            for chunk_index, chunk in enumerate(chunks):
                # Preserve original metadata and add chunk info
                chunk_metadata = document.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_index,
                    "total_chunks": len(chunks),
                    "source_document_index": doc_index
                })
                
                split_doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                split_docs.append(split_doc)
        
        return split_docs


class CharacterTextSplitter:
    """Simple character-based text splitter"""
    
    def __init__(self, 
                 separator: str = "\n\n",
                 chunk_size: int = 4000,
                 chunk_overlap: int = 200,
                 length_function: Callable[[str], int] = len):
        
        self.separator = separator
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
    
    def split_text(self, text: str) -> List[str]:
        """Split text by separator and merge into chunks"""
        splits = text.split(self.separator)
        return self._merge_splits(splits)
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits into chunks with size limits"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            if current_chunk and current_length + split_length > self.chunk_size:
                # Create chunk and start new one
                chunk_text = self.separator.join(current_chunk)
                chunks.append(chunk_text)
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text, split]
                    current_length = self.length_function(overlap_text) + split_length
                else:
                    current_chunk = [split]
                    current_length = split_length
            else:
                current_chunk.append(split)
                current_length += split_length
        
        # Add final chunk
        if current_chunk:
            chunks.append(self.separator.join(current_chunk))
        
        return chunks
    
    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[Document]:
        """Create Document objects from texts"""
        documents = []
        
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            
            for j, chunk in enumerate(chunks):
                chunk_metadata = metadatas[i].copy()
                chunk_metadata.update({
                    "chunk_index": j,
                    "total_chunks": len(chunks)
                })
                
                documents.append(Document(chunk, chunk_metadata))
        
        return documents