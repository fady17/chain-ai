# src/minichain/text_splitters/character.py
"""
Provides a character-based text splitter for the Mini-Chain framework.
This splitter operates directly on strings and characters.
"""
import re
from typing import List, Callable, Optional
from .base import BaseTextSplitter

class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    Splits text by recursively trying a sequence of separators.

    This splitter attempts to find the most semantically relevant separator
    (like paragraph breaks) first. If a resulting chunk is still too large,
    it moves to the next separator in the list (like line breaks), and so on.
    This helps keep related pieces of text together in the same chunk.
    """
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 100,
                 length_function: Callable[[str], int] = len,
                 separators: Optional[List[str]] = None):
        """
        Initializes the RecursiveCharacterTextSplitter.

        Args:
            chunk_size (int): The maximum size of a chunk (measured by `length_function`).
            chunk_overlap (int): The overlap between consecutive chunks.
            length_function (Callable): Function to measure text length. Defaults to `len`.
            separators (Optional[List[str]]): A list of strings to split on,
                in order of priority.
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Splits the text recursively until all chunks are under the chunk_size.
        
        Args:
            text (str): The input text.

        Returns:
            List[str]: A list of text chunks.
        """
        # Handle empty or whitespace-only input to avoid creating empty chunks.
        if not text or text.isspace():
            return []
            
        final_chunks: List[str] = []
        
        # Start with the highest-priority separator
        separator = self.separators[0]
        # Find the best separator that actually exists in the text
        for s in self.separators:
            # An empty separator ("") is our last resort
            if s == "" or re.search(s, text):
                separator = s
                break
        
        # If the text is already small enough, no need to split
        if self.length_function(text) < self.chunk_size:
            return [text]

        # Split the text by the chosen separator
        splits = text.split(separator)
        
        # Process the resulting splits
        good_chunks: List[str] = []
        for chunk in splits:
            if self.length_function(chunk) < self.chunk_size:
                good_chunks.append(chunk)
            else:
                # If we have some "good" chunks, merge them before handling the large one
                if good_chunks:
                    merged = self._merge_splits(good_chunks, separator)
                    final_chunks.extend(merged)
                    good_chunks = []
                
                # This chunk is too big, so we recursively call split_text on it
                # with the next set of lower-priority separators.
                next_separators = self.separators[self.separators.index(separator) + 1:]
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=self.length_function,
                    separators=next_separators
                )
                final_chunks.extend(recursive_splitter.split_text(chunk))
        
        # Merge any leftover "good" chunks at the end
        if good_chunks:
            merged = self._merge_splits(good_chunks, separator)
            final_chunks.extend(merged)

        # Filter out any empty or whitespace-only strings that might have been created
        return [c for c in final_chunks if c.strip()]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """A simple greedy algorithm to merge small splits into larger chunks."""
        merged: List[str] = []
        current_chunk: List[str] = []
        current_length = 0
        separator_len = self.length_function(separator)

        for split in splits:
            # Filter out empty splits immediately
            if not split.strip():
                continue

            split_len = self.length_function(split)
            # Check if adding the next split would exceed the chunk size
            if current_length + split_len + (separator_len if current_chunk else 0) > self.chunk_size:
                if current_chunk:
                    merged.append(separator.join(current_chunk))
                current_chunk = [split]
                current_length = split_len
            else:
                current_chunk.append(split)
                current_length += split_len + (separator_len if len(current_chunk) > 1 else 0)
        
        if current_chunk:
            merged.append(separator.join(current_chunk))
        
        return merged

