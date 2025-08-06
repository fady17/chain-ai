from __future__ import annotations
import re
from typing import List, Optional, Callable, Any
from minichain.text_splitters.base import BaseTextSplitter


class RecursiveCharacterTextSplitter(BaseTextSplitter):
    """
    A character-based text splitter that recursively splits text and then
    merges it back together.

    This is a direct, simplified adaptation of LangChain's battle-tested
    splitter, inheriting its robust `_merge_splits` logic from our Base class.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        # Explicitly define ALL possible parent arguments for robust subclassing
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len,
        add_start_index: bool = False,
        strip_whitespace: bool = True,
    ) -> None:
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            keep_separator=keep_separator,
            add_start_index=add_start_index,
            strip_whitespace=strip_whitespace,
        )
        self._separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """The core recursive splitting logic."""
        final_chunks: List[str] = []
        
        # Find the highest-priority separator that exists in the text.
        separator = separators[-1] # Fallback to the last separator
        for s in separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break
        
        # Split the text by the best separator.
        # Use re.split to handle keeping the separator correctly.
        if separator:
            # The parentheses in the pattern keep the delimiters in the result.
            splits = re.split(f"({re.escape(separator)})", text)
            # Group the separator with the text before it.
            # e.g., ["text1", "\n\n", "text2"] -> ["text1\n\n", "text2"]
            grouped_splits = []
            for i in range(0, len(splits), 2):
                chunk = splits[i]
                if i + 1 < len(splits):
                    chunk += splits[i+1]
                if chunk:
                    grouped_splits.append(chunk)
            splits = grouped_splits
        else:
            splits = list(text)
        
        # Now, recursively process any split that is still too large
        good_splits: List[str] = []
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    # Merge the "good" splits before handling the oversized one
                    merged = self._merge_splits(good_splits, "") # Don't add extra separators
                    final_chunks.extend(merged)
                    good_splits = []
                
                # Recurse on the oversized chunk
                next_separators = separators[separators.index(separator) + 1:]
                other_chunks = self._split_text(s, next_separators)
                final_chunks.extend(other_chunks)
        
        # Merge any final remaining good splits
        if good_splits:
            merged = self._merge_splits(good_splits, "")
            final_chunks.extend(merged)
            
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """The main public method that kicks off the splitting process."""
        # The `_split_text` method breaks the text into small pieces.
        # The `_merge_splits` from the base class does the final, robust chunking.
        splits = self._split_text(text, self._separators)
        return self._merge_splits(splits, "")
