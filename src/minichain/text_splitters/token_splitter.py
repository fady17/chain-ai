# src/minichain/text_splitters/token_splitter.py
"""
A more advanced text splitter that splits based on token counts,
making it language-agnostic and more precise.
"""
import tiktoken
from typing import List, Optional
from ..core.types import Document

class TokenTextSplitter:
    """
    Splits text based on token count using a tokenizer (tiktoken).
    This is generally more reliable across different languages than
    character-based splitting.
    """

    def __init__(
        self,
        model_name: str = "gpt-4", # Model name to infer encoding
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Chunk overlap ({chunk_overlap}) cannot be larger than chunk size ({chunk_size})."
            )
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Get the encoding for the specified model
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback for models not in tiktoken's registry
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> List[str]:
        """Splits a single text into chunks based on token count."""
        if not text:
            return []
            
        # First, encode the entire text into tokens
        tokens = self.tokenizer.encode(text)
        
        if not tokens:
            return []

        chunks = []
        start_index = 0
        while start_index < len(tokens):
            # Determine the end of the chunk
            end_index = min(start_index + self.chunk_size, len(tokens))
            
            # Decode the tokens for this chunk back into text
            chunk_tokens = tokens[start_index:end_index]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move the start index for the next chunk, accounting for overlap
            # We subtract the overlap from the end of the last chunk
            start_index += (self.chunk_size - self.chunk_overlap)

        return chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create Document objects from a list of texts."""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts.")
        
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