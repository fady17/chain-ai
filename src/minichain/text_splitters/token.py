import tiktoken
from typing import List, Any
from minichain.text_splitters.base import BaseTextSplitter

class TokenTextSplitter(BaseTextSplitter):
    """
    A text splitter that operates on tokens. This is the recommended
    splitter for use with modern language models as it aligns with their
    context windows.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        **kwargs: Any,
    ):
        """
        Initializes the TokenTextSplitter.

        Args:
            model_name (str): The name of the model to select the tokenizer.
            **kwargs: Arguments passed to the BaseTextSplitter parent class,
                      such as chunk_size and chunk_overlap.
        """
     
        super().__init__(**kwargs)
        self.model_name = model_name
        
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str) -> List[str]:
        """Encodes a text to tokens and splits them into text chunks."""
        if not text or text.isspace():
            return []
            
        tokens = self.tokenizer.encode(text)
        
        chunks: List[str] = []
        start_index = 0
        while start_index < len(tokens):
            end_index = min(start_index + self._chunk_size, len(tokens))
            chunk_tokens = tokens[start_index:end_index]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append(chunk_text)
            
            # Use the overlap from the base class for the next window
            start_index += (self._chunk_size - self._chunk_overlap)

        return chunks
# # src/minichain/text_splitters/token.py
# """
# Provides a token-based text splitter for the Mini-Chain framework.
# This is the recommended splitter for use with modern language models.
# """
# import tiktoken
# from typing import List
# from .base import BaseTextSplitter

# class TokenTextSplitter(BaseTextSplitter):
#     """
#     Splits text into chunks of a specified token size using a tokenizer.

#     This class leverages the `tiktoken` library to encode text into tokens and
#     then splits the token list into chunks. This method is highly effective
#     for ensuring that chunks do not exceed the context window of a downstream
#     language model and often produces more semantically coherent chunks than
#     character-based methods.
#     """

#     def __init__(
#         self,
#         model_name: str = "gpt-4",
#         chunk_size: int = 500,
#         chunk_overlap: int = 50,
#     ):
#         """
#         Initializes the TokenTextSplitter.

#         Args:
#             model_name (str): The name of the model to select the appropriate
#                 tokenizer. Helps ensure token counts are accurate for that model.
#             chunk_size (int): The maximum number of tokens per chunk.
#             chunk_overlap (int): The number of tokens to overlap between chunks.
#         """
#         super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         self.model_name = model_name
        
#         try:
#             self.tokenizer = tiktoken.encoding_for_model(self.model_name)
#         except KeyError:
#             # Fallback to a common default encoding if the model name is unknown
#             self.tokenizer = tiktoken.get_encoding("cl100k_base")

#     def split_text(self, text: str) -> List[str]:
#         """
#         Encodes a text to tokens and splits them into text chunks.

#         This method includes a check to handle empty or whitespace-only input
#         gracefully, preventing the creation of empty documents.

#         Args:
#             text (str): The text to be split.

#         Returns:
#             List[str]: A list of text chunks.
#         """
#         # Handle empty or whitespace-only input to avoid creating empty chunks.
#         if not text or text.isspace():
#             return []
            
#         tokens = self.tokenizer.encode(text)
        
#         if not tokens:
#             return []

#         chunks: List[str] = []
#         start_index = 0
#         while start_index < len(tokens):
#             end_index = min(start_index + self.chunk_size, len(tokens))
            
#             chunk_tokens = tokens[start_index:end_index]
#             chunk_text = self.tokenizer.decode(chunk_tokens)
            
#             # Ensure the decoded chunk itself isn't just whitespace before adding.
#             if chunk_text.strip():
#                 chunks.append(chunk_text)
            
#             # Advance the window for the next chunk
#             start_index += (self.chunk_size - self.chunk_overlap)

#         return chunks
