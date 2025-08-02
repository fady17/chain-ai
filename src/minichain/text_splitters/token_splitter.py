# # src/minichain/text_splitters/token_splitter.py
# """
# Provides a token-based text splitter for the Mini-Chain framework.

# This splitter is generally recommended over character-based splitters as it
# aligns more closely with how language models process text. By splitting based
# on tokens, it can more accurately preserve semantic units across various
# languages and produce chunks of a predictable token size, which is crucial for
# managing LLM context windows.
# """
# import tiktoken
# from typing import List, Optional
# from ..core.types import Document

# class TokenTextSplitter:
#     """
#     Splits text into chunks of a specified token size using a tokenizer.

#     This class leverages the `tiktoken` library to encode text into tokens and
#     then splits the token list into chunks. This method is highly effective
#     for ensuring that chunks do not exceed the context window of a downstream
#     language model.
#     """

#     def __init__(
#         self,
#         model_name: str = "gpt-4", # Used to determine the correct tokenizer
#         chunk_size: int = 500,
#         chunk_overlap: int = 50,
#     ):
#         """
#         Initializes the TokenTextSplitter.

#         Args:
#             model_name (str): The name of the model that will eventually process
#                 the text. This is used to select the appropriate tokenizer
#                 from `tiktoken` to ensure token counts are accurate.
#             chunk_size (int): The maximum number of tokens allowed in a chunk.
#             chunk_overlap (int): The number of tokens from the end of one chunk
#                 to include at the beginning of the next, to maintain context.
#         """
#         if chunk_overlap > chunk_size:
#             raise ValueError(
#                 f"Chunk overlap ({chunk_overlap}) cannot be larger than chunk size ({chunk_size})."
#             )
#         self.model_name = model_name
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
        
#         # Select the appropriate tokenizer. Falls back to a common default.
#         try:
#             self.tokenizer = tiktoken.encoding_for_model(self.model_name)
#         except KeyError:
#             self.tokenizer = tiktoken.get_encoding("cl100k_base")

#     def split_text(self, text: str) -> List[str]:
#         """
#         Encodes a text to tokens and splits them into text chunks.

#         Args:
#             text (str): The text to be split.

#         Returns:
#             List[str]: A list of text chunks, each corresponding to a chunk
#                        of tokens.
#         """
#         if not text:
#             return []
            
#         tokens = self.tokenizer.encode(text)
        
#         if not tokens:
#             return []

#         chunks = []
#         start_index = 0
#         while start_index < len(tokens):
#             end_index = min(start_index + self.chunk_size, len(tokens))
            
#             chunk_tokens = tokens[start_index:end_index]
#             chunk_text = self.tokenizer.decode(chunk_tokens)
#             chunks.append(chunk_text)
            
#             # Advance the window, accounting for overlap
#             start_index += (self.chunk_size - self.chunk_overlap)

#         return chunks

#     def create_documents(
#         self, texts: List[str], metadatas: Optional[List[dict]] = None
#     ) -> List[Document]:
#         """
#         Processes a list of texts, splitting each and creating Document objects.

#         This is a convenience method that combines splitting and Document
#         creation. It ensures that metadata from the original documents is
#         preserved and augmented with chunk-specific information.

#         Args:
#             texts (List[str]): The list of original texts to process.
#             metadatas (Optional[List[dict]]): A list of metadata dictionaries,
#                 one for each text.

#         Returns:
#             List[Document]: A list of new Document objects, one for each chunk.
#         """
#         if metadatas is None:
#             metadatas = [{}] * len(texts)
#         elif len(metadatas) != len(texts):
#             raise ValueError("The number of metadatas must match the number of texts.")
        
#         documents = []
#         for i, text in enumerate(texts):
#             chunks = self.split_text(text)
#             for j, chunk in enumerate(chunks):
#                 # Create a copy of the original metadata to avoid mutation
#                 chunk_metadata = metadatas[i].copy()
#                 chunk_metadata.update({
#                     "chunk_index": j,
#                     "total_chunks": len(chunks),
#                     "source_index": i
#                 })
#                 # Instantiate Document objects using Pydantic's keyword arguments
#                 # for clarity and type safety.
#                 documents.append(Document(page_content=chunk, metadata=chunk_metadata))
#         return documents

#     def split_documents(self, documents: List[Document]) -> List[Document]:
#         """
#         Takes a list of existing Document objects and splits them into smaller
#         documents.

#         This method is useful when you have already loaded documents from a
#         source and now need to chunk them for a vector store. It preserves
#         the original metadata in each new chunk.

#         Args:
#             documents (List[Document]): A list of Document objects to split.

#         Returns:
#             List[Document]: A new list of smaller Document objects.
#         """
#         split_docs = []
#         for doc_index, document in enumerate(documents):
#             chunks = self.split_text(document.page_content)
#             for chunk_index, chunk in enumerate(chunks):
#                 # Inherit metadata from the parent document
#                 new_metadata = document.metadata.copy()
#                 new_metadata.update({
#                     "chunk_index": chunk_index,
#                     "total_chunks": len(chunks),
#                     "source_document_index": doc_index
#                 })
#                 # Instantiate new Document objects using keyword arguments.
#                 split_doc = Document(page_content=chunk, metadata=new_metadata)
#                 split_docs.append(split_doc)
#         return split_docs
# # # src/minichain/text_splitters/token_splitter.py
# # """
# # A more advanced text splitter that splits based on token counts,
# # making it language-agnostic and more precise.
# # """
# # import tiktoken
# # from typing import List, Optional
# # from ..core.types import Document

# # class TokenTextSplitter:
# #     """
# #     Splits text based on token count using a tokenizer (tiktoken).
# #     This is generally more reliable across different languages than
# #     character-based splitting.
# #     """

# #     def __init__(
# #         self,
# #         model_name: str = "gpt-4", # Model name to infer encoding
# #         chunk_size: int = 500,
# #         chunk_overlap: int = 50,
# #     ):
# #         if chunk_overlap > chunk_size:
# #             raise ValueError(
# #                 f"Chunk overlap ({chunk_overlap}) cannot be larger than chunk size ({chunk_size})."
# #             )
# #         self.model_name = model_name
# #         self.chunk_size = chunk_size
# #         self.chunk_overlap = chunk_overlap
        
# #         # Get the encoding for the specified model
# #         try:
# #             self.tokenizer = tiktoken.encoding_for_model(self.model_name)
# #         except KeyError:
# #             # Fallback for models not in tiktoken's registry
# #             self.tokenizer = tiktoken.get_encoding("cl100k_base")

# #     def split_text(self, text: str) -> List[str]:
# #         """Splits a single text into chunks based on token count."""
# #         if not text:
# #             return []
            
# #         # First, encode the entire text into tokens
# #         tokens = self.tokenizer.encode(text)
        
# #         if not tokens:
# #             return []

# #         chunks = []
# #         start_index = 0
# #         while start_index < len(tokens):
# #             # Determine the end of the chunk
# #             end_index = min(start_index + self.chunk_size, len(tokens))
            
# #             # Decode the tokens for this chunk back into text
# #             chunk_tokens = tokens[start_index:end_index]
# #             chunk_text = self.tokenizer.decode(chunk_tokens)
# #             chunks.append(chunk_text)
            
# #             # Move the start index for the next chunk, accounting for overlap
# #             # We subtract the overlap from the end of the last chunk
# #             start_index += (self.chunk_size - self.chunk_overlap)

# #         return chunks

# #     def create_documents(
# #         self, texts: List[str], metadatas: Optional[List[dict]] = None
# #     ) -> List[Document]:
# #         """Create Document objects from a list of texts."""
# #         if metadatas is None:
# #             metadatas = [{}] * len(texts)
# #         elif len(metadatas) != len(texts):
# #             raise ValueError("Number of metadatas must match number of texts.")
        
# #         documents = []
# #         for i, text in enumerate(texts):
# #             chunks = self.split_text(text)
# #             for j, chunk in enumerate(chunks):
# #                 chunk_metadata = metadatas[i].copy()
# #                 chunk_metadata.update({
# #                     "chunk_index": j,
# #                     "total_chunks": len(chunks),
# #                     "source_index": i
# #                 })
# #                 documents.append(Document(page_content=chunk, metadata=chunk_metadata))
# #         return documents