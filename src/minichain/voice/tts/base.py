# src/minichain/voice/tts/base.py
from abc import ABC, abstractmethod
from typing import Iterator, AsyncIterator

class BaseTTSModel(ABC):
    """Abstract base class for all Text-to-Speech models."""

    @abstractmethod
    def stream(self, text_chunk_iterator: Iterator[str]) -> Iterator[bytes]:
        """
        Synthesizes a stream of text chunks into a stream of audio.

        Args:
            text_chunk_iterator: An iterator that yields text chunks to be synthesized.

        Returns:
            An iterator that yields raw audio chunks (bytes) of the synthesized speech.
        """
        pass
    
    async def astream(self, text_chunk_iterator: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """
        Asynchronous version of the stream method.
        
        Raises:
            NotImplementedError: If the model does not support async streaming.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support asynchronous streaming."
        )