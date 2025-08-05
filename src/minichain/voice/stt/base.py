# src/minichain/voice/stt/base.py
from abc import ABC, abstractmethod
from typing import Iterator, AsyncIterator

class BaseSTTModel(ABC):
    """Abstract base class for all Speech-to-Text models."""

    @abstractmethod
    def stream(self, audio_chunk_iterator: Iterator[bytes]) -> Iterator[str]:
        """
        Transcribes a stream of audio chunks into a stream of text.
        
        Args:
            audio_chunk_iterator: An iterator that yields raw audio chunks (bytes).
            
        Returns:
            An iterator that yields transcribed text segments.
        """
        pass

    async def astream(self, audio_chunk_iterator: AsyncIterator[bytes]) -> AsyncIterator[str]:
        """
        Asynchronous version of the stream method.
        
        Raises:
            NotImplementedError: If the model does not support async streaming.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support asynchronous streaming."
        )