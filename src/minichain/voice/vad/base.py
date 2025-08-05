# src/minichain/voice/vad/base.py
from abc import ABC, abstractmethod

class BaseVAD(ABC):
    """Abstract base class for Voice Activity Detection components."""

    @abstractmethod
    def is_speech(self, chunk: bytes) -> bool:
        """
        Determines if a given audio chunk contains speech.

        Args:
            chunk: A raw audio chunk in bytes.

        Returns:
            True if the chunk contains speech, False otherwise.
        """
        pass