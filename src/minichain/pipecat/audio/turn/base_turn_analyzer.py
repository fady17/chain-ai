# src/minichain/pipecat/audio/turn/base_turn_analyzer.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple

# We will define this later, for now it's a placeholder.
# from minichain.pipecat.metrics.metrics import MetricsData
MetricsData = object # Placeholder

class EndOfTurnState(Enum):
    """State enumeration for end-of-turn analysis results."""
    COMPLETE = 1
    INCOMPLETE = 2


class BaseTurnAnalyzer(ABC):
    """Abstract base class for analyzing user end of turn."""
    def __init__(self, *, sample_rate: Optional[int] = None):
        self._init_sample_rate = sample_rate
        self._sample_rate = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def set_sample_rate(self, sample_rate: int):
        self._sample_rate = self._init_sample_rate or sample_rate

    @property
    @abstractmethod
    def speech_triggered(self) -> bool:
        """Determines if speech has been detected."""
        pass

    @property
    @abstractmethod
    def params(self):
        """Get the current turn analyzer parameters."""
        pass

    @abstractmethod
    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Appends audio data for analysis."""
        pass

    @abstractmethod
    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        """Analyzes if an end of turn has occurred based on the audio input."""
        pass

    @abstractmethod
    def clear(self):
        """Reset the turn analyzer to its initial state."""
        pass