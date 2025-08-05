# src/minichain/pipecat/audio/vad/vad_analyzer.py
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# We will create this utils file
from minichain.pipecat.audio.utils import calculate_audio_volume_rms, exp_smoothing

class VADState(Enum):
    """
    Voice Activity Detection states, mirroring Pipecat's robust model.
    """
    QUIET = 1
    STARTING = 2
    SPEAKING = 3
    STOPPING = 4

class VADParams(BaseModel):
    """Configuration parameters for Voice Activity Detection."""
    start_secs: float = Field(default=0.2, description="Seconds of speech to start a turn.")
    stop_secs: float = Field(default=0.8, description="Seconds of silence to end a turn.")
    min_volume: float = Field(default=0.01, ge=0.0, le=1.0, description="Minimum smoothed audio volume to be considered speech.")

class BaseVADAnalyzer(ABC):
    """
    Abstract base class for a stateful Voice Activity Detection analyzer,
    inspired by Pipecat's production-grade design.
    """
    def __init__(self, sample_rate: int, frames_per_chunk: int, params: Optional[VADParams] = None):
        self._sample_rate = sample_rate
        self._frames_per_chunk = frames_per_chunk
        self._params = params or VADParams()

        # Internal state
        self._vad_buffer = b""
        self._state: VADState = VADState.QUIET
        self._smoothing_factor = 0.2
        self._prev_volume = 0.0
        
        # Convert time-based params to frame-based counts
        self._recalculate_params()

    def _recalculate_params(self):
        """Converts user-friendly seconds into internal frame counts."""
        chunks_per_sec = self._sample_rate / self._frames_per_chunk
        self._start_chunks = int(self._params.start_secs * chunks_per_sec)
        self._stop_chunks = int(self._params.stop_secs * chunks_per_sec)
        self._starting_count = 0
        self._stopping_count = 0

    def set_params(self, params: VADParams):
        """Allows for updating VAD parameters at runtime."""
        self._params = params
        self._recalculate_params()
        
    @property
    def state(self) -> VADState:
        return self._state

    @abstractmethod
    def get_voice_confidence(self, audio_chunk: bytes) -> float:
        """
        Subclasses must implement this method to return a speech probability
        (0.0 to 1.0) for a given audio chunk.
        """
        pass

    def analyze_chunk(self, audio_chunk: bytes) -> Optional[VADState]:
        """
        Analyzes a chunk of audio, updates the internal state, and returns the
        new state ONLY if it's a definitive change (QUIET or SPEAKING).
        """
        old_state = self._state
        
        confidence = self.get_voice_confidence(audio_chunk)
        
        raw_volume = calculate_audio_volume_rms(audio_chunk)
        volume = exp_smoothing(raw_volume, self._prev_volume, self._smoothing_factor)
        self._prev_volume = volume
        
        # Pipecat uses a confidence threshold from the VAD model. WebRTC VAD is binary,
        # so we'll treat its "True" as confidence 1.0 and "False" as 0.0.
        is_considered_speech = confidence > 0.5 and volume >= self._params.min_volume

        if is_considered_speech:
            if self._state == VADState.QUIET:
                self._state = VADState.STARTING
                self._starting_count = 1
            elif self._state == VADState.STARTING:
                self._starting_count += 1
            elif self._state == VADState.STOPPING:
                self._state = VADState.SPEAKING # Recovered from a pause
                self._stopping_count = 0
        else: # Not speech
            if self._state == VADState.STARTING:
                self._state = VADState.QUIET # False start
                self._starting_count = 0
            elif self._state == VADState.SPEAKING:
                self._state = VADState.STOPPING
                self._stopping_count = 1
            elif self._state == VADState.STOPPING:
                self._stopping_count += 1
        
        # Check for definitive state transitions
        new_definitive_state: Optional[VADState] = None
        if self._state == VADState.STARTING and self._starting_count >= self._start_chunks:
            self._state = VADState.SPEAKING
            self._starting_count = 0
        
        if self._state == VADState.STOPPING and self._stopping_count >= self._stop_chunks:
            self._state = VADState.QUIET
            self._stopping_count = 0

        # Return the new state only if it changed from the last definitive state
        if self._state != old_state and self._state in [VADState.QUIET, VADState.SPEAKING]:
             return self._state
        
        return None