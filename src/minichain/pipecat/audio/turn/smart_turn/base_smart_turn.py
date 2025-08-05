# src/minichain/pipecat/audio/turn/smart_turn/base_smart_turn.py
import time
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from ..base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
# from minichain.pipecat.metrics.metrics import MetricsData, SmartTurnMetricsData # Placeholder
MetricsData = object # Placeholder

class SmartTurnParams(BaseModel):
    """Configuration parameters for smart turn analysis."""
    stop_secs: float = Field(default=3.0, description="Maximum silence duration in seconds before ending turn.")
    pre_speech_ms: float = Field(default=0.0, description="Milliseconds of audio to include before speech starts.")
    max_duration_secs: float = Field(default=8.0, description="Maximum duration in seconds for audio segments.")

class SmartTurnTimeoutException(Exception):
    """Exception raised when smart turn analysis times out."""
    pass

class BaseSmartTurn(BaseTurnAnalyzer):
    """Base class for smart turn analyzers using ML models."""
    def __init__(self, *, sample_rate: Optional[int] = None, params: Optional[SmartTurnParams] = None):
        super().__init__(sample_rate=sample_rate)
        self._params = params or SmartTurnParams()
        self._stop_ms = self._params.stop_secs * 1000
        
        self._audio_buffer: list[tuple[float, np.ndarray]] = []
        self._speech_triggered = False
        self._silence_ms = 0
        self._speech_start_time = 0

    @property
    def speech_triggered(self) -> bool:
        return self._speech_triggered

    @property
    def params(self) -> SmartTurnParams: # type: ignore
        return self._params

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        audio_int16 = np.frombuffer(buffer, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        self._audio_buffer.append((time.time(), audio_float32))

        state = EndOfTurnState.INCOMPLETE

        if is_speech:
            self._silence_ms = 0
            self._speech_triggered = True
            if self._speech_start_time == 0:
                self._speech_start_time = time.time()
        else:
            if self._speech_triggered:
                chunk_duration_ms = len(audio_int16) / (self._sample_rate / 1000)
                self._silence_ms += chunk_duration_ms
                if self._silence_ms >= self._stop_ms:
                    state = EndOfTurnState.COMPLETE
                    self._clear(state)
            else:
                max_buffer_time = (self._params.pre_speech_ms / 1000) + self._params.stop_secs + self._params.max_duration_secs
                while self._audio_buffer and self._audio_buffer[0][0] < time.time() - max_buffer_time:
                    self._audio_buffer.pop(0)

        return state

    async def analyze_end_of_turn(self) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        state, result = await self._process_speech_segment(self._audio_buffer)
        if state == EndOfTurnState.COMPLETE:
            self._clear(state)
        return state, result

    def clear(self):
        self._clear(EndOfTurnState.COMPLETE)

    def _clear(self, turn_state: EndOfTurnState):
        self._speech_triggered = turn_state == EndOfTurnState.INCOMPLETE
        self._audio_buffer = []
        self._speech_start_time = 0
        self._silence_ms = 0

    async def _process_speech_segment(self, audio_buffer) -> Tuple[EndOfTurnState, Optional[MetricsData]]:
        if not audio_buffer:
            return EndOfTurnState.INCOMPLETE, None

        start_time = self._speech_start_time - (self._params.pre_speech_ms / 1000)
        start_index = 0
        for i, (t, _) in enumerate(audio_buffer):
            if t >= start_time:
                start_index = i
                break

        segment_audio_chunks = [chunk for _, chunk in audio_buffer[start_index:]]
        if not segment_audio_chunks:
            return EndOfTurnState.INCOMPLETE, None
            
        segment_audio = np.concatenate(segment_audio_chunks)

        max_samples = int(self._params.max_duration_secs * self.sample_rate)
        if len(segment_audio) > max_samples:
            segment_audio = segment_audio[-max_samples:]

        try:
            result = await self._predict_endpoint(segment_audio)
            state = EndOfTurnState.COMPLETE if result["prediction"] == 1 else EndOfTurnState.INCOMPLETE
            # We'll skip metrics for now
            result_data = None
        except SmartTurnTimeoutException:
            state = EndOfTurnState.COMPLETE
            result_data = None
        
        return state, result_data

    @abstractmethod
    async def _predict_endpoint(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Subclasses must implement this to call an ML model."""
        pass