# src/minichain/pipecat/audio/utils.py
import numpy as np

def calculate_audio_volume_rms(audio: bytes) -> float:
    """
    Calculates the volume of an audio chunk using Root Mean Square (RMS).
    A simpler alternative to loudness calculation. Normalized to [0, 1].
    """
    if not audio:
        return 0.0
    # Convert bytes to numpy array of 16-bit integers
    audio_np = np.frombuffer(audio, dtype=np.int16)
    # Calculate RMS
    rms = np.sqrt(np.mean(audio_np.astype(np.float32)**2))
    # Normalize. Max value for int16 is 32767.
    normalized_rms = rms / 32767.0
    return min(normalized_rms, 1.0)

def exp_smoothing(value: float, prev_value: float, factor: float) -> float:
    """Apply exponential smoothing to a value."""
    return prev_value + factor * (value - prev_value)