# src/minichain/voice/vad/webrtc.py
import webrtcvad
from minichain.voice.vad.base import BaseVAD
from minichain.voice.audio import RATE, CHANNELS, AUDIO_FORMAT

# WebRTC VAD requires specific frame durations: 10, 20, or 30 ms.
# We calculate the number of bytes per frame duration for validation.
# Sample width is 2 bytes for 16-bit audio (pyaudio.paInt16)
SAMPLE_WIDTH = 2 # Bytes
FRAMES_PER_10_MS = int(RATE / 100) # 16000 / 100 = 160 frames
BYTES_PER_10_MS = FRAMES_PER_10_MS * SAMPLE_WIDTH * CHANNELS # 160 * 2 * 1 = 320 bytes

VALID_CHUNK_SIZES = {
    BYTES_PER_10_MS,      # 10ms
    BYTES_PER_10_MS * 2,  # 20ms
    BYTES_PER_10_MS * 3   # 30ms
}

class WebRtcVAD(BaseVAD):
    """
    A Voice Activity Detector powered by Google's WebRTC library.
    """
    def __init__(self, aggressiveness: int = 1):
        """
        Initializes the WebRTC VAD.

        Args:
            aggressiveness (int): The VAD's aggressiveness mode. An integer
                between 0 and 3. 0 is the least aggressive about filtering
                out non-speech, 3 is the most aggressive.
        """
        if not 0 <= aggressiveness <= 3:
            raise ValueError("Aggressiveness must be between 0 and 3.")
        
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, chunk: bytes) -> bool:
        """
        Uses the WebRTC VAD to determine if an audio chunk contains speech.

        Args:
            chunk: A raw audio chunk. Must be a valid frame length and sample rate.

        Returns:
            bool: True if speech is detected, False otherwise.
        """
        if not isinstance(chunk, bytes):
            raise TypeError("Audio chunk must be of type bytes.")
            
        # --- THIS IS THE CRITICAL VALIDATION ---
        # The VAD is extremely sensitive to the chunk length.
        # This check ensures we only process valid frame sizes.
        if len(chunk) not in VALID_CHUNK_SIZES:
            raise ValueError(
                f"Invalid chunk size for VAD. Received {len(chunk)} bytes, "
                f"but expected one of {list(VALID_CHUNK_SIZES)}."
            )
        
        return self._vad.is_speech(chunk, sample_rate=RATE)