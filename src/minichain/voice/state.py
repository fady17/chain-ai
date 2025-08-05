# src/minichain/voice/state.py
from enum import Enum, auto

class VoiceState(Enum):
    """
    Defines the possible states of the voice pipeline.
    
    - IDLE: The pipeline is initialized but not actively listening or speaking.
    - LISTENING: The pipeline is actively capturing audio from the source.
    - PROCESSING: The pipeline has detected the end of user speech and is
                  processing the input (STT, LLM, etc.). This is a transient state.
    - SPEAKING: The pipeline is generating audio and sending it to the sink.
    - INTERRUPTED: The user has started speaking while the AI was speaking.
    """
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()
    INTERRUPTED = auto()