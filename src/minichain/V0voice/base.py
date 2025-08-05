# src/minichain/voice/base.py
"""
Defines abstract base classes and Pydantic configuration models for all
voice components in the Mini-Chain framework.
"""
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

# --- Configuration Models ---

class STTConfig(BaseModel):
    """Base configuration model for all STT services."""
    provider: str = Field(description="The name of the STT provider (e.g., 'azure').")

class AzureSTTConfig(STTConfig):
    """Configuration for the Azure Speech-to-Text service."""
    provider: str = "azure"
    api_key: str
    region: str

class TTSConfig(BaseModel):
    """Base configuration model for all TTS services."""
    provider: str = Field(description="The name of the TTS provider (e.g., 'azure').")

class AzureTTSConfig(TTSConfig):
    """Configuration for the Azure Text-to-Speech service."""
    provider: str = "azure"
    api_key: str
    region: str
    voice: str

# --- Service Interfaces ---

class BaseSTTService(ABC):
    """Abstract base class for a Speech-to-Text service."""
    def __init__(self, config: STTConfig):
        self.config = config
    
    @abstractmethod
    def get_pipecat_service(self) -> Any:
        """
        Must be implemented by subclasses to return a configured instance of
        the underlying Pipecat service object.
        """
        pass

class BaseTTSService(ABC):
    """Abstract base class for a Text-to-Speech service."""
    def __init__(self, config: TTSConfig):
        self.config = config
        
    @abstractmethod
    def get_pipecat_service(self) -> Any:
        """
        Must be implemented by subclasses to return a configured instance of
        the underlying Pipecat service object.
        """
        pass

class BaseVoiceService(ABC):
    """Abstract base class for a high-level voice conversation service."""
    @abstractmethod
    def run(self):
        """Starts the main conversation loop."""
        pass