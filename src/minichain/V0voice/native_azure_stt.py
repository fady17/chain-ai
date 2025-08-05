# src/minichain/voice/native_azure_stt.py
"""
Native Azure Speech-to-Text service that bypasses Pipecat for better Arabic support.
"""
import asyncio
import threading
import queue
from typing import Optional, Callable
import logging

try:
    import azure.cognitiveservices.speech as speechsdk
    import pyaudio
    import wave
    import io
except ImportError:
    raise ImportError(
        "Native Azure voice dependencies not installed. "
        "Run: pip install azure-cognitiveservices-speech pyaudio"
    )

from .base import BaseSTTService, AzureSTTConfig

logger = logging.getLogger(__name__)

class NativeAzureSTTService(BaseSTTService):
    """
    Native Azure STT service with real-time audio processing.
    Provides better Arabic language support than Pipecat wrapper.
    """
    
    def __init__(self, config: AzureSTTConfig, language: str = "ar-EG"):
        super().__init__(config)
        self.language = language
        self._speech_config = None
        self._audio_config = None
        self._recognizer = None
        self._is_listening = False
        
        # Audio settings
        self.CHUNK_SIZE = 1024
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        self._setup_azure_config()
    
    def _setup_azure_config(self):
        """Initialize Azure Speech SDK configuration."""
        config = self.config
        assert isinstance(config, AzureSTTConfig)
        
        self._speech_config = speechsdk.SpeechConfig(
            subscription=config.api_key,
            region=config.region
        )
        
        # Configure for Arabic recognition
        self._speech_config.speech_recognition_language = self.language
        
        # Optimize for real-time recognition
        self._speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "500"
        )
        self._speech_config.set_property(
            speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500"
        )
    
    async def start_continuous_recognition(
        self, 
        on_transcription: Callable[[str], None],
        on_error: Optional[Callable[[str], None]] = None
    ):
        """
        Start continuous speech recognition with real-time transcription callbacks.
        """
        if self._is_listening:
            logger.warning("STT service is already listening")
            return
        
        try:
            # Setup audio input from microphone
            audio_format = speechsdk.AudioDataStream(
                samples_per_second=self.SAMPLE_RATE, # type: ignore
                bits_per_sample=16,  # type: ignore
                channels=self.CHANNELS  # type: ignore
            )
            
            # Create push audio input stream
            push_stream = speechsdk.audio.PushAudioInputStream(audio_format)  # type: ignore
            self._audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
            
            # Create speech recognizer
            self._recognizer = speechsdk.SpeechRecognizer(
                speech_config=self._speech_config,  # type: ignore
                audio_config=self._audio_config
            )
            
            # Setup event handlers
            def recognized_handler(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = evt.result.text.strip()
                    if text:
                        logger.info(f"Recognized: {text}")
                        on_transcription(text)
            
            def error_handler(evt):
                error_msg = f"Recognition error: {evt.result.reason}"
                logger.error(error_msg)
                if on_error:
                    on_error(error_msg)
            
            # Connect event handlers
            self._recognizer.recognized.connect(recognized_handler)
            self._recognizer.canceled.connect(error_handler)
            
            # Start continuous recognition
            self._recognizer.start_continuous_recognition()
            self._is_listening = True
            
            # Start audio capture in separate thread
            audio_thread = threading.Thread(
                target=self._audio_capture_loop,
                args=(push_stream,),
                daemon=True
            )
            audio_thread.start()
            
            logger.info(f"🎤 Started continuous Arabic STT recognition ({self.language})")
            
        except Exception as e:
            error_msg = f"Failed to start STT recognition: {e}"
            logger.error(error_msg)
            if on_error:
                on_error(error_msg)
            raise
    
    def _audio_capture_loop(self, push_stream):
        """
        Capture audio from microphone and push to Azure Speech SDK.
        """
        audio = pyaudio.PyAudio()
        
        try:
            # Open microphone stream
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )
            
            logger.info("🎤 Audio capture started")
            
            while self._is_listening:
                try:
                    # Read audio data
                    audio_data = stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    
                    # Push to Azure Speech SDK
                    push_stream.write(audio_data)
                    
                except Exception as e:
                    logger.error(f"Audio capture error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Failed to initialize audio capture: {e}")
        finally:
            if 'stream' in locals():
                stream.stop_stream()  # type: ignore
                stream.close()  # type: ignore
            audio.terminate()
            push_stream.close()
    
    def stop_recognition(self):
        """Stop continuous speech recognition."""
        if not self._is_listening:
            return
        
        self._is_listening = False
        
        if self._recognizer:
            self._recognizer.stop_continuous_recognition()
            self._recognizer = None
        
        logger.info("🛑 STT recognition stopped")
    
    def get_pipecat_service(self):
        """
        Legacy compatibility method - not used in native implementation.
        """
        raise NotImplementedError(
            "Native Azure STT service doesn't use Pipecat. "
            "Use start_continuous_recognition() instead."
        )
