# src/minichain/voice/native_azure_tts.py
"""
Native Azure Text-to-Speech service that bypasses Pipecat for better Arabic support.
"""
import asyncio
import threading
import queue
from typing import Optional, Callable
import logging
import io

try:
    import azure.cognitiveservices.speech as speechsdk
    import pyaudio
except ImportError:
    raise ImportError(
        "Native Azure voice dependencies not installed. "
        "Run: pip install azure-cognitiveservices-speech pyaudio"
    )

from .base import BaseTTSService, AzureTTSConfig

logger = logging.getLogger(__name__)

class NativeAzureTTSService(BaseTTSService):
    """
    Native Azure TTS service with direct audio output.
    Provides better Arabic language support than Pipecat wrapper.
    """
    
    def __init__(self, config: AzureTTSConfig):
        super().__init__(config)
        self._speech_config = None
        self._synthesizer = None
        self._is_speaking = False
        
        # Audio output settings
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1
        self.FORMAT = pyaudio.paInt16
        
        self._setup_azure_config()
    
    def _setup_azure_config(self):
        """Initialize Azure Speech SDK configuration."""
        config = self.config
        assert isinstance(config, AzureTTSConfig)
        
        self._speech_config = speechsdk.SpeechConfig(
            subscription=config.api_key,
            region=config.region
        )
        
        # Set the Arabic voice
        self._speech_config.speech_synthesis_voice_name = config.voice
        
        # Optimize audio quality
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
        )
    
    async def speak_async(
        self, 
        text: str,
        on_start: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        """
        Asynchronously synthesize and play speech.
        """
        try:
            if self._is_speaking:
                logger.warning("TTS service is already speaking")
                return
            
            self._is_speaking = True
            
            if on_start:
                on_start()
            
            logger.info(f"🔊 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Run synthesis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._synthesize_and_play, text)
            
            if on_complete:
                on_complete()
                
            logger.info("✅ Speech synthesis complete")
            
        except Exception as e:
            error_msg = f"TTS synthesis failed: {e}"
            logger.error(error_msg)
            if on_error:
                on_error(error_msg)
            raise
        finally:
            self._is_speaking = False
    
    def _synthesize_and_play(self, text: str):
        """
        Synchronously synthesize speech and play through speakers.
        """
        # Create synthesizer with null audio output (we'll handle playback manually)
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self._speech_config, # type: ignore
            audio_config=audio_config
        )
        
        # Synthesize speech
        result = synthesizer.speak_text(text)
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Play the audio data
            self._play_audio_data(result.audio_data)
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"Speech synthesis canceled: {cancellation.reason}"
            if cancellation.error_details:
                error_msg += f" - {cancellation.error_details}"
            raise Exception(error_msg)
        else:
            raise Exception(f"Unexpected synthesis result: {result.reason}")
    
    def _play_audio_data(self, audio_data: bytes):
        """
        Play raw audio data through the system speakers.
        """
        audio = pyaudio.PyAudio()
        
        try:
            # Open audio output stream
            stream = audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                output=True
            )
            
            # Play audio data
            stream.write(audio_data)
            
            # Wait for playback to complete
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            raise
        finally:
            audio.terminate()
    
    def speak_blocking(self, text: str):
        """
        Synchronously synthesize and play speech (blocking call).
        """
        try:
            logger.info(f"🔊 Speaking: {text[:50]}{'...' if len(text) > 50 else ''}")
            self._synthesize_and_play(text)
            logger.info("✅ Speech synthesis complete")
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise
    
    def stop_speaking(self):
        """Stop current speech synthesis."""
        self._is_speaking = False
        # Note: Actual interruption of synthesis would require more complex state management
        logger.info("🛑 TTS synthesis stopped")
    
    def get_pipecat_service(self):
        """
        Legacy compatibility method - not used in native implementation.
        """
        raise NotImplementedError(
            "Native Azure TTS service doesn't use Pipecat. "
            "Use speak_async() or speak_blocking() instead."
        )

