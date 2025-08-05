# tests/voice/test_native_voice.py - Fixed async tests
"""
Comprehensive tests for native Azure voice services.
"""
import pytest
import asyncio
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from minichain.V0voice import (
    NativeAzureSTTService,
    NativeAzureTTSService, 
    NativeAzureVoiceService,
    AzureSTTConfig,
    AzureTTSConfig
)


@pytest.mark.unit
class TestNativeAzureSTTService:
    """Test native Azure STT service."""
    
    @pytest.fixture
    def stt_config(self, mock_azure_credentials):
        return AzureSTTConfig(**mock_azure_credentials)
    
    @pytest.fixture
    def stt_service(self, stt_config):
        return NativeAzureSTTService(stt_config, language="ar-EG")
    
    def test_initialization(self, stt_service):
        """Test service initialization."""
        assert stt_service.language == "ar-EG"
        assert not stt_service._is_listening
        assert stt_service.SAMPLE_RATE == 16000
        assert stt_service.CHANNELS == 1
    
    @patch('minichain.voice.native_azure_stt.speechsdk')
    def test_setup_azure_config(self, mock_speechsdk, stt_service):
        """Test Azure SDK configuration."""
        stt_service._setup_azure_config()
        
        # Verify SpeechConfig was created
        mock_speechsdk.SpeechConfig.assert_called_once_with(
            subscription="test_azure_key_12345",
            region="test_region"
        )
        
        # Verify language was set
        speech_config = mock_speechsdk.SpeechConfig.return_value
        assert speech_config.speech_recognition_language == "ar-EG"
    
    def test_get_pipecat_service_raises(self, stt_service):
        """Test that legacy method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Native Azure STT service doesn't use Pipecat"):
            stt_service.get_pipecat_service()
    
    @patch('minichain.voice.native_azure_stt.threading.Thread')
    @patch('minichain.voice.native_azure_stt.speechsdk')
    async def test_start_continuous_recognition(self, mock_speechsdk, mock_thread, stt_service):
        """Test starting continuous recognition."""
        # Mock callbacks
        on_transcription = Mock()
        on_error = Mock()
        
        # Mock Azure SDK components
        mock_push_stream = Mock()
        mock_speechsdk.audio.PushAudioInputStream.return_value = mock_push_stream
        
        mock_recognizer = Mock()
        mock_speechsdk.SpeechRecognizer.return_value = mock_recognizer
        
        # Start recognition
        await stt_service.start_continuous_recognition(on_transcription, on_error)
        
        # Verify recognizer was started
        mock_recognizer.start_continuous_recognition.assert_called_once()
        assert stt_service._is_listening
        
        # Verify thread was started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()


@pytest.mark.unit
class TestNativeAzureTTSService:
    """Test native Azure TTS service."""
    
    @pytest.fixture
    def tts_config(self, mock_azure_credentials):
        return AzureTTSConfig(
            voice="ar-EG-SalmaNeural",
            **mock_azure_credentials
        )
    
    @pytest.fixture
    def tts_service(self, tts_config):
        return NativeAzureTTSService(tts_config)
    
    def test_initialization(self, tts_service):
        """Test service initialization."""
        assert not tts_service._is_speaking
        assert tts_service.SAMPLE_RATE == 16000
        assert tts_service.CHANNELS == 1
    
    @patch('minichain.voice.native_azure_tts.speechsdk')
    def test_setup_azure_config(self, mock_speechsdk, tts_service):
        """Test Azure SDK configuration."""
        tts_service._setup_azure_config()
        
        # Verify SpeechConfig was created
        mock_speechsdk.SpeechConfig.assert_called_once_with(
            subscription="test_azure_key_12345",
            region="test_region"
        )
        
        # Verify voice was set
        speech_config = mock_speechsdk.SpeechConfig.return_value
        assert speech_config.speech_synthesis_voice_name == "ar-EG-SalmaNeural"
    
    @patch('minichain.voice.native_azure_tts.pyaudio')
    @patch('minichain.voice.native_azure_tts.speechsdk')
    def test_speak_blocking(self, mock_speechsdk, mock_pyaudio, tts_service):
        """Test blocking speech synthesis."""
        # Mock successful synthesis
        mock_result = Mock()
        mock_result.reason = mock_speechsdk.ResultReason.SynthesizingAudioCompleted
        mock_result.audio_data = b"fake_audio_data"
        
        mock_synthesizer = Mock()
        mock_synthesizer.speak_text.return_value = mock_result
        mock_speechsdk.SpeechSynthesizer.return_value = mock_synthesizer
        
        # Mock audio playback
        mock_audio = Mock()
        mock_stream = Mock()
        mock_audio.open.return_value = mock_stream
        mock_pyaudio.PyAudio.return_value = mock_audio
        
        # Test
        tts_service.speak_blocking("مرحبا")
        
        # Verify synthesis was called
        mock_synthesizer.speak_text.assert_called_once_with("مرحبا")
        
        # Verify audio playback
        mock_stream.write.assert_called_once_with(b"fake_audio_data")
        mock_stream.stop_stream.assert_called_once()
        mock_stream.close.assert_called_once()
    
    @patch('minichain.voice.native_azure_tts.speechsdk')
    def test_speak_blocking_failure(self, mock_speechsdk, tts_service):
        """Test handling of synthesis failure."""
        # Mock failed synthesis
        mock_result = Mock()
        mock_result.reason = mock_speechsdk.ResultReason.Canceled
        mock_cancellation = Mock()
        mock_cancellation.reason = "TestError"
        mock_cancellation.error_details = "Test error details"
        mock_result.cancellation_details = mock_cancellation
        
        mock_synthesizer = Mock()
        mock_synthesizer.speak_text.return_value = mock_result
        mock_speechsdk.SpeechSynthesizer.return_value = mock_synthesizer
        
        # Test that exception is raised
        with pytest.raises(Exception, match="Speech synthesis canceled: TestError"):
            tts_service.speak_blocking("مرحبا")
    
    def test_get_pipecat_service_raises(self, tts_service):
        """Test that legacy method raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Native Azure TTS service doesn't use Pipecat"):
            tts_service.get_pipecat_service()


@pytest.mark.unit
class TestNativeAzureVoiceService:
    """Test native Azure voice conversation service."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock chat model."""
        model = Mock()
        mock_response = Mock()
        mock_response.content = "مرحبا بك"
        model.invoke.return_value = mock_response
        return model
    
    @pytest.fixture
    def mock_stt_service(self):
        """Mock STT service."""
        service = Mock(spec=NativeAzureSTTService)
        service.start_continuous_recognition = AsyncMock()
        service.stop_recognition = Mock()
        return service
    
    @pytest.fixture
    def mock_tts_service(self):
        """Mock TTS service."""
        service = Mock(spec=NativeAzureTTSService)
        service.config = Mock(voice="ar-EG-SalmaNeural")
        service.speak_async = AsyncMock()
        return service
    
    @pytest.fixture
    def voice_service(self, mock_model, mock_stt_service, mock_tts_service):
        """Create voice service with mocked dependencies."""
        return NativeAzureVoiceService(
            model=mock_model,
            stt_service=mock_stt_service,
            tts_service=mock_tts_service,
            system_prompt="أنت مساعد ذكي"
        )
    
    def test_initialization(self, voice_service):
        """Test service initialization."""
        assert len(voice_service.conversation_history) == 1
        assert voice_service.conversation_history[0]["role"] == "system"
        assert voice_service.conversation_history[0]["content"] == "أنت مساعد ذكي"
        assert not voice_service._is_listening
        assert not voice_service._is_speaking
        assert not voice_service._conversation_active
        assert voice_service.silence_timeout == 2.0
    
    async def test_process_user_input(self, voice_service, mock_model):
        """Test user input processing."""
        result = await voice_service._process_user_input("مرحبا")
        
        assert result == "مرحبا بك"
        assert len(voice_service.conversation_history) == 3  # system + user + assistant
        
        # Check conversation history
        assert voice_service.conversation_history[1]["role"] == "user"
        assert voice_service.conversation_history[1]["content"] == "مرحبا"
        assert voice_service.conversation_history[2]["role"] == "assistant"
        assert voice_service.conversation_history[2]["content"] == "مرحبا بك"
        
        mock_model.invoke.assert_called_once()
    
    async def test_process_user_input_error_handling(self, voice_service, mock_model):
        """Test error handling in user input processing."""
        # Make model raise an exception
        mock_model.invoke.side_effect = Exception("Model error")
        
        result = await voice_service._process_user_input("مرحبا")
        
        # Should return error message in Arabic
        assert "عذراً" in result
        assert "لم أتمكن من معالجة طلبك" in result
    
    async def test_speak(self, voice_service, mock_stt_service, mock_tts_service):
        """Test TTS with STT management."""
        # Mock STT as listening
        voice_service._is_listening = True
        
        await voice_service._speak("مرحبا")
        
        # Verify STT was stopped
        mock_stt_service.stop_recognition.assert_called_once()
        
        # Verify TTS was called
        mock_tts_service.speak_async.assert_called_once_with("مرحبا")
        
        # Verify speaking state was managed
        assert not voice_service._is_speaking
    
    async def test_cleanup(self, voice_service, mock_stt_service, mock_tts_service):
        """Test cleanup process."""
        # Set up state
        voice_service._conversation_active = True
        voice_service._is_listening = True
        voice_service._is_speaking = True
        
        await voice_service._cleanup()
        
        # Verify cleanup
        assert not voice_service._conversation_active
        mock_stt_service.stop_recognition.assert_called_once()
        mock_tts_service.stop_speaking.assert_called_once()


@pytest.mark.integration
class TestIntegration:
    """Integration tests (require real credentials)."""
    
    def test_real_tts(self):
        """Test real Azure TTS synthesis."""
        config = AzureTTSConfig(
            api_key=os.getenv('AZURE_SPEECH_KEY', 'dummy'),
            region=os.getenv('AZURE_SPEECH_REGION', 'eastus'),
            voice="ar-EG-SalmaNeural"
        )
        
        tts_service = NativeAzureTTSService(config)
        
        # This should not raise an exception in real environment
        # Note: Audio won't play in CI, but synthesis should work
        if os.getenv('AZURE_SPEECH_KEY'):
            tts_service.speak_blocking("مرحبا")
        else:
            pytest.skip("Real Azure credentials not available")
    
    async def test_real_stt_setup(self):
        """Test real Azure STT setup (no actual speech)."""
        config = AzureSTTConfig(
            api_key=os.getenv('AZURE_SPEECH_KEY', 'dummy'),
            region=os.getenv('AZURE_SPEECH_REGION', 'eastus')
        )
        
        stt_service = NativeAzureSTTService(config, language="ar-EG")
        
        if os.getenv('AZURE_SPEECH_KEY'):
            # Test configuration setup (no actual recognition)
            stt_service._setup_azure_config()
            assert stt_service._speech_config is not None
        else:
            pytest.skip("Real Azure credentials not available")

