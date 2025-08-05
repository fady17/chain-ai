# tests/voice/test_vad.py
import os
import sys
import pytest
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Mock ALL the problematic modules before any imports
sys.modules['pkg_resources'] = MagicMock()
sys.modules['webrtcvad'] = MagicMock()

# Now we can safely import our modules
from minichain.voice.vad.webrtc import WebRtcVAD
from minichain.voice.audio import RATE

# Get a reference to the mocked webrtcvad for our tests
import webrtcvad as webrtcvad_mock

def test_webrtc_vad_initialization():
    """Tests that our wrapper correctly initializes the underlying VAD."""
    # Reset the mock to clear any previous calls
    webrtcvad_mock.reset_mock() # type: ignore
    
    # Test with a valid aggressiveness
    vad = WebRtcVAD(aggressiveness=2)
    webrtcvad_mock.Vad.assert_called_once_with(2) # type: ignore

def test_webrtc_vad_invalid_aggressiveness():
    """Tests that our wrapper raises an error for invalid aggressiveness."""
    with pytest.raises(ValueError, match="Aggressiveness must be between 0 and 3."):
        WebRtcVAD(aggressiveness=5)
        
    with pytest.raises(ValueError):
        WebRtcVAD(aggressiveness=-1)

def test_is_speech_calls_underlying_method():
    """
    Tests that our is_speech method correctly calls the library's method
    and returns its result.
    """
    # Reset and configure the mock
    webrtcvad_mock.reset_mock() # type: ignore
    mock_vad_instance = MagicMock()
    webrtcvad_mock.Vad.return_value = mock_vad_instance # type: ignore
    
    vad = WebRtcVAD()
    
    # Test case 1: The underlying VAD returns True
    mock_vad_instance.is_speech.return_value = True
    fake_speech_chunk = b'\x01' * 320 # 20ms of fake data
    assert vad.is_speech(fake_speech_chunk) is True
    # Verify it was called with the correct arguments
    mock_vad_instance.is_speech.assert_called_with(fake_speech_chunk, sample_rate=RATE)

    # Test case 2: The underlying VAD returns False
    mock_vad_instance.is_speech.return_value = False
    fake_silence_chunk = b'\x00' * 320 # 20ms of fake silence
    assert vad.is_speech(fake_silence_chunk) is False
    # Verify it was called again with the new data
    mock_vad_instance.is_speech.assert_called_with(fake_silence_chunk, sample_rate=RATE)

def test_is_speech_handles_invalid_input_type():
    """Tests that our wrapper enforces the input type to be bytes."""
    # Reset and configure the mock
    webrtcvad_mock.reset_mock() # type: ignore
    mock_vad_instance = MagicMock()
    webrtcvad_mock.Vad.return_value = mock_vad_instance # type: ignore
    
    vad = WebRtcVAD()
    with pytest.raises(TypeError, match="Audio chunk must be of type bytes."):
        vad.is_speech("not a byte string") # type: ignore