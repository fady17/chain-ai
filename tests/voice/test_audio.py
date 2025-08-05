# tests/voice/test_audio.py
import os
import sys
import threading
import pytest
import queue
import time
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.voice.audio import AudioSource, AudioSink, CHUNK_SIZE

@pytest.fixture
def mock_pyaudio():
    """A pytest fixture to mock the entire pyaudio library."""
    with patch('minichain.voice.audio.pyaudio') as mock_pyaudio_lib:
        # Configure the mock to return another mock when PyAudio() is called
        mock_instance = MagicMock()
        mock_stream = MagicMock()
        
        # When p.open() is called, return our mock stream
        mock_instance.open.return_value = mock_stream
        
        # When PyAudio() is instantiated, return our mock instance
        mock_pyaudio_lib.PyAudio.return_value = mock_instance
        
        yield mock_pyaudio_lib, mock_instance, mock_stream

def test_audio_source_captures_data(mock_pyaudio):
    """
    Tests that AudioSource correctly starts, reads from a (mocked) stream,
    and puts the data into the provided queue.
    """
    mock_pyaudio_lib, mock_instance, mock_stream = mock_pyaudio
    
    # Configure the mock stream to return fake data once, then wait
    fake_audio_chunk = b'\x01' * (CHUNK_SIZE * 2)
    
    # Use an event to control when the mock should return data
    data_ready = threading.Event()
    call_count = 0
    
    def mock_read(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return fake_audio_chunk
        else:
            # Wait for the event (which we'll never set) to simulate blocking
            data_ready.wait(timeout=0.5)
            return b''  # Return empty data after timeout
    
    mock_stream.read.side_effect = mock_read
   
    test_queue = queue.Queue()

    with AudioSource(audio_queue=test_queue):
        # Get the first chunk - this should succeed
        captured_chunk = test_queue.get(timeout=0.2)
        # Give a tiny bit more time to ensure no additional chunks
        time.sleep(0.05)

    # Assertions
    mock_instance.open.assert_called_once()
    mock_stream.read.assert_called()
    assert captured_chunk == fake_audio_chunk
    
    # The queue should be empty since subsequent reads return empty data
    assert test_queue.empty()
    
    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_instance.terminate.assert_called_once()
def test_audio_sink_plays_data(mock_pyaudio):
    """
    Tests that AudioSink correctly starts, takes data from its play_chunk method,
    and writes it to a (mocked) stream.
    """
    mock_pyaudio_lib, mock_instance, mock_stream = mock_pyaudio
    fake_audio_chunk = b'\x02' * CHUNK_SIZE

    with AudioSink() as sink:
        sink.play_chunk(fake_audio_chunk)
        # Allow the thread time to get the item from the queue and write it
        time.sleep(0.1)

    # Assertions
    mock_instance.open.assert_called_once()
    mock_stream.write.assert_called_once_with(fake_audio_chunk)

    mock_stream.stop_stream.assert_called_once()
    mock_stream.close.assert_called_once()
    mock_instance.terminate.assert_called_once()

def test_audio_sink_stop_clears_queue(mock_pyaudio):
    """

    Tests that calling stop() on the sink immediately clears any buffered audio,
    which is critical for interruption handling.
    """
    mock_pyaudio_lib, mock_instance, mock_stream = mock_pyaudio

    with AudioSink() as sink:
        # Load the queue with some data
        sink.play_chunk(b'first_chunk')
        sink.play_chunk(b'second_chunk_that_should_be_cleared')

        # Call stop, which should clear the queue
        sink.stop()
        
        # Give the thread a moment to process the stop signal
        time.sleep(0.1)

    # The playback loop might have played the very first chunk before stop() was called,
    # which is a valid race condition. The key is that it should NOT play the second.
    # So we check that write was called at most once.
    assert mock_stream.write.call_count <= 1
    if mock_stream.write.call_count == 1:
        mock_stream.write.assert_called_once_with(b'first_chunk')