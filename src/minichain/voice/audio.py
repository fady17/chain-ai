# src/minichain/voice/audio.py
import pyaudio
import threading
import queue
from typing import Optional

# Standard audio settings for VoIP
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit integers
CHANNELS = 1                     # Mono
RATE = 16000                     # 16kHz, common for STT/TTS
# CHUNK_SIZE = 1024                # Number of frames per buffer
CHUNK_SIZE = 480

class AudioSource:
    """
    Manages microphone input in a separate thread.

    It continuously captures audio from the microphone and places the audio chunks
    into a queue for consumption by other parts of the voice pipeline (e.g., VAD).
    """
    def __init__(self, audio_queue: queue.Queue[bytes], device_index: Optional[int] = None):
        self.audio_queue = audio_queue
        self.device_index = device_index
        self._p = pyaudio.PyAudio()
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def _capture_loop(self):
        """The main loop for the audio capture thread."""
        while self._running.is_set():
            try:
                data = self._stream.read(CHUNK_SIZE, exception_on_overflow=False) # type: ignore
                if data:
                    self.audio_queue.put(data)
            except IOError as e:
                print(f"AudioSource: IO error in capture loop: {e}. Stopping thread.")
                # We can decide to stop or attempt to recover here
                break
        
        print("AudioSource: Capture loop finished.")

    def __enter__(self):
        """Opens the audio stream and starts the capture thread."""
        print("AudioSource: Opening stream...")
        self._stream = self._p.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            input_device_index=self.device_index
        )
        self._running.set()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print("AudioSource: Stream opened and capture thread started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stops the capture thread and closes the audio stream."""
        print("AudioSource: Closing stream...")
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1.0) # Wait for the thread to finish

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        
        self._p.terminate()
        print("AudioSource: Stream closed and resources released.")


class AudioSink:
    """
    Manages speaker output in a separate thread.
    ...
    """
    def __init__(self, device_index: Optional[int] = None):
        self.audio_queue = queue.Queue[bytes | None]() # Allow None as a sentinel
        self.device_index = device_index
        self._p: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def _playback_loop(self):
        """The main loop for the audio playback thread."""
        while self._running.is_set():
            try:
                chunk = self.audio_queue.get(block=True, timeout=0.1)
                
                # The sentinel 'None' now signals a clean shutdown
                if chunk is None:
                    # Mark task as done so join() can unblock
                    self.audio_queue.task_done()
                    break
                
                if self._stream and self._running.is_set():
                    self._stream.write(chunk)
                # Mark this chunk as processed
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Catch potential errors during write if stream is closed
                print(f"AudioSink: Error during playback: {e}")
                break
        
        print("AudioSink: Playback loop finished.")

    def play_chunk(self, chunk: bytes):
        """Adds an audio chunk to the playback queue."""
        self.audio_queue.put(chunk)

    def join(self):
        """
        Blocks until all audio in the queue has been played.
        This is crucial for a graceful shutdown.
        """
        print("AudioSink: Waiting for playback queue to finish...")
        self.audio_queue.join()
        print("AudioSink: Playback queue finished.")

    def stop(self):
        """
        Stops playback immediately by clearing the queue and signaling the thread.
        Used for interruptions.
        """
        print("AudioSink: Immediate stop requested.")
        self._running.clear() # Signal thread to stop
        # Clear any pending audio
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self.audio_queue.put(None) # Sentinel to unblock the thread's `get()` call

    def __enter__(self):
        print("AudioSink: Opening stream...")
        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE,
            output=True, frames_per_buffer=CHUNK_SIZE,
            output_device_index=self.device_index
        )
        self._running.set()
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        print("AudioSink: Stream opened and playback thread started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("AudioSink: Closing stream...")
        
        # Signal the thread to finish up and wait for it
        if self._running.is_set():
            self.join() # Wait for pending items to play
            self.audio_queue.put(None) # Send sentinel to exit the loop
        self._running.clear()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            
        if self._p:
            self._p.terminate()
        print("AudioSink: Stream closed and resources released.")