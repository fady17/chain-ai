# examples/voice/test_audio_loopback.py
import sys
import pathlib
import queue
import time

# Add the project root to the Python path to allow importing 'minichain'
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from minichain.voice.audio import AudioSource, AudioSink

def main():
    """
    Runs a real-time audio loopback test.
    Captures audio from the microphone and plays it back through the speakers.
    """
    print("="*50)
    print(" MiniChain Audio Loopback Test ".center(50, " "))
    print("="*50)
    print("\nSpeak into your microphone. You should hear your own voice.")
    print("This confirms that your audio input and output devices are working.")
    print("\nPress Ctrl+C to exit.")

    # A queue to bridge the source and sink
    audio_data_queue = queue.Queue()

    try:
        # The 'with' statements ensure that resources are properly managed
        # even if an error occurs.
        with AudioSource(audio_queue=audio_data_queue) as source, \
             AudioSink() as sink:
            
            # This loop represents the core job of our main pipeline thread:
            # moving data between components.
            while True:
                # Get audio data from the microphone (via the source's queue)
                chunk = audio_data_queue.get()
                
                # Send the audio data to the speakers (via the sink's queue)
                sink.play_chunk(chunk)

    except KeyboardInterrupt:
        print("\n\nExiting loopback test. Resources will be cleaned up.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have PyAudio installed (`pip install pyaudio`)")
        print("and that you have granted microphone permissions.")
    finally:
        print("Test finished.")

if __name__ == "__main__":
    main()