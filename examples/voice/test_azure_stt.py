# examples/voice/test_azure_stt.py
import os
import sys
import pathlib
import queue
import threading
import time
# from dotenv import load_dotenv

# Add the project root to the Python path
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from minichain.voice.audio import AudioSource
from minichain.voice.stt.azure import AzureSTTModel

def main():
    """
    Runs a real-time test of the AzureSTTModel using a live microphone.
    """
    # load_dotenv()
    print("="*50)
    print(" MiniChain Azure STT Test ".center(50, " "))
    print("="*50)
    
    # IMPORTANT: Set your Azure Speech credentials as environment variables
    # export AZURE_SPEECH_KEY="your_key_here"
    # export AZURE_SPEECH_REGION="your_region_here"
    if "AZURE_SPEECH_KEY" not in os.environ or "AZURE_SPEECH_REGION" not in os.environ:
        print("\nERROR: Please set the AZURE_SPEECH_KEY and AZURE_SPEECH_REGION env vars.")
        return

    print("\nSpeak into your microphone. The recognized text will appear below.")
    print("Test will run for 15 seconds. Press Ctrl+C to exit early.")
    
    # 1. Initialize the STT Model. We'll test with Arabic.
    # stt_model = AzureSTTModel(language="en-US")
    stt_model = AzureSTTModel(language="ar-SA")

    # 2. A queue to hold audio chunks for the STT model
    stt_audio_queue = queue.Queue[bytes]()

    try:
        # 3. Use AudioSource to capture microphone input
        with AudioSource(audio_queue=stt_audio_queue) as source:
            
            # 4. Define a simple iterator that pulls from the queue
            def audio_iterator():
                while True:
                    try:
                        yield stt_audio_queue.get(timeout=1.0)
                    except queue.Empty:
                        print("Audio iterator timed out. Ending stream.")
                        break

            # 5. Call the stream method and print the results
            print("\n--- Transcription ---")
            for text in stt_model.stream(audio_iterator()):
                print(text, end='\r', flush=True)

    except KeyboardInterrupt:
        print("\n\nUser interrupted. Shutting down.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nTest finished.")

if __name__ == "__main__":
    main()