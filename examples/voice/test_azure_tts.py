# examples/voice/test_azure_tts.py
import sys
import pathlib
import time

# Add the project root to the Python path
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from minichain.voice.audio import AudioSink
from minichain.voice.tts.azure import AzureTTSModel

def main():
    """
    Runs a real-time test of the AzureTTSModel, playing the audio.
    """
    print("="*50)
    print(" MiniChain Azure TTS Test ".center(50, " "))
    print("="*50)
    print("\nThis test will synthesize text and play it through your speakers.")
    
    # 1. Initialize the TTS Model (using a standard English voice for this test)
    # tts_model = AzureTTSModel(voice_name="en-US-JennyNeural")
    # For Arabic:
    tts_model = AzureTTSModel(voice_name="ar-SA-ZariyahNeural")

    # The text we want to synthesize. We'll simulate a text stream using an iterator.
    # text_to_speak = "Hello, this is a test of the streaming text to speech system in MiniChain."
    # For Arabic:
    text_to_speak = "مرحباً، هذا اختبار لنظام تحويل النص إلى كلام في مشروع ميني تشين."

    # Simulate a stream of text chunks from an LLM
    def text_iterator():
        for word in text_to_speak.split():
            yield word + " "
            # time.sleep(0.05) # Simulate delay between tokens

    try:
        # 2. Use AudioSink to play the audio
        with AudioSink() as sink:
            print(f"\nSynthesizing and playing: '{text_to_speak}'")
            
            # 3. Get the audio stream from the TTS model
            audio_stream = tts_model.stream(text_iterator())
            
            # 4. Play the stream chunk by chunk
            for audio_chunk in audio_stream:
                sink.play_chunk(audio_chunk)
            
            sink.join()
            # Wait for the sink's internal buffer to finish playing
            print("\nPlayback complete. Waiting for buffer to clear...")
            # time.sleep(2)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Test finished.")

if __name__ == "__main__":
    main()