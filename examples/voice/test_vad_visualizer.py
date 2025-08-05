# examples/voice/test_vad_visualizer.py
import sys
import pathlib
import queue
import time

# Add the project root to the Python path
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from minichain.voice.audio import AudioSource
from minichain.voice.vad.webrtc import WebRtcVAD

# Visualizer settings
BAR_WIDTH = 40

def main():
    """
    Runs a real-time VAD visualizer.
    Captures audio and prints whether speech is detected in each chunk.
    """
    print("="*50)
    print(" MiniChain VAD Visualizer ".center(50, " "))
    print("="*50)
    print("\nSpeak and stop speaking. The bar should appear when you talk.")
    print("This confirms that the VAD is detecting your voice.")
    print("\nTry adjusting the 'aggressiveness' in the code (0-3).")
    print("Press Ctrl+C to exit.")

    audio_queue = queue.Queue()
    
    # --- VAD Configuration ---
    # Try changing this value from 0 (least aggressive) to 3 (most aggressive)
    vad_aggressiveness = 2 
    vad = WebRtcVAD(aggressiveness=vad_aggressiveness)

    try:
        with AudioSource(audio_queue=audio_queue) as source:
            print(f"\n[VAD Aggressiveness: {vad_aggressiveness}]")
            print("-" * 50)
            while True:
                chunk = audio_queue.get()
                is_speech = vad.is_speech(chunk)
                
                # Print a simple visualizer
                if is_speech:
                    bar = "█" * BAR_WIDTH
                    print(f"SPEECH DETECTED |{bar}|")
                else:
                    bar = "─" * BAR_WIDTH
                    print(f"Silence......... |{bar}|", end='\r', flush=True)

    except KeyboardInterrupt:
        print("\n\nExiting VAD visualizer.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("Test finished.")

if __name__ == "__main__":
    main()