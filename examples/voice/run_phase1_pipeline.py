# examples/voice/run_phase1_pipeline.py
import sys
import pathlib
import time

# Add the project root to the Python path
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from minichain.voice.pipeline import VoicePipeline
from minichain.voice.vad.webrtc import WebRtcVAD

def main():
    print("="*50)
    print(" MiniChain Voice Pipeline - Phase 1 Test ".center(50, " "))
    print("="*50)
    print("\nThis test will run the core state machine.")
    print("Speak to transition from IDLE -> LISTENING.")
    print("Stop speaking to transition from LISTENING -> PROCESSING -> IDLE.")
    print("\nPress Ctrl+C to exit.")

    # 1. Initialize the VAD
    vad = WebRtcVAD(aggressiveness=3)

    # 2. Initialize the Pipeline
    pipeline = VoicePipeline(vad=vad)

    try:
        # 3. Start the pipeline
        pipeline.start()
        
        # Keep the main thread alive to let the pipeline run
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nUser interrupted. Shutting down...")
    finally:
        # 4. Stop the pipeline gracefully
        pipeline.stop()
        print("Test finished.")

if __name__ == "__main__":
    main()