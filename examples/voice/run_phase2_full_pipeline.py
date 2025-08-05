# examples/voice/run_phase2_full_pipeline.py
import sys
import pathlib
import time

# Add the project root to the Python path
project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

# MiniChain Components
from minichain.chat_models import AzureOpenAIChatModel, AzureChatConfig
from minichain.voice.pipeline import VoicePipeline
from minichain.voice.stt.azure import AzureSTTModel
from minichain.voice.tts.azure import AzureTTSModel
from minichain.voice.vad.webrtc import WebRtcVAD

def main():
    print("="*50)
    print(" MiniChain Voice Pipeline - Phase 2 FULL TEST ".center(50, " "))
    print("="*50)
    print("\nThis test will run the full STT -> LLM -> TTS pipeline.")
    print("Speak into your microphone and the AI will respond.")
    print("\nPress Ctrl+C to exit.")

    try:
        # 1. Initialize the VAD
        vad = WebRtcVAD(aggressiveness=3)
        
        # 2. Initialize the STT model (ensure your .env is set)
        stt = AzureSTTModel(language="ar-SA")
        
        # 3. Initialize the TTS model (ensure your .env is set)
        tts = AzureTTSModel(voice_name="ar-SA-ZariyahNeural")
        
        # 4. Initialize the LLM (ensure your .env is set for Azure OpenAI)
        llm_config = AzureChatConfig(
            deployment_name="gpt-4", # Or your preferred model
            temperature=0.7
        )
        llm = AzureOpenAIChatModel(config=llm_config)
        
        # 5. Initialize the Full Pipeline
        pipeline = VoicePipeline(
            llm=llm,
            stt=stt,
            tts=tts,
            vad=vad
        )
        
        # 6. Start the pipeline
        pipeline.start()
        
        # Keep the main thread alive to let the pipeline run
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nUser interrupted. Shutting down...")
    except Exception as e:
        print(f"An error occurred during setup: {e}")
        # This will catch errors from missing .env file settings
    finally:
        # Stop the pipeline gracefully if it was started
        if 'pipeline' in locals() and pipeline._running.is_set(): # type: ignore
            pipeline.stop() # type: ignore
        print("Test finished.")

if __name__ == "__main__":
    main()