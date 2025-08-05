# examples/voice/01_hello_stt.py
"""
Example 1: Hello, STT! (Speech-to-Text)

This script demonstrates the simplest possible use of a Mini-Chain STT component.
It uses the high-level `run_stt` function, which handles all the complex
pipeline logic for a simple transcription task.
"""
import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.V0voice import AzureSTTConfig, AzureSTTService, run_stt

def print_transcript(text: str):
    """A simple callback that prints the final transcribed text."""
    print(f"[ You Said ] -> \"{text}\"")

def main():
    print("\n--- Mini-Chain STT Example: Hello, World! ---")
    load_dotenv()
    
    api_key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not api_key or not region:
        print("\n❌ ERROR: Azure Speech credentials not found in .env file.")
        return

    try:
        stt_config = AzureSTTConfig(api_key=api_key, region=region)
        stt_service = AzureSTTService(config=stt_config)
        
        # Start the interactive transcription loop, passing our callback.
        run_stt(stt_service=stt_service, on_transcription=print_transcript)

    except (ImportError, TypeError) as e:
        print(f"\n❌ ERROR: Failed to initialize STT service. Is `minichain-ai[voice]` installed?")
        print(f"Details: {e}")

if __name__ == "__main__":
    main()