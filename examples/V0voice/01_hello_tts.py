# examples/voice/01_hello_tts.py
"""
Example 1: Hello, TTS! (Text-to-Speech)

This script demonstrates the simplest possible use of a Mini-Chain TTS component.

It follows three clean steps:
1.  Load credentials from a .env file.
2.  Create a Pydantic configuration object (`AzureTTSConfig`) for the service.
3.  Pass the configured service to the high-level `run_tts` function.
"""
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.V0voice import AzureTTSConfig, AzureTTSService, run_tts

def main():
    """Main execution function."""
    print("\n--- Mini-Chain TTS Example: Hello, World! ---")
    
    load_dotenv()
    api_key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    
    if not api_key or not region:
        print("\n❌ ERROR: Azure Speech credentials not found in .env file.")
        return

    try:
        # 1. Create a Pydantic configuration object for the service.
        tts_config = AzureTTSConfig(
            api_key=api_key,
            region=region,
            voice="en-US-JennyNeural"
        )
        
        # 2. Instantiate the service with the configuration object.
        tts_service = AzureTTSService(config=tts_config)

    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize TTS service: {e}")
        return
        
    # 3. Define the text and use the high-level runner to speak it.
    text_to_say = "Hello from the Mini-Chain library! This is a test of the streaming voice API."
    run_tts(tts_service=tts_service, text_to_speak=text_to_say)

if __name__ == "__main__":
    main()