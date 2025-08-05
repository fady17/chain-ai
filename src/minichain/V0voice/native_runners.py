# src/minichain/voice/native_runners.py
"""
High-level runner functions for native Azure voice services.
"""
import asyncio
from typing import Callable

from .native_azure_stt import NativeAzureSTTService
from .native_azure_tts import NativeAzureTTSService

def run_native_stt(stt_service: NativeAzureSTTService, on_transcription: Callable[[str], None]):
    """
    Run native STT with continuous recognition.
    """
    async def main():
        print("🎤 Native Azure STT - Listening... Press Ctrl+C to exit.")
        
        def on_error(error: str):
            print(f"❌ STT Error: {error}")
        
        try:
            await stt_service.start_continuous_recognition(
                on_transcription=on_transcription,
                on_error=on_error
            )
            
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n👋 STT session stopped.")
        finally:
            stt_service.stop_recognition()
    
    asyncio.run(main())

def run_native_tts(tts_service: NativeAzureTTSService, text_to_speak: str):
    """
    Run native TTS to speak text.
    """
    try:
        print(f"🔊 Native Azure TTS - Speaking: \"{text_to_speak[:50]}{'...' if len(text_to_speak) > 50 else ''}\"")
        tts_service.speak_blocking(text_to_speak)
        print("✅ Speech complete.")
    except Exception as e:
        print(f"❌ TTS Error: {e}")
