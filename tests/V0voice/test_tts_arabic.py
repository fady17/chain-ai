# Enhanced TTS configuration with fallback voices

import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
# Direct Azure Speech SDK test - bypassing Pipecat
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

def test_direct_azure_tts():
    """Test Azure TTS directly without Pipecat"""
    load_dotenv()
    
    # Configure Azure Speech
    speech_config = speechsdk.SpeechConfig(
        subscription=os.environ.get('AZURE_SPEECH_KEY'), 
        region=os.environ.get('AZURE_SPEECH_REGION')
    )
    
    # Test different Arabic voices
    arabic_voices = [
        'ar-EG-SalmaNeural',      # Egyptian Female
        'ar-EG-ShakirNeural',     # Egyptian Male
        'ar-SA-ZariyahNeural',    # Saudi Female
        'ar-SA-HamedNeural',      # Saudi Male
    ]
    
    test_text = "مرحبا، هذا اختبار للصوت العربي"
    
    for voice in arabic_voices:
        print(f"\n🔧 Testing voice: {voice}")
        
        try:
            # Set the voice
            speech_config.speech_synthesis_voice_name = voice
            
            # Use default speaker
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            
            # Create synthesizer
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            # Synthesize
            print(f"🔊 Speaking: '{test_text}'")
            result = speech_synthesizer.speak_text_async(test_text).get()
            
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:  # type: ignore
                print(f"✅ SUCCESS: {voice} worked!")
                return voice  # Return the first working voice
                
            elif result.reason == speechsdk.ResultReason.Canceled: # type: ignore
                cancellation_details = result.cancellation_details  # type: ignore
                print(f"❌ FAILED: {voice}")
                print(f"   Reason: {cancellation_details.reason}")
                if cancellation_details.error_details:
                    print(f"   Error: {cancellation_details.error_details}")
                    
        except Exception as e:
            print(f"❌ Exception with {voice}: {e}")
    
    print("\n❌ All Arabic voices failed! Trying English as control test...")
    
    # Test English as control
    try:
        speech_config.speech_synthesis_voice_name = 'en-US-JennyNeural'
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = speech_synthesizer.speak_text_async("Hello, this is a test").get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:  # type: ignore
            print("✅ English voice works - issue is Arabic-specific")
        else:
            print("❌ Even English failed - check your Azure credentials")
            
    except Exception as e:
        print(f"❌ English test failed: {e}")
    
    return None

def get_available_voices():
    """Get list of available voices from Azure"""
    load_dotenv()
    
    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=os.environ.get('AZURE_SPEECH_KEY'), 
            region=os.environ.get('AZURE_SPEECH_REGION')
        )
        
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
        voices = synthesizer.get_voices_async().get()
        
        print(f"\n🔧 Available Arabic voices in region {os.environ.get('AZURE_SPEECH_REGION')}:")
        arabic_voices = [v for v in voices.voices if v.locale.startswith('ar-')]  # type: ignore
        
        if arabic_voices:
            for voice in arabic_voices:
                print(f"   - {voice.short_name} ({voice.local_name}) - {voice.locale}")
        else:
            print("   ❌ NO Arabic voices available in this region!")
            
        return arabic_voices
        
    except Exception as e:
        print(f"❌ Failed to get voices: {e}")
        return []

def main():
    """Run direct Azure TTS tests"""
    print("=" * 70)
    print("DIRECT AZURE SPEECH SDK TEST")
    print("=" * 70)
    
    # Check environment
    print(f"🔧 Region: {os.environ.get('AZURE_SPEECH_REGION')}")
    print(f"🔧 Key set: {bool(os.environ.get('AZURE_SPEECH_KEY'))}")
    
    # Get available voices first
    available_voices = get_available_voices()
    
    # Test TTS
    working_voice = test_direct_azure_tts()
    
    if working_voice:
        print(f"\n🎯 SOLUTION: Use voice '{working_voice}' in your Pipecat config")
    else:
        print(f"\n💡 SUGGESTION: Try changing your Azure region to 'eastus' or 'westeurope'")
        print("   These regions have better Arabic voice support")

if __name__ == "__main__":
    main()