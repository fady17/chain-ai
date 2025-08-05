# src/minichain/voice/tts/azure.py
import os
from typing import Iterator

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

from minichain.voice.tts.base import BaseTTSModel
from minichain.text_splitters.streaming import StreamingArabicSentenceSplitter

load_dotenv()

class AzureTTSModel(BaseTTSModel):
    def __init__(self, voice_name: str = "ar-SA-ZariyahNeural"):
        self.voice_name = voice_name
        
        speech_key = os.environ.get("AZURE_SPEECH_KEY")
        speech_region = os.environ.get("AZURE_SPEECH_REGION")
        if not speech_key or not speech_region:
            raise ValueError("Azure credentials not found. Check .env file.")
            
        self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        self.speech_config.speech_synthesis_voice_name = self.voice_name
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
        )

    def stream(self, text_chunk_iterator: Iterator[str]) -> Iterator[bytes]:
        """
        Uses a single, persistent SpeechSynthesizer to stream sentence audio,
        preventing resource exhaustion and ensuring reliable playback.
        """
        # --- THIS IS THE FIX ---
        # Create a single synthesizer for the entire streaming operation.
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        
        splitter = StreamingArabicSentenceSplitter()

        def text_stream_processor():
            # Process the stream of text chunks
            for text_chunk in text_chunk_iterator:
                for sentence in splitter.add_chunk(text_chunk):
                    yield sentence
            
            # Flush any remaining text
            for sentence in splitter.flush():
                yield sentence
        
        for sentence in text_stream_processor():
            if not sentence.strip():
                continue
            
            print(f"AzureTTS: Synthesizing sentence: '{sentence}'")
            
            # Use the single, reused synthesizer instance
            result = synthesizer.speak_text_async(sentence).get()

            if result.reason == speechsdk.ResultReason.Canceled: # type: ignore
                print(f"AzureTTS: Synthesis CANCELED for text: {sentence}")
                continue
            
            audio_data = result.audio_data # type: ignore
            if audio_data:
                # Strip the 44-byte WAV header
                yield audio_data[44:]
# # src/minichain/voice/tts/azure.py
# import os
# from typing import Iterator

# import azure.cognitiveservices.speech as speechsdk
# from dotenv import load_dotenv

# from minichain.voice.tts.base import BaseTTSModel
# # Import our new streaming splitter
# from minichain.text_splitters.streaming import StreamingArabicSentenceSplitter

# load_dotenv()

# class AzureTTSModel(BaseTTSModel):
#     """
#     A streaming Text-to-Speech implementation using Azure Cognitive Services.
#     This version uses a dedicated StreamingSentenceSplitter for low-latency response.
#     """
#     def __init__(self, voice_name: str = "ar-SA-ZariyahNeural"):
#         self.voice_name = voice_name
        
#         speech_key = os.environ.get("AZURE_SPEECH_KEY")
#         speech_region = os.environ.get("AZURE_SPEECH_REGION")
#         if not speech_key or not speech_region:
#             raise ValueError("Azure credentials not found. Check .env file.")
            
#         self.speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
#         self.speech_config.speech_synthesis_voice_name = self.voice_name
#         self.speech_config.set_speech_synthesis_output_format(
#             speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
#         )

#     def _synthesize_sentence(self, text: str) -> bytes:
#         """Synthesizes a single sentence and returns the raw PCM audio data."""
#         synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
#         result = synthesizer.speak_text_async(text).get()

#         if result.reason == speechsdk.ResultReason.Canceled: # type: ignore
#             print(f"AzureTTS: Synthesis CANCELED for text: {text}")
#             return b''
        
#         return result.audio_data[44:] if result.audio_data else b'' # type: ignore

#     def stream(self, text_chunk_iterator: Iterator[str]) -> Iterator[bytes]:
#         """
#         Uses StreamingArabicSentenceSplitter to buffer text and yields audio data
#         as each sentence is synthesized.
#         """
#         # Use the superior, specialized splitter
#         splitter = StreamingArabicSentenceSplitter()
        
#         # Process the stream of text chunks
#         for text_chunk in text_chunk_iterator:
#             for sentence in splitter.add_chunk(text_chunk):
#                 print(f"AzureTTS: Synthesizing sentence: '{sentence}'")
#                 audio_data = self._synthesize_sentence(sentence)
#                 if audio_data:
#                     yield audio_data
        
#         # After the loop, flush any remaining text from the splitter's buffer
#         for sentence in splitter.flush():
#             print(f"AzureTTS: Synthesizing final sentence: '{sentence}'")
#             audio_data = self._synthesize_sentence(sentence)
#             if audio_data:
#                 yield audio_data