# src/minichain/voice/stt/azure.py
import os
import queue
import threading
from typing import Iterator
from dotenv import load_dotenv

import azure.cognitiveservices.speech as speechsdk
from minichain.voice.stt.base import BaseSTTModel

load_dotenv()
class AzureSTTModel(BaseSTTModel):
    """
    A streaming Speech-to-Text implementation using Azure Cognitive Services.
    """
    def __init__(self, language: str = "en-US"):
        """
        Initializes the Azure STT model.

        Args:
            language: The language code for speech recognition (e.g., "en-US", "ar-SA").
        """
        self.language = language
        
        # Load configuration from environment variables
        speech_key = os.environ.get("AZURE_SPEECH_KEY")
        speech_region = os.environ.get("AZURE_SPEECH_REGION")
        if not speech_key or not speech_region:
            raise ValueError(
                "Azure Speech key and region must be provided via "
                "AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables."
            )
            
        self.speech_config = speechsdk.SpeechConfig(
            subscription=speech_key, region=speech_region
        )
        self.speech_config.speech_recognition_language = self.language
        # Optional: Profanity masking
        self.speech_config.set_profanity(speechsdk.ProfanityOption.Masked)

    def _audio_pusher_thread(
        self, audio_chunk_iterator: Iterator[bytes], push_stream: speechsdk.audio.PushAudioInputStream
    ):
        """
        A dedicated thread that pushes audio chunks from our pipeline into Azure's stream.
        """
        try:
            for chunk in audio_chunk_iterator:
                push_stream.write(chunk)
            print("AzureSTT: Audio pusher thread finished.")
        finally:
            # Closing the stream is crucial. It signals to Azure that the audio is complete.
            push_stream.close()

    def stream(self, audio_chunk_iterator: Iterator[bytes]) -> Iterator[str]:
        """
        Transcribes a stream of audio chunks into a stream of text.
        """
        # 1. Setup the audio stream that the SpeechRecognizer will pull from
        push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        
        # 2. Create the SpeechRecognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config, audio_config=audio_config
        )

        # 3. Setup the queue to bridge the event-based SDK with our iterator-based method
        text_queue = queue.Queue[str | None]()
        
        # 4. Define event handlers
        def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs):
            """Called for intermediate recognition results."""
            if evt.result.text:
                text_queue.put(evt.result.text)

        def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs):
            """Called for final recognition results."""
            # We can use this for a final, more polished transcript if needed.
            # For pure streaming, we often rely on the `recognizing` event.
            if evt.result.text:
                print(f"AzureSTT: Finalized text: '{evt.result.text}'")
        
        def on_session_stopped(evt: speechsdk.SessionEventArgs):
            """Called when the session ends."""
            print("AzureSTT: Session stopped.")
            text_queue.put(None) # Sentinel value to end the loop

        def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs):
            """Called if there's an error."""
            # Pylance may incorrectly flag 'reason' and 'error_details' as unknown,
            # but they are correct attributes according to the Azure SDK documentation.
            reason = evt.reason # type: ignore
            print(f"AzureSTT: Canceled. Reason: {reason}")
            
            if reason == speechsdk.CancellationReason.Error:
                error_details = evt.error_details # type: ignore
                print(f"AzureSTT: Error details: {error_details}")
                
            text_queue.put(None) # Sentinel value to end the loop

        # 5. Connect the handlers to the recognizer's events
        recognizer.recognizing.connect(on_recognizing)
        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_session_stopped)
        recognizer.canceled.connect(on_canceled)
        
        # 6. Start the producer thread to push audio into the stream
        pusher = threading.Thread(
            target=self._audio_pusher_thread, args=(audio_chunk_iterator, push_stream)
        )
        pusher.daemon = True
        pusher.start()
        
        # 7. Start the recognizer. This is non-blocking.
        recognizer.start_continuous_recognition()
        print("AzureSTT: Recognition started.")
        
        try:
            # 8. The consumer loop: Get results from the queue and yield them
            while True:
                text = text_queue.get()
                if text is None: # Check for the sentinel value
                    break
                yield text
        finally:
            # 9. Cleanup: Ensure recognition is stopped
            print("AzureSTT: Stopping recognition...")
            recognizer.stop_continuous_recognition()
            pusher.join(timeout=1.0) # Wait for the pusher thread to finish
            print("AzureSTT: Recognition stopped.")