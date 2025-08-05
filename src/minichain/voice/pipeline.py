# src/minichain/voice/pipeline.py

import queue
import time
import threading
from typing import Optional, Iterator

from minichain.voice.audio import AudioSource, AudioSink
from minichain.voice.state import VoiceState
from minichain.voice.vad.base import BaseVAD
from minichain.voice.stt.base import BaseSTTModel
from minichain.voice.tts.base import BaseTTSModel
from minichain.chat_models.base import BaseChatModel
from minichain.core.types import AIMessage, HumanMessage
from minichain.text_splitters.streaming import StreamingArabicSentenceSplitter

class VoicePipeline:
    def __init__(self, llm: BaseChatModel, stt: BaseSTTModel, tts: BaseTTSModel, vad: BaseVAD, end_of_speech_timeout: float = 0.8, max_phrase_duration: float = 20.0):
        self.llm, self.stt, self.tts, self.vad = llm, stt, tts, vad
        self.end_of_speech_timeout, self.max_phrase_duration = end_of_speech_timeout, max_phrase_duration
        
        self.state = VoiceState.IDLE
        self.conversation_history = []
        
        self._audio_queue = queue.Queue[bytes]()
        self._source: Optional[AudioSource] = None
        self._sink: Optional[AudioSink] = None
        
        self._pipeline_thread: Optional[threading.Thread] = None
        self._processing_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    def _clear_audio_queue(self):
        with self._audio_queue.mutex: self._audio_queue.queue.clear()

    def _pipeline_loop(self):
        """ The main, real-time loop. Its only job is to manage the user's audio input and detect interruptions. """
        voice_buffer = []
        speech_started_at, silence_started_at = 0, 0
        
        while self._running.is_set():
            try:
                chunk = self._audio_queue.get(block=True, timeout=0.1)
                is_speech = self.vad.is_speech(chunk)

                # Interruption Logic
                if self.state == VoiceState.SPEAKING and is_speech:
                    print(f"[{time.time():.2f}] INTERRUPTION DETECTED!")
                    if self._sink: self._sink.stop()
                    # Wait for the processing thread to finish its cleanup
                    if self._processing_thread and self._processing_thread.is_alive():
                        self._processing_thread.join(timeout=0.5)
                    self.state = VoiceState.LISTENING
                    speech_started_at, silence_started_at = time.time(), 0
                    voice_buffer.clear()
                    voice_buffer.append(chunk)
                    continue

                # Core VAD Logic
                if self.state == VoiceState.IDLE and is_speech:
                    self.state = VoiceState.LISTENING
                    speech_started_at, silence_started_at = time.time(), 0
                    voice_buffer.append(chunk)

                elif self.state == VoiceState.LISTENING:
                    voice_buffer.append(chunk)
                    if is_speech:
                        silence_started_at = 0
                    elif silence_started_at == 0:
                        silence_started_at = time.time()

                    if time.time() - speech_started_at > self.max_phrase_duration or \
                       (silence_started_at > 0 and time.time() - silence_started_at >= self.end_of_speech_timeout):
                        
                        print(f"[{time.time():.2f}] End of speech. Launching processing thread.")
                        self.state = VoiceState.PROCESSING
                        self._processing_thread = threading.Thread(
                            target=self._handle_processing_and_speaking, 
                            args=(list(voice_buffer),)
                        )
                        voice_buffer.clear()
                        self._processing_thread.start()

            except queue.Empty:
                if self.state == VoiceState.LISTENING and silence_started_at > 0 and (time.time() - silence_started_at >= self.end_of_speech_timeout):
                    print(f"[{time.time():.2f}] End of speech (timeout). Launching processing thread.")
                    self.state = VoiceState.PROCESSING
                    self._processing_thread = threading.Thread(
                        target=self._handle_processing_and_speaking, 
                        args=(list(voice_buffer),)
                    )
                    voice_buffer.clear()
                    self._processing_thread.start()
                continue
            except Exception as e:
                print(f"CRITICAL ERROR in pipeline loop: {e}")
                self.state = VoiceState.IDLE

    def _handle_processing_and_speaking(self, audio_data: list[bytes]):
        """ Runs in a separate thread to handle blocking I/O (STT, LLM, TTS). """
        try:
            # 1. STT
            user_text = "".join(self.stt.stream(iter(audio_data)))
            if not user_text.strip():
                self.state = VoiceState.IDLE
                return

            print(f"You -> {user_text}")
            self.conversation_history.append(HumanMessage(content=user_text))
            
            # 2. LLM
            self.state = VoiceState.SPEAKING
            llm_text_stream = self.llm.stream(self.conversation_history)
            
            # 3. TTS
            splitter = StreamingArabicSentenceSplitter()
            def sentence_stream():
                for chunk in llm_text_stream:
                    yield from splitter.add_chunk(chunk)
                yield from splitter.flush()

            tts_audio_stream = self.tts.stream(sentence_stream())
            
            if self._sink:
                for audio_chunk in tts_audio_stream:
                    if self.state != VoiceState.SPEAKING:
                        print("Processing thread: Interruption detected. Halting TTS.")
                        return # Gracefully exit the thread
                    self._sink.play_chunk(audio_chunk)
                self._sink.join()
            
            # If we completed without being interrupted, add to history and go to IDLE
            if self.state == VoiceState.SPEAKING:
                # TODO: Add full AI response to history
                self._clear_audio_queue()
                self.state = VoiceState.IDLE

        except Exception as e:
            print(f"Error in processing thread: {e}")
            self.state = VoiceState.IDLE

                
    # start() and stop() remain the same
    def start(self):
        if self._running.is_set():
            return
        print("Starting pipeline...")
        self._running.set()
        
        self._source = AudioSource(audio_queue=self._audio_queue)
        self._sink = AudioSink()
        self._source.__enter__()
        self._sink.__enter__()

        self._pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._pipeline_thread.start()
        print("Pipeline started.")

    def stop(self):
        if not self._running.is_set():
            return
        print("Stopping pipeline...")
        self._running.clear()
        
        if self._pipeline_thread and self._pipeline_thread.is_alive():
            self._pipeline_thread.join(timeout=2.0)
        
        if self._sink: self._sink.__exit__(None, None, None)
        if self._source: self._source.__exit__(None, None, None)
            
        self.state = VoiceState.IDLE
        print("Pipeline stopped.")
# # src/minichain/voice/pipeline.py
# import queue
# import time
# import threading
# from typing import Optional

# from minichain.voice.audio import AudioSource
# from minichain.voice.state import VoiceState
# from minichain.voice.vad.base import BaseVAD

# class VoicePipeline:
#     """
#     The main orchestrator for a voice conversation.
    
#     This pipeline manages the flow of audio data from the microphone,
#     through Voice Activity Detection (VAD), and controls the conversation state.
#     """
#     def __init__(
#         self,
#         vad: BaseVAD,
#         end_of_speech_timeout: float = 0.8,
#         max_phrase_duration: float = 20.0,
#     ):
#         """
#         Initializes the VoicePipeline.

#         Args:
#             vad: An instance of a VAD implementation (e.g., WebRtcVAD).
#             end_of_speech_timeout: Seconds of silence to wait for before considering
#                                    the user to have finished speaking.
#             max_phrase_duration: Maximum seconds a single user utterance can be.
#         """
#         self.vad = vad
#         self.end_of_speech_timeout = end_of_speech_timeout
#         self.max_phrase_duration = max_phrase_duration
        
#         self.state = VoiceState.IDLE
#         self._audio_queue = queue.Queue[bytes]()
#         self._source: Optional[AudioSource] = None
#         self._pipeline_thread: Optional[threading.Thread] = None
#         self._running = threading.Event()
        
#     def _pipeline_loop(self):
#         """The main processing loop for the voice pipeline."""
        
#         speech_started_at: float = 0
#         silence_started_at: float = 0
        
#         # This will hold the audio chunks of the current user utterance
#         voice_buffer = []

#         while self._running.is_set():
#             try:
#                 chunk = self._audio_queue.get(block=False)
#             except queue.Empty:
#                 time.sleep(0.01)
#                 continue

#             is_speech = self.vad.is_speech(chunk)

#             if self.state == VoiceState.IDLE:
#                 if is_speech:
#                     print(f"[{time.time():.2f}] Speech detected. Transitioning to LISTENING.")
#                     self.state = VoiceState.LISTENING
#                     speech_started_at = time.time()
#                     silence_started_at = 0
#                     voice_buffer.append(chunk) # Start collecting audio
            
#             elif self.state == VoiceState.LISTENING:
#                 # --- THIS IS THE CORRECTED LOGIC ---
#                 voice_buffer.append(chunk)

#                 if time.time() - speech_started_at > self.max_phrase_duration:
#                     print(f"[{time.time():.2f}] Max phrase duration reached. Transitioning to PROCESSING.")
#                     self.state = VoiceState.PROCESSING
#                     continue

#                 if is_speech:
#                     # If user is speaking, reset the silence timer.
#                     # This is the key to not cutting them off.
#                     silence_started_at = 0
#                 else:
#                     # If user is silent, start or check the silence timer.
#                     if silence_started_at == 0:
#                         silence_started_at = time.time()
                    
#                     # ONLY check for timeout if the silence timer is running.
#                     if time.time() - silence_started_at >= self.end_of_speech_timeout:
#                         print(f"[{time.time():.2f}] End of speech detected ({self.end_of_speech_timeout}s). Transitioning to PROCESSING.")
#                         self.state = VoiceState.PROCESSING
            
#             elif self.state == VoiceState.PROCESSING:
#                 print(f"Processing {len(voice_buffer)} audio chunks.")
                
#                 # In the future, voice_buffer will be sent to the STT stream.
#                 # For now, we just clear it.
#                 voice_buffer.clear()
                
#                 print("Processing complete. Transitioning back to IDLE.")
#                 with self._audio_queue.mutex:
#                     self._audio_queue.queue.clear()
#                 self.state = VoiceState.IDLE

#     def start(self):
#         """Starts the voice pipeline."""
#         if self.state != VoiceState.IDLE:
#             print("Pipeline is already running.")
#             return

#         print("Starting pipeline...")
#         self._running.set()
        
#         # AudioSource is managed here, ensuring it's only active when the pipeline is
#         self._source = AudioSource(audio_queue=self._audio_queue)
#         self._source.__enter__() # Manually enter the context

#         self._pipeline_thread = threading.Thread(target=self._pipeline_loop, daemon=True)
#         self._pipeline_thread.start()
#         print("Pipeline started.")

#     def stop(self):
#         """Stops the voice pipeline and cleans up resources."""
#         if not self._running.is_set():
#             return
            
#         print("Stopping pipeline...")
#         self._running.clear()
        
#         if self._pipeline_thread:
#             self._pipeline_thread.join(timeout=1.0)
        
#         if self._source:
#             self._source.__exit__(None, None, None) # Manually exit the context
            
#         self.state = VoiceState.IDLE
#         print("Pipeline stopped.")