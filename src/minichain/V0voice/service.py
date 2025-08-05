# src/minichain/voice/service.py
"""
The primary orchestrator for creating real-time, streaming, turn-taking
voice conversations.
"""
import asyncio
from typing import List

from .base import BaseVoiceService, BaseSTTService, BaseTTSService
from minichain.chat_models.base import BaseChatModel
from minichain.core.types import SystemMessage, HumanMessage, AIMessage, BaseMessage

# Lazy imports are the most robust pattern for optional dependencies.
# All pipecat-specific code is contained within the run() method.
class PipecatVoiceService(BaseVoiceService):
    """
    A robust voice service orchestrator that manages turn-taking to prevent
    the AI from hearing its own voice. This implementation prioritizes stability
    by getting the full LLM response before starting TTS.
    """
    def __init__(self,
                 model: BaseChatModel,
                 stt_service: BaseSTTService,
                 tts_service: BaseTTSService,
                 system_prompt: str = ""):
        
        self.model = model
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.conversation_history: List[dict] = []
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})

    def run(self):
        """
        Assembles and runs the full, turn-managed voice pipeline.
        """
        # --- Lazy Imports for Robustness ---
        try:
            from pipecat.frames.frames import Frame, TextFrame, TranscriptionFrame
            from pipecat.pipeline.task import PipelineTask
            from pipecat.pipeline.runner import PipelineRunner
            from pipecat.pipeline.pipeline import Pipeline
            from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
            from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
            from pipecat.audio.vad.silero import SileroVADAnalyzer
        except ImportError as e:
            raise ImportError(
                f"Pipecat dependencies not fully installed. Please run `pip install 'minichain-ai[voice]'`. Original error: {e}"
            )

        # --- Internal Bridge Processor (Now Stable) ---
        class _LLMBridge(FrameProcessor):
            def __init__(self, model: BaseChatModel, history: list, tts_task_queue: asyncio.Queue):
                super().__init__()
                self._model = model
                self._history = history
                self._tts_task_queue = tts_task_queue

            async def process_frame(self, frame: Frame, direction: FrameDirection):
                await super().process_frame(frame, direction)
                if isinstance(frame, TranscriptionFrame):
                    user_text = frame.text.strip()
                    if not user_text: return

                    print(f"[ User ] -> {user_text}")
                    self._history.append({"role": "user", "content": user_text})
                    
                    messages: List[BaseMessage] = [
                        SystemMessage(content=msg["content"]) if msg["role"] == "system" else
                        HumanMessage(content=msg["content"]) if msg["role"] == "user" else
                        AIMessage(content=msg["content"]) for msg in self._history
                    ]
                    
                    print("[ AI   ] -> Thinking...")
                    
                    # --- The Definitive Stability Fix ---
                    # Run the entire blocking LLM `.invoke()` call in a separate thread.
                    # This prevents the LLM from freezing the real-time audio pipeline.
                    loop = asyncio.get_event_loop()
                    full_response = await loop.run_in_executor(
                        None, self._model.invoke, messages
                    )
                    full_response = full_response.strip()
                    # --- End Fix ---
                    
                    print(f"[ AI   ] -> Responding: {full_response}")
                    self._history.append({"role": "assistant", "content": full_response})
                    
                    # Put the complete, final text onto a queue for the
                    # separate TTS task to handle.
                    await self._tts_task_queue.put(full_response)
        
        # --- Main async function to run the pipeline ---
        async def main():
            transport = LocalAudioTransport(
                params=LocalAudioTransportParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    vad_analyzer=SileroVADAnalyzer()
                )
            )
            stt = self.stt_service.get_pipecat_service()
            tts = self.tts_service.get_pipecat_service()
            
            # This queue allows the main pipeline to safely communicate with the TTS pipeline.
            tts_task_queue = asyncio.Queue()
            
            llm_bridge = _LLMBridge(model=self.model, history=self.conversation_history, tts_task_queue=tts_task_queue)
            
            # This task will run in parallel, listening for text and speaking it.
            async def tts_task_handler(queue: asyncio.Queue):
                # The TTS pipeline is separate to ensure it has a clean flow.
                pipeline = Pipeline([tts, transport.output()])
                task = PipelineTask(pipeline)
                
                async def text_streamer():
                    while True:
                        text_to_speak = await queue.get()
                        words = text_to_speak.split()
                        for i, word in enumerate(words):
                            chunk = word + (" " if i < len(words) - 1 else "")
                            await task.queue_frame(TextFrame(text=chunk))
                
                # Run the TTS pipeline and the text streamer concurrently.
                await asyncio.gather(PipelineRunner().run(task), text_streamer())

            # This is the main STT -> LLM pipeline.
            main_pipeline = Pipeline([
                transport.input(),
                stt,
                llm_bridge,
            ])
            main_task = PipelineTask(main_pipeline)
            
            print("\n🔊 Mini-Chain Voice Assistant is running.")
            print("Speak into your microphone. Press Ctrl+C to exit.")
            
            # Run the main STT/LLM task and the TTS task concurrently.
            await asyncio.gather(
                PipelineRunner().run(main_task),
                tts_task_handler(tts_task_queue)
            )

        # --- Start the asyncio event loop ---
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Voice assistant shut down.")
