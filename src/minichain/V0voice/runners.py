# src/minichain/voice/runners.py
"""
Provides high-level, simplified runner functions for executing common voice tasks.
"""
import asyncio
from typing import Callable

from .base import BaseSTTService, BaseTTSService

def run_stt(stt_service: BaseSTTService, on_transcription: Callable[[str], None]):
    """
    Runs a blocking Speech-to-Text loop using the local microphone.
    """
    # --- FIX: Lazy Imports ---
    # Imports are done inside the function. This allows the library to be
    # imported without error, even if voice dependencies are not installed.
    try:
        from pipecat.frames.frames import Frame, TranscriptionFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.processors.frame_processor import FrameProcessor
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    except ImportError:
        raise ImportError("Pipecat dependencies not found. Run `pip install 'minichain-ai[voice]'`.")

    class _TranscriptionCallbackProcessor(FrameProcessor):
        def __init__(self, callback: Callable[[str], None]):
            super().__init__()
            self._callback = callback

        async def process_frame(self, frame: Frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                await asyncio.get_event_loop().run_in_executor(None, self._callback, frame.text)

    async def main():
        transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True))
        pipecat_stt = stt_service.get_pipecat_service()
        callback_processor = _TranscriptionCallbackProcessor(callback=on_transcription)
        
        pipeline = Pipeline([transport.input(), pipecat_stt, callback_processor])
        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        
        print("🎤 Listening... Speak into your microphone. Press Ctrl+C to exit.")
        await runner.run(task)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 STT session stopped.")

def run_tts(tts_service: BaseTTSService, text_to_speak: str):
    """
    Runs a blocking Text-to-Speech task that speaks the given text.
    """
    try:
        from pipecat.frames.frames import TextFrame, EndFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.task import PipelineTask
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
    except ImportError:
        raise ImportError("Pipecat dependencies not found. Run `pip install 'minichain-ai[voice]'`.")
        
    async def main():
        transport = LocalAudioTransport(LocalAudioTransportParams(audio_out_enabled=True))
        pipecat_tts = tts_service.get_pipecat_service()
        pipeline = Pipeline([pipecat_tts, transport.output()])
        task = PipelineTask(pipeline)
        runner = PipelineRunner()
        
        print(f"\n🔊 Speaking: \"{text_to_speak}\"...")

        async def send_text_stream():
            await asyncio.sleep(1)
            words = text_to_speak.split()
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                await task.queue_frame(TextFrame(text=chunk))
                await asyncio.sleep(0.01)
            
            await asyncio.sleep(3)
            await task.queue_frame(EndFrame())

        await asyncio.gather(runner.run(task), send_text_stream())
        print("✅ TTS task complete.")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 TTS task stopped.")