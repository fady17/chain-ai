# src/minichain/voice/native_service.py
"""
Native voice conversation service using Azure Speech SDK directly.
Provides better Arabic support and more reliable turn-taking than Pipecat.
"""
import asyncio
import logging
from typing import List, Optional, Callable

from .native_azure_stt import NativeAzureSTTService
from .native_azure_tts import NativeAzureTTSService
from .base import BaseVoiceService
from minichain.chat_models.base import BaseChatModel
from minichain.core.types import SystemMessage, HumanMessage, AIMessage, BaseMessage

logger = logging.getLogger(__name__)

class NativeAzureVoiceService(BaseVoiceService):
    """
    Native Azure voice conversation service with proper turn-taking.
    Bypasses Pipecat entirely for better Arabic language support.
    """
    
    def __init__(
        self,
        model: BaseChatModel,
        stt_service: NativeAzureSTTService,
        tts_service: NativeAzureTTSService,
        system_prompt: str = "",
        silence_timeout: float = 2.0
    ):
        self.model = model
        self.stt_service = stt_service
        self.tts_service = tts_service
        self.conversation_history: List[dict] = []
        self.silence_timeout = silence_timeout
        
        # Conversation state
        self._is_listening = False
        self._is_speaking = False
        self._conversation_active = False
        
        if system_prompt:
            self.conversation_history.append({"role": "system", "content": system_prompt})
    
    async def run(self): # type: ignore
        """
        Start the native voice conversation loop with proper turn-taking.
        """
        logger.info("🚀 Starting Native Azure Voice Assistant")
        logger.info("Arabic TTS Voice: " + self.tts_service.config.voice) # type: ignore
        
        try:
            self._conversation_active = True
            
            # Start the conversation loop
            await self._conversation_loop()
            
        except KeyboardInterrupt:
            logger.info("👋 Voice assistant interrupted by user")
        except Exception as e:
            logger.error(f"Voice assistant error: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _conversation_loop(self):
        """
        Main conversation loop with turn-taking management.
        """
        # Welcome message
        await self._speak("مرحباً! أنا مساعدك الصوتي. كيف يمكنني مساعدتك؟")
        
        while self._conversation_active:
            try:
                # Listen for user input
                user_text = await self._listen_for_user_input()
                
                if not user_text or not user_text.strip():
                    continue
                
                # Process user input and get AI response
                ai_response = await self._process_user_input(user_text.strip())
                
                if ai_response:
                    # Speak the AI response
                    await self._speak(ai_response)
                
            except Exception as e:
                logger.error(f"Conversation loop error: {e}")
                await self._speak("عذراً، حدث خطأ. يرجى المحاولة مرة أخرى.")
    
    async def _listen_for_user_input(self) -> Optional[str]:
        """
        Listen for user speech input with timeout.
        """
        if self._is_speaking:
            return None
        
        logger.info("🎤 Listening for user input...")
        self._is_listening = True
        
        # Create a queue to receive transcription
        transcription_queue = asyncio.Queue()
        
        def on_transcription(text: str):
            asyncio.create_task(transcription_queue.put(text))
        
        def on_error(error: str):
            logger.error(f"STT Error: {error}")
            asyncio.create_task(transcription_queue.put(None))
        
        try:
            # Start continuous recognition
            await self.stt_service.start_continuous_recognition(
                on_transcription=on_transcription,
                on_error=on_error
            )
            
            # Wait for transcription with timeout
            try:
                result = await asyncio.wait_for(
                    transcription_queue.get(),
                    timeout=30.0  # 30 second timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.info("⏰ Listening timeout - no speech detected")
                return None
        
        finally:
            self.stt_service.stop_recognition()
            self._is_listening = False
    
    async def _process_user_input(self, user_text: str) -> Optional[str]:
        """
        Process user input through the LLM and return response.
        """
        try:
            logger.info(f"👤 User: {user_text}")
            
            # Add user message to history
            self.conversation_history.append({"role": "user", "content": user_text})
            
            # Convert to BaseMessage format
            messages: List[BaseMessage] = []
            for msg in self.conversation_history:
                if msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            logger.info("🤖 AI thinking...")
            
            # Get LLM response (run in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            ai_message = await loop.run_in_executor(None, self.model.invoke, messages)
            ai_response = ai_message.content.strip() # type: ignore
            
            # Add AI response to history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            logger.info(f"🤖 AI: {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}")
            
            return ai_response
            
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            return "عذراً، لم أتمكن من معالجة طلبك. يرجى المحاولة مرة أخرى."
    
    async def _speak(self, text: str):
        """
        Speak text using TTS with proper state management.
        """
        if self._is_listening:
            self.stt_service.stop_recognition()
            self._is_listening = False
            # Wait a bit for the STT to fully stop
            await asyncio.sleep(0.5)
        
        self._is_speaking = True
        
        try:
            await self.tts_service.speak_async(text)
        finally:
            self._is_speaking = False
            # Wait a bit before allowing new input to prevent echo
            await asyncio.sleep(1.0)
    
    async def _cleanup(self):
        """
        Clean up resources when shutting down.
        """
        logger.info("🧹 Cleaning up voice assistant resources...")
        
        self._conversation_active = False
        
        if self._is_listening:
            self.stt_service.stop_recognition()
        
        if self._is_speaking:
            self.tts_service.stop_speaking()
        
        logger.info("✅ Cleanup complete")


