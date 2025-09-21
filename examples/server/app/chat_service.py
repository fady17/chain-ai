# File: app/chat_service.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from chain.chat_models import LocalChatModel, LocalChatConfig
from chain.core.types import HumanMessage, SystemMessage
from typing import AsyncGenerator
import json

class ChatService:
    """A stateless service for direct conversations with the local LLM."""
    
    def __init__(self):
        try:
            config = LocalChatConfig()
            self.chat_model = LocalChatModel(config=config)
            self.system_prompt = "You are a helpful legal AI assistant for an egyption attorney you should always be formal and always speak on egyption arabic never use english responses even if the user input is english with no emojis."
        except Exception as e:
            print(f"FATAL: Could not initialize LocalChatModel. Is LM Studio running? Error: {e}")
            self.chat_model = None

    async def stream_chat(self, question: str) -> AsyncGenerator[str, None]:
        if not self.chat_model:
            error_obj = {"type": "error", "message": "Chat model is not available."}
            yield f"{json.dumps(error_obj)}\n"
            return

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question)
        ]
        
        try:
            is_reasoning = False
            for chunk in self.chat_model.stream(messages):
                # This is the new logic to separate text from reasoning
                if '<think>' in chunk:
                    is_reasoning = True
                    chunk = chunk.replace('<think>', '')
                if '</think>' in chunk:
                    is_reasoning = False
                    chunk = chunk.replace('</think>', '')
                
                if is_reasoning and chunk.strip():
                    reasoning_obj = {"type": "reasoning", "delta": chunk}
                    yield f"{json.dumps(reasoning_obj)}\n"
                elif chunk.strip():
                    text_obj = {"type": "text", "delta": chunk}
                    yield f"{json.dumps(text_obj)}\n"

        except Exception as e:
            error_obj = {"type": "error", "message": str(e)}
            yield f"{json.dumps(error_obj)}\n"