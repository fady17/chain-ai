# src/minichain/chat_models/run.py
"""
Provides a high-level, interactive runner function for chat models.
"""
from typing import List, Dict, Union
from .base import BaseChatModel
from ..core.types import SystemMessage, HumanMessage, AIMessage, BaseMessage

def run_chat(model: BaseChatModel, system_prompt: Union[str, None] = None):
    """
    Starts an interactive, streaming chat session in the terminal.

    Args:
        model: A configured instance of a Mini-Chain chat model.
        system_prompt: An optional system message to set the AI's persona.
    """
    print("\n" + "="*50)
    print(" Mini-Chain Interactive Chat ".center(50, " "))
    print("="*50)
    print("Enter your message. Type 'exit' or 'quit' to end the session.")
    
    history: List[Dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})
        
    while True:
        try:
            user_input = input("\n[ You ] -> ")
            if user_input.lower() in ["exit", "quit"]:
                print("\n🤖 Session ended. Goodbye!")
                break
                
            history.append({"role": "user", "content": user_input})
            
            # Convert history to Pydantic message objects for the model
            messages_for_llm: List[BaseMessage] = [
                SystemMessage(content=msg["content"]) if msg["role"] == "system"
                else HumanMessage(content=msg["content"]) if msg["role"] == "user"
                else AIMessage(content=msg["content"])
                for msg in history
            ]
            
            print("[ AI  ] -> ", end="", flush=True)
            
            # Use the streaming interface for a responsive feel
            full_response = ""
            for chunk in model.stream(messages_for_llm):
                print(chunk, end="", flush=True)
                full_response += chunk
            print() # for newline
            
            # Add the full response to history for the next turn
            history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\n\n🤖 Session ended. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break