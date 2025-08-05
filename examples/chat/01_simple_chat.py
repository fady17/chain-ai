# examples/chat/01_simple_chat.py
"""
Example 1: A simple, interactive chat session.

This script demonstrates the high-level `run_chat` function, which provides
the easiest way to start a conversation with a configured model.
"""
import os, sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.chat_models import LocalChatModel, LocalChatConfig, run_chat

def main():
    load_dotenv()
    
    # 1. Create a Pydantic configuration object.
    #    This cleanly separates configuration from code.
    config = LocalChatConfig(
        # You can override defaults here, e.g.:
        # model_name="different-model/gguf"
    )

    # 2. Instantiate the model with the config.
    try:
        chat_model = LocalChatModel(config=config)
    except Exception as e:
        print(f"Error initializing model. Is LM Studio running? Details: {e}")
        return
        
    # 3. (Optional) Define a system prompt for the AI's persona.
    system_prompt = "You are desha, a friendly and helpful AI assistant."
    
    # 4. Start the interactive session with a single function call.
    run_chat(model=chat_model, system_prompt=system_prompt)

if __name__ == "__main__":
    main()