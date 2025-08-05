# examples/voice/03_text_in_voice_out_chatbot.py
"""
Example 3: A Streaming Text-in, Voice-out Chatbot.

This script demonstrates a low-latency conversational loop where the user
types their input, and the AI's response is generated and spoken
token-by-token.

It showcases the combination of:
- `LocalChatModel`'s `.stream()` method for fast responses.
- `run_tts`'s streaming capability for smooth, uninterrupted audio.
"""
import os
import sys
from dotenv import load_dotenv

# Add the project's 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.chat_models import LocalChatModel, LocalChatConfig
from minichain.V0voice import AzureTTSConfig, AzureTTSService, run_tts
from minichain.core.types import SystemMessage, HumanMessage, AIMessage

def print_header(title: str):
    """Prints a formatted header for clear sections."""
    print("\n" + "=" * 70)
    print(f" {title.upper()} ".center(70, " "))
    print("=" * 70)

def main():
    """Main execution function."""
    print_header("Mini-Chain: Streaming Text-in, Voice-out Chatbot")
    load_dotenv()
    
    # --- Step 1: Configure and Initialize Components ---
    try:
        chat_config = LocalChatConfig()
        chat_model = LocalChatModel(config=chat_config)
        
        arabic_voice = " ar-AE-FatimaNeural"  # Try female voice first
        print(f"🔧 DEBUG: Setting voice to: {arabic_voice}")

        tts_config = AzureTTSConfig(
            api_key=os.getenv("AZURE_SPEECH_KEY", ""),
            region=os.getenv("AZURE_SPEECH_REGION", ""),
            voice=arabic_voice
        )
        print(f"🔧 DEBUG: TTS Config voice: {tts_config.voice}")
        
        tts_service = AzureTTSService(config=tts_config)
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize components: {e}")
        return
        
    print("✅ AI Brain and Voice are ready.")

    # --- Step 2: Define the Conversation Logic ---
    conversation_history = [
        # {"role": "system", "content": "You are desha, a friendly and helpful AI assistant. You are having a text-based conversation but your responses will be spoken aloud. Keep your answers conversational and relatively short."}
        {"role": "system", "content": "You are desha, a friendly ai assistant always respond in arabic."}
    ]
    
    # --- Step 3: The Interactive Chat Loop ---
    print("\nChat with desha. Type 'exit' to end.")
    
    while True:
        try:
            user_input = input("\n[ You ] -> ")
            if user_input.lower() in ["exit", "quit"]:
                farewell_message = "Goodbye! It was nice talking to you."
                print(f"[ desha ] -> {farewell_message}")
                run_tts(tts_service=tts_service, text_to_speak=farewell_message)
                break
            
            conversation_history.append({"role": "user", "content": user_input})
            
            messages_for_llm = [
                SystemMessage(content=msg["content"]) if msg["role"] == "system"
                else HumanMessage(content=msg["content"]) if msg["role"] == "user"
                else AIMessage(content=msg["content"])
                for msg in conversation_history
            ]
            
            # --- The Streaming Upgrade ---
            print("[ desha ] -> ", end="", flush=True)
            
            # 1. Get the stream iterator from the chat model
            response_stream = chat_model.stream(messages_for_llm) # type: ignore
            
            # 2. Consume the stream: print each chunk to the console AND
            #    collect the full response.
            full_response = ""
            for chunk in response_stream:
                print(chunk, end="", flush=True)
                full_response += chunk
            print() # for newline
            
            # 3. Add the complete response to history for the next turn
            conversation_history.append({"role": "assistant", "content": full_response})
            
            # 4. Speak the complete response aloud using the streaming TTS runner.
            run_tts(tts_service=tts_service, text_to_speak=full_response)

        except KeyboardInterrupt:
            print("\n\n🤖 Conversation ended. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()