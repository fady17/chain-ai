import os
import socket
import sys
import signal
import threading
import time
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.chat_models import LocalChatModel, LocalChatConfig
from minichain.V0voice import (
    PipecatVoiceService,
    AzureSTTService, AzureSTTConfig,
    AzureTTSService, AzureTTSConfig
)

class VoiceChatbotManager:
    def __init__(self):
        self.voice_assistant = None
        self.running = False
        self.shutdown_event = threading.Event()

    def signal_handler(self, signum, frame):
        """Handle SIGINT (Ctrl+C) gracefully"""
        print(f"\n🛑 Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown()

    def shutdown(self):
        """Gracefully shutdown the voice assistant"""
        if self.running and self.voice_assistant:
            print("⏹️  Stopping voice assistant...")
            self.running = False
            self.shutdown_event.set()
            
            # If the voice service has a stop method, call it
            if hasattr(self.voice_assistant, 'stop'):
                self.voice_assistant.stop() # type: ignore
            
            print("✅ Voice assistant stopped successfully.")
        else:
            print("⚠️  Voice assistant was not running.")

    def keyboard_monitor(self):
        """Monitor for keyboard input to trigger shutdown"""
        try:
            while self.running:
                user_input = input().strip().lower()
                if user_input in ['quit', 'exit', 'stop', 'q']:
                    print("🛑 Shutdown command received.")
                    self.shutdown()
                    break
                elif user_input == 'help':
                    print("Available commands: quit, exit, stop, q, help")
        except (EOFError, KeyboardInterrupt):
            # Handle cases where input is interrupted
            pass

    def run_voice_assistant(self, chat_model, stt_service, tts_service, system_prompt):
        """Run the voice assistant in a separate thread"""
        try:
            self.voice_assistant = PipecatVoiceService(
                model=chat_model,
                stt_service=stt_service,
                tts_service=tts_service,
                system_prompt=system_prompt
            )
            
            self.running = True
            print("🎤 Voice assistant is now running...")
            print("💡 Type 'quit', 'exit', 'stop', or 'q' to shutdown gracefully")
            print("💡 Or press Ctrl+C for immediate shutdown")
            print("-" * 50)
            
            # Start keyboard monitoring in a separate thread
            keyboard_thread = threading.Thread(target=self.keyboard_monitor, daemon=True)
            keyboard_thread.start()
            
            # Run the voice assistant
            self.voice_assistant.run()
            
        except Exception as e:
            print(f"❌ Error running voice assistant: {e}")
            self.running = False

def check_server(host='localhost', port=1234) -> bool:
    """Checks if a server is running on the specified host and port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex((host, port)) == 0

def main():
    print("\n--- Mini-Chain Streaming Voice Chatbot with Kill Switch ---")
    load_dotenv()
    
    # Create the manager instance
    manager = VoiceChatbotManager()
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, manager.signal_handler)
    signal.signal(signal.SIGTERM, manager.signal_handler)
    
    # --- Pre-flight check for the local server ---
    if not check_server():
        print("\n❌ ERROR: Local model server not detected on localhost:1234.")
        print("Please start the server in LM Studio before running this example.")
        return
    print("✅ Local model server connection verified.")

    # --- 1. Configure all components ---
    try:
        chat_config = LocalChatConfig()
        chat_model = LocalChatModel(config=chat_config)
        
        stt_config = AzureSTTConfig(
            api_key=os.getenv("AZURE_SPEECH_KEY", ""), 
            region=os.getenv("AZURE_SPEECH_REGION", "")
        )
        stt_service = AzureSTTService(config=stt_config)
        
        tts_config = AzureTTSConfig(
            api_key=os.getenv("AZURE_SPEECH_KEY", ""), 
            region=os.getenv("AZURE_SPEECH_REGION", ""), 
            voice="en-US-JennyNeural"
        )
        tts_service = AzureTTSService(config=tts_config)
        
        system_prompt = "You are desha, a friendly AI assistant. Keep your answers conversational and concise."
    
    except (ValueError, ImportError) as e:
        print(f"❌ Configuration Error: {e}")
        return
        
    print("✅ AI Brain, Ears, and Voice are ready.")

    # --- 2. Initialize and Run the Voice Service with Kill Switch ---
    try:
        manager.run_voice_assistant(chat_model, stt_service, tts_service, system_prompt)
        
        # Wait for shutdown signal
        while manager.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Keyboard interrupt received.")
        manager.shutdown()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        manager.shutdown()
    finally:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
# # examples/voice/04_voice_chatbot.py
# import os
# import socket
# import sys
# from dotenv import load_dotenv

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
# from minichain.chat_models import LocalChatModel, LocalChatConfig
# from minichain.voice import (
#     PipecatVoiceService,
#     AzureSTTService, AzureSTTConfig,
#     AzureTTSService, AzureTTSConfig
# )

# def check_server(host='localhost', port=1234) -> bool:
#     """Checks if a server is running on the specified host and port."""
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         s.settimeout(2)
#         return s.connect_ex((host, port)) == 0

# def main():
#     print("\n--- Mini-Chain Streaming Voice Chatbot ---")
#     load_dotenv()
    
#     # --- Pre-flight check for the local server ---
#     if not check_server():
#         print("\n❌ ERROR: Local model server not detected on localhost:1234.")
#         print("Please start the server in LM Studio before running this example.")
#         return
#     print("✅ Local model server connection verified.")

#     # --- 1. Configure all components ---
#     try:
#         chat_config = LocalChatConfig()
#         chat_model = LocalChatModel(config=chat_config)
        
#         stt_config = AzureSTTConfig(api_key=os.getenv("AZURE_SPEECH_KEY", ""), region=os.getenv("AZURE_SPEECH_REGION", ""))
#         stt_service = AzureSTTService(config=stt_config)
        
#         tts_config = AzureTTSConfig(api_key=os.getenv("AZURE_SPEECH_KEY", ""), region=os.getenv("AZURE_SPEECH_REGION", ""), voice="en-US-JennyNeural")
#         tts_service = AzureTTSService(config=tts_config)
        
#         system_prompt = "You are desha, a friendly AI assistant. Keep your answers conversational and concise."
    
#     except (ValueError, ImportError) as e:
#         print(f"❌ Configuration Error: {e}")
#         return
        
#     print("✅ AI Brain, Ears, and Voice are ready.")

#     # --- 2. Initialize and Run the Voice Service ---
#     voice_assistant = PipecatVoiceService(
#         model=chat_model,
#         stt_service=stt_service,
#         tts_service=tts_service,
#         system_prompt=system_prompt
#     )
    
#     voice_assistant.run()

# if __name__ == "__main__":
#     main()