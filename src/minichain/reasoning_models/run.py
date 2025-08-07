# # src/minichain/reasoning_models/run.py
# """
# Provides a high-level, interactive runner function for reasoning models.
# """

# from typing import List, Dict, Union
# from .base import BaseReasoningModel
# from ..core.types import SystemMessage, HumanMessage, AIMessage, BaseMessage

# def run_reasoning_chat(model: BaseReasoningModel, system_prompt: Union[str, None] = None):
#     """
#     Starts an interactive, streaming chat session with a reasoning model.
#     This function mirrors the `run_chat` interface for consistency.
#     """
#     print("\n" + "="*50)
#     print(" Mini-Chain Reasoning Chat ".center(50, " "))
#     print("="*50)
#     print("Enter your message. Type 'exit' or 'quit' to end the session.")
    
#     history: List[Dict[str, str]] = []
    
#     if system_prompt:
#         history.append({"role": "system", "content": system_prompt})
        
#     while True:
#         try:
#             user_input = input("\n[ You ] -> ")
#             if user_input.lower() in ["exit", "quit"]:
#                 print("\n🤖 Session ended. Goodbye!")
#                 break
                
#             history.append({"role": "user", "content": user_input})
            
#             messages_for_llm: List[BaseMessage] = [
#                 SystemMessage(content=msg["content"]) if msg["role"] == "system"
#                 else HumanMessage(content=msg["content"]) if msg["role"] == "user"
#                 else AIMessage(content=msg["content"])
#                 for msg in history
#             ]
            
#             print("[ AI  ] -> ", end="", flush=True)
            
#             full_response = ""
#             for chunk in model.stream(messages_for_llm):
#                 print(chunk, end="", flush=True)
#                 full_response += chunk
#             print() # for newline
            
#             history.append({"role": "assistant", "content": full_response})

#         except KeyboardInterrupt:
#             print("\n\n🤖 Session ended. Goodbye!")
#             break
#         except Exception as e:
#             print(f"\nAn error occurred: {e}")
#             break