# examples/01_hello_world_local.py
"""
Example 1: The absolute simplest way to use Mini-Chain.

This script demonstrates the most fundamental component: connecting to a
local language model (via LM Studio) and getting a response.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.chat_models import LocalChatModel

# 1. Initialize the LocalChatModel
# This connects to your LM Studio server running on the default port.
try:
    local_model = LocalChatModel()
    print("✅ Successfully connected to local model server.")
except Exception as e:
    print(f"❌ Could not connect to local model server. Is LM Studio running? Error: {e}")
    sys.exit(1)

# 2. Define a prompt and get a response
prompt = "In one sentence, what is the purpose of a CPU?"
print(f"\nUser Prompt: {prompt}")

response = local_model.invoke(prompt)

print("\nAI Response:")
print(response)