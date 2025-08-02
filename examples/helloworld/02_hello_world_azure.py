# examples/02_hello_world_azure.py
"""
Example 2: Using a cloud-based model with Mini-Chain.

This script shows how to swap the local model for a managed Azure OpenAI
deployment with a single line of code change.
"""
import sys
import os

from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.chat_models import AzureOpenAIChatModel

# 1. Load Azure credentials from your .env file
load_dotenv()
if not os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
    print("❌ Azure credentials not found. Please set them in your .env file.")
    sys.exit(1)

# 2. Initialize the AzureOpenAIChatModel
# Note: The component is different, but the usage is identical to the local model.
azure_model = AzureOpenAIChatModel(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
)
print("✅ Successfully initialized Azure OpenAI chat model.")

# 3. Define a prompt and get a response
prompt = "In one sentence, what is the purpose of a GPU?"
print(f"\nUser Prompt: {prompt}")

response = azure_model.invoke(prompt)

print("\nAI Response:")
print(response)