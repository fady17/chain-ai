# Mini-Chain

**Mini-Chain** is a micro-framework for building applications with Large Language Models, inspired by LangChain. Its core principle is transparency and modularity, providing a "glass-box" design for engineers who value control and clarity.

## Core Features

- **Modular Components**: Swappable classes for Chat Models, Embeddings, Memory, and more.
- **Local & Cloud Ready**: Supports both local models (via LM Studio) and cloud services (Azure).
- **Modern Tooling**: Built with Pydantic for type-safety and Jinja2 for powerful templating.
- **GPU Acceleration**: Optional `faiss-gpu` support for high-performance indexing.

## Installation

```bash
pip install minichain-ai
#For Local FAISS (CPU) Support:
pip install minichain-ai[local]
#For NVIDIA GPU FAISS Support:
pip install minichain-ai[gpu]
#For Azure Support (Azure AI Search, Azure OpenAI):
pip install minichain-ai[azure]
#To install everything:
pip install minichain-ai[all]
```
Quick Start
Here is the simplest possible RAG pipeline with Mini-Chain:
```bash
# examples/01_hello_world_local.py
"""
Example 1: The absolute simplest way to use Mini-Chain.

This script demonstrates the most fundamental component: connecting to a
local language model (via LM Studio) and getting a response.
"""
import sys
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