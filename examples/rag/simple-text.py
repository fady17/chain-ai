import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.rag_runner import create_rag_from_texts

# Your knowledge base
knowledge_texts = [
    "The Minichain library is a lightweight LangChain alternative.",
    "It was created by Fady Mohamed during his internship at IST Networks."
]

# Create and run - just 2 lines!
rag = create_rag_from_texts(knowledge_texts, system_prompt="You are a helpful assistant.")
rag.run_chat()