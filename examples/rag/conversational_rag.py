#!/usr/bin/env python3
"""
Simple example using the new RAG runner.
This replaces your original conversational_rag.py with much less code.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.rag_runner import create_rag_from_texts

# Define your knowledge base
knowledge_texts = [
    "The chain library's core principle is transparency and modularity. "
    "It was created by fady mohamed as a task from engineer amin at ist networks as a task on his internship.",
    
    "Fady Mohamed is an intern at IST Networks working under Engineer Amin. "
    "He created the chain library as part of his internship project.",
    
    "IST Networks is the company where Fady is doing his internship. "
    "Engineer Amin is his supervisor who assigned the chain development task."
]

# System prompt
system_prompt = """You are a helpful assistant with access to a knowledge base and conversation memory.

For conversation questions (about our chat history), use our conversation above.
For knowledge questions (about chain, Fady, IST Networks, etc.), use the provided context.
Be accurate about what was actually said in our conversation."""

# Create and run RAG - that's it!
if __name__ == "__main__":
    rag = create_rag_from_texts(
        knowledge_texts=knowledge_texts,
        system_prompt=system_prompt,
        chunk_size=100,  # Small chunks for this simple example
        chunk_overlap=0,
        retrieval_k=2,
        debug=True
    )
    
    rag.run_chat()
