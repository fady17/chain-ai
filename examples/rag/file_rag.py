import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.rag_runner import create_rag_from_files

# Load knowledge from files
rag = create_rag_from_files(
    file_paths=["/Users/fady/Desktop/internship/langchain-clone/chainforge-ai/examples/rag/docs/manual.txt"], #, "README.md"
    system_prompt="You are an anttorney ai assistant that helps an egypytion attorney to search through the documents for citation , always respond within the provided text and always respond on egtption arabic.",
    chunk_size=500,
    retrieval_k=3
)
rag.run_chat()