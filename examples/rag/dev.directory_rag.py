import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.rag_runner import create_rag_from_directory

# Load all Python files from a directory
rag = create_rag_from_directory(
    directory="/Users/fady/Desktop/internship/langchain-clone/chainforge-ai/examples/rag/docs",
    file_extensions=['.txt'],
    system_prompt="You are a helpful assistant."
)
rag.run_chat()