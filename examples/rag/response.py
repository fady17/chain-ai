import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.rag_runner import create_rag_from_directory

# Load all text files from a directory
rag = create_rag_from_directory(
    directory="/Users/fady/Desktop/internship/langchain-clone/chainforge-ai/examples/rag/docs",
    file_extensions=['.txt'],
    system_prompt="You are a helpful assistant."
)

# Now you can query with single messages!
response = rag.query("What is the main topic discussed in the documents?")
print(response)
