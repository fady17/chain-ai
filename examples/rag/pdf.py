import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.rag_runner import create_smart_rag

# Load PDF and create RAG
rag = create_smart_rag(knowledge_files=["/Users/fady/Desktop/internship/langchain-clone/chainforge-ai/examples/rag/docs/ww.pdf"])

# Query the PDF
response = rag.run_chat()
