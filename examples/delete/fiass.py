import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.core.types import Document
from chain.embeddings.local import LocalEmbeddings
from chain.vectors import FAISSVectorStore

# This script will now work
embeddings = LocalEmbeddings()
documents = [Document(page_content="hello world")]
vector_store = FAISSVectorStore.from_documents(documents, embeddings)

print("Vector store created successfully!")
print(vector_store)