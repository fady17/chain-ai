# examples/05_persistent_rag_local.py
"""
Example 5: Local RAG with persistence.

This script shows how to save a FAISS vector store to disk and load it
back, avoiding the need to re-index documents every time.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.text_splitters import TokenTextSplitter
from minichain.embeddings import LocalEmbeddings
from minichain.memory import FAISSVectorStore
from minichain.core.types import Document

# --- Configuration ---
INDEX_PATH = "./my_faiss_index"
embeddings = LocalEmbeddings()

if not os.path.exists(INDEX_PATH):
    print("Index not found. Creating and saving a new one...")
    # 1. Create and save the index if it doesn't exist
    source_text = "The Mini-Chain project's lead reviewer is Amin."
    docs = TokenTextSplitter().create_documents(texts=[source_text])
    vector_store = FAISSVectorStore.from_documents(docs, embeddings)
    vector_store.save_local(INDEX_PATH)
    print(f"✅ Index saved to {INDEX_PATH}")
else:
    print(f"Loading existing index from {INDEX_PATH}...")
    # 2. Load the index if it exists
    vector_store = FAISSVectorStore.load_local(INDEX_PATH, embeddings)
    print("✅ Index loaded.")

# 3. Use the loaded store for a search
question = "Who is the lead reviewer?"
results = vector_store.similarity_search(question, k=1)

print(f"\nQuestion: {question}")
print(f"Retrieved Context: {results[0][0].page_content}")