# -*- coding: utf-8 -*-
"""
A Professional and Verbose Local RAG Pipeline Demonstration.

This script executes a sophisticated, end-to-end RAG process on a local machine,
providing detailed terminal output for every step to showcase the inner workings
of the Mini-Chain framework.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import socket
import numpy as np
from minichain.core.types import Document
from minichain.text_splitters.token_splitter import TokenTextSplitter
from minichain.embeddings.local import LocalEmbeddings
from minichain.chat_models.local import LocalChatModel
from minichain.vector_stores.faiss import FAISSVectorStore
from minichain.prompts.implementations import PromptTemplate


def print_header(title: str):
    print("\n" + "#" * 70)
    print(f"## {title.upper()} ".ljust(68) + "##")
    print("#" * 70)

def print_vector_preview(vector: list, prefix: str = ""):
    preview = f"[{', '.join(map(str, np.round(vector[:3], 4)))}, ..., {', '.join(map(str, np.round(vector[-3:], 4)))}]"
    print(f"{prefix}Vector Preview: {preview} | Dimensions: {len(vector)}")

def check_server(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(('localhost', port)) == 0

def main():
    print_header("Step 1: Initializing Local Pipeline")
    if not check_server(1234):
        print("❌ ERROR: Local server not detected on port 1234. Please start LM Studio.")
        return
    print("✅ Local server connection verified.")

    chat_model = LocalChatModel()
    embeddings = LocalEmbeddings()
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
    vector_store = FAISSVectorStore(embeddings=embeddings)
    print("✅ All local components initialized successfully.")

    print_header("Step 2: Processing Source Document")
    adr_text = """
    Project: Mini-Chain Framework
    ADR-003: Validation of a Modular, Local-First Architecture

    Author: Fady, Intern Developer
    Reviewer: Amin, Senior AI Engineer

    1. Context:
    Modern LLM frameworks often hide their internal logic, creating a "black box"
    that hinders deep customization and debugging. The goal of Mini-Chain is to
    provide a transparent, modular alternative. The core development was undertaken
    by Fady, focusing on creating swappable components for each stage of the RAG pipeline.

    2. Decision:
    The architecture's validity is confirmed through the expert review of Amin.
    His extensive experience, including building computer vision systems in 2017,
    provides a crucial, non-LLM-centric perspective. His approval verifies that the
    decoupled design (splitter, embedder, vector store, model) is not just a
    novelty, but a robust engineering principle that ensures long-term maintainability
    and flexibility, independent of any single model provider.
    """
    print(f"Source Document:\n---\n{adr_text.strip()}\n---")
    
    documents = text_splitter.create_documents([adr_text], [{"source": "ADR-003"}])
    print(f"\n📄 Document split into {len(documents)} chunks:")
    for i, doc in enumerate(documents):
        print(f"  Chunk {i+1}: '{doc.page_content}'")

    print_header("Step 3: Indexing (Local Embedding)")
    print(f"Embedding {len(documents)} chunks using local Nomic model...")
    vector_store.add_documents(documents)
    print("✅ Documents embedded and stored in local FAISS index.")

    print_header("Step 4: Retrieval and Generation")
    question = "What is the primary value of Amin's review for the project's architecture?"
    print(f"User Query: '{question}'\n")

    print("--- 4a: Embedding the Query ---")
    query_vector = embeddings.embed_query(question)
    print_vector_preview(query_vector, prefix="Query ")
    
    print("\n--- 4b: Performing Similarity Search ---")
    retrieved_docs = vector_store.similarity_search(question, k=2)
    print(f"Retrieved {len(retrieved_docs)} most relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs):
        print(f"  Retrieved Chunk {i+1}: '{doc.page_content}'")

    context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    print("\n--- 4c: Building the Final Prompt ---")
    assistant_prompt = PromptTemplate(
        template="""You are a funny professional technical assistant. Answer the user's question based on the provided context strictly while keep it short and simple, no buzzwords.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    )
    final_prompt = assistant_prompt.format(context=context_string, question=question)
    print("Final prompt being sent to the local LLM:\n---")
    print(final_prompt)
    print("---")

    print("\n--- 4d: Generating the Final Answer ---")
    final_answer = chat_model.invoke(final_prompt)
    
    print("\n" + "*" * 70)
    print(" ANALYSIS COMPLETE (LOCAL) ".center(70, " "))
    print("*" * 70)
    print(f"Question: {question}")
    print("\nGenerated Response:")
    print(final_answer.strip())
    print("*" * 70)

if __name__ == "__main__":
    main()