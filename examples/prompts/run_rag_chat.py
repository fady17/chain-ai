"""
RAG Pipeline with a ChatPromptTemplate for a Persona-Driven Agent.

Purpose: This script demonstrates how to build a conversational RAG agent
with a distinct personality. The `ChatPromptTemplate` is used to structure
the final prompt for a chat model, separating the system's instructions,
the retrieved context, and the user's question into a clean, multi-message format.

Reviewer Focus: Observe how the `ChatPromptTemplate` creates a list of
messages. This is the modern, preferred way to interact with chat-optimized
LLMs, allowing for clearer instructions and better persona management than a
single, large string prompt.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import socket
from minichain.core.types import Document, SystemMessage, HumanMessage
from minichain.text_splitters.token_splitter import TokenTextSplitter
from minichain.embeddings.local import LocalEmbeddings
from minichain.chat_models.local import LocalChatModel
from minichain.vector_stores.faiss import FAISSVectorStore
from minichain.prompts.implementations import ChatPromptTemplate

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def print_header(title: str):
    print("\n" + "#" * 70)
    print(f"## {title.upper()} ".ljust(68) + "##")
    print("#" * 70)

def check_server(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(('localhost', port)) == 0

def main():
    print_header("Initializing Local RAG Pipeline")
    if not check_server(1234):
        print("❌ ERROR: Local server not detected. Please start LM Studio.")
        return
    chat_model = LocalChatModel(temperature=0.5)
    embeddings = LocalEmbeddings()
    text_splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=20)
    vector_store = FAISSVectorStore(embeddings=embeddings)
    print("✅ Local components initialized.")

    # --- Step 1: Ingesting Technical Documentation ---
    print_header("Step 1: Ingesting Technical Docs")
    docs_text = """
    In the Mini-Chain project, the FAISSVectorStore component provides an efficient, in-memory
    solution for similarity search, ideal for local development. It uses the `faiss-cpu` library.
    
    The AzureAISearchVectorStore component in Mini-Chain offers a scalable, cloud-native
    alternative. It requires credentials and an index name for setup and is suited for production.
    
    The LocalChatModel component of the Mini-Chain framework allows users to connect to any
    OpenAI-compatible API, such as one served by LM Studio, for local inference.
    """
    documents = text_splitter.create_documents([docs_text], [{"source": "tech_docs.md"}])
    vector_store.add_documents(documents)
    print("✅ Technical documentation indexed in FAISS.")

    # --- Step 2: User Query and Standard RAG ---
    print_header("Step 2: Standard Retrieval for Chat Agent")
    user_question = "What's the difference between the FAISS and Azure vector stores in your framework?"
    print(f"User Question: '{user_question}'")
    
    retrieved_docs = vector_store.similarity_search(user_question, k=2)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(f"\nRetrieved Context:\n---\n{context}\n---")

    # --- Step 3: Generation with ChatPromptTemplate ---
    print_header("Step 3: Persona-Driven Generation with ChatPromptTemplate")
    
    # Define the conversational structure
    chat_template = ChatPromptTemplate(
        messages=[
            {"role": "system", "content": "You are the 'Mini-Chain Support Bot', a helpful and friendly AI assistant. Your goal is to answer user questions clearly and concisely based on the provided technical context."},
            {"role": "user", "content": "I have a question: {question}\n\nHere is some context I found:\n{context}"}
        ]
    )
    
    # Format the template into a list of message dictionaries
    formatted_messages = chat_template.format(question=user_question, context=context)
    
    print("\n--- Final Structured Messages for Chat Model ---")
    import json
    print(json.dumps(formatted_messages, indent=2))
    print("------------------------------------------------")
    
    # Convert to our framework's message objects for the final LLM call
    final_message_objects = [
        SystemMessage(content=formatted_messages[0]['content']),
        HumanMessage(content=formatted_messages[1]['content'])
    ]
    
    final_answer = chat_model.invoke(final_message_objects)

    print("\n" + "*" * 70)
    print(" FINAL ANSWER (CHAT PIPELINE) ".center(70, " "))
    print("*" * 70)
    print(final_answer)
    print("*" * 70)

if __name__ == "__main__":
    main()