# -*- coding: utf-8 -*-
"""
A Professional and Verbose Cloud RAG Pipeline Demonstration on Azure.

This script executes a sophisticated, end-to-end RAG process using a full suite
of managed Azure services, providing detailed terminal output for every step to
showcase the framework's power and transparency.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import time
from dotenv import load_dotenv
import numpy as np
from minichain.core.types import Document
from minichain.text_splitters.token_splitter import TokenTextSplitter
from minichain.embeddings.azure import AzureOpenAIEmbeddings
from minichain.chat_models.azure import AzureOpenAIChatModel
from minichain.vector_stores.azure_ai_search import AzureAISearchVectorStore
from minichain.prompts.implementations import PromptTemplate


def print_header(title: str):
    print("\n" + "#" * 70)
    print(f"## {title.upper()} ".ljust(68) + "##")
    print("#" * 70)

def print_vector_preview(vector: list, prefix: str = ""):
    preview = f"[{', '.join(map(str, np.round(vector[:3], 4)))}, ..., {', '.join(map(str, np.round(vector[-3:], 4)))}]"
    print(f"{prefix}Vector Preview: {preview} | Dimensions: {len(vector)}")

def main():
    print_header("Step 1: Initializing Azure Cloud Pipeline")
    load_dotenv()
    if not os.getenv("AZURE_AI_SEARCH_ENDPOINT"):
        print("❌ ERROR: Azure credentials not found in .env file.")
        return
    print("✅ Azure credentials verified.")

    chat_model = AzureOpenAIChatModel(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")) # type: ignore
    embeddings = AzureOpenAIEmbeddings(deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")) # type: ignore
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
    
    index_name = f"professional-demo-{int(time.time())}"
    vector_store = AzureAISearchVectorStore(embeddings=embeddings, index_name=index_name)
    print(f"Using temporary Azure AI Search index: '{index_name}'")
    print("✅ All Azure components initialized successfully.")

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

    print_header("Step 3: Indexing (Azure Embedding & Storage)")
    print(f"Embedding {len(documents)} chunks and uploading to Azure AI Search...")
    vector_store.add_documents(documents)
    print("\nWaiting 5 seconds for cloud indexing...")
    time.sleep(5)
    print("✅ Document indexed in Azure AI Search.")

    print_header("Step 4: Retrieval and Generation via Azure")
    question = "What is the primary value of Amin's review for the project's architecture?"
    print(f"User Query: '{question}'\n")

    print("--- 4a: Embedding the Query with Azure OpenAI ---")
    query_vector = embeddings.embed_query(question)
    print_vector_preview(query_vector, prefix="Query ")
    
    print("\n--- 4b: Performing Similarity Search in Azure AI Search ---")
    retrieved_docs = vector_store.similarity_search(question, k=2)
    print(f"Retrieved {len(retrieved_docs)} most relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs):
        print(f"  Retrieved Chunk {i+1}: '{doc.page_content}' (Score: {doc.metadata.get('score'):.4f})")

    context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    print("\n--- 4c: Building the Final Prompt ---")
    assistant_prompt = PromptTemplate(
        template="""You are a precise and professional AI technical assistant. Answer the user's question based strictly on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    )
    final_prompt = assistant_prompt.format(context=context_string, question=question)
    print("Final prompt being sent to Azure OpenAI:\n---")
    print(final_prompt)
    print("---")

    print("\n--- 4d: Generating the Final Answer ---")
    final_answer = chat_model.invoke(final_prompt)
    
    print("\n" + "*" * 70)
    print(" ANALYSIS COMPLETE (CLOUD) ".center(70, " "))
    print("*" * 70)
    print(f"Question: {question}")
    print("\nGenerated Response:")
    print(final_answer.strip())
    print("*" * 70)

    print_header("Step 5: Cleaning Up Cloud Resources")
    try:
        vector_store.index_client.delete_index(index_name)
        print(f"✅ Successfully deleted temporary index '{index_name}'.")
    except Exception as e:
        print(f"⚠️ Could not delete index '{index_name}'. Please clean up manually in Azure.")

if __name__ == "__main__":
    main()
# """
# A fully cloud-native, end-to-end RAG pipeline with verbose, step-by-step
# logging for a comprehensive demonstration.

# This script showcases a Retrieval-Augmented Generation system on Azure,
# using the team's origin story as its knowledge base.
# """

# import time
# import numpy as np
# from dotenv import load_dotenv

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# # Import the necessary components from our Mini-Chain framework
# from minichain.core.types import Document
# from minichain.text_splitters.implementations import RecursiveCharacterTextSplitter
# from minichain.embeddings.azure import AzureOpenAIEmbeddings
# from minichain.chat_models.azure import AzureOpenAIChatModel
# from minichain.vector_stores.azure_ai_search import AzureAISearchVectorStore
# from minichain.prompts.implementations import PromptTemplate

# # --- Helper Functions for Clean Output ---

# def print_header(title: str):
#     """Prints a formatted header to the console."""
#     print("\n" + "=" * 70)
#     print(f" {title.upper()} ".center(70, " "))
#     print("=" * 70)

# def print_vector_preview(vector: list, prefix: str = ""):
#     """Prints a readable preview of a high-dimensional vector."""
#     if not vector:
#         print(f"{prefix}Vector Preview: [Empty Vector]")
#         return
#     preview = f"[{', '.join(map(str, np.round(vector[:3], 4)))}, ..., {', '.join(map(str, np.round(vector[-3:], 4)))}]"
#     print(f"{prefix}Vector Preview: {preview} | Dimensions: {len(vector)}")

# def main():
#     """
#     Executes the main cloud-native RAG pipeline demonstration.
#     """
#     # --- PRE-FLIGHT CHECK ---
#     print_header("Pre-flight Check: Verifying Azure Credentials")
#     load_dotenv()
    
#     required_vars = [
#         "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME",
#         "AZURE_OPENAI_ENDPOINT_EMBEDDINGS", "AZURE_OPENAI_EMBEDDINGS_API_KEY", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
#         "AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_ADMIN_KEY"
#     ]
#     if not all(os.getenv(var) for var in required_vars):
#         print("❌ ERROR: Missing one or more required Azure environment variables.")
#         return

#     print("✅ All required Azure credentials found.")
    
#     # --- STEP 1: INITIALIZE CLOUD-NATIVE COMPONENTS ---
#     print_header("Step 1: Initializing Azure-Native Mini-Chain Components")

#     azure_chat_model = AzureOpenAIChatModel(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")) # type: ignore
#     azure_embeddings = AzureOpenAIEmbeddings(deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")) # type: ignore
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)

#     index_name = f"minichain-demo-index-{int(time.time())}"
#     print(f"Using temporary Azure AI Search index: '{index_name}'")

#     vector_store = AzureAISearchVectorStore(embeddings=azure_embeddings, index_name=index_name)
#     print("\n✅ All cloud components initialized successfully.")

#     # --- STEP 2: THE TEAM'S ORIGIN STORY ---
#     print_header("Step 2: Processing the Team's Origin Story")

#     team_story = """
#     The philosophy behind the 'Mini-Chain' framework is rooted in the unique experiences of its team.
#     The project is being developed by Fady, a passionate intern, but its soul is shaped by its reviewers.

#     Amin, one of the senior reviewers, brings a depth of experience that predates the modern LLM era.
#     He was a true pioneer, building and deploying computer vision models back in 2017. In those days,
#     success meant bleeding on Stack Overflow for hours, wrestling with libraries like Theano and Caffe.

#     The other key reviewer, Yehya, embodies the spirit of open-source and true system mastery.
#     As a long-time Arch Linux user, he understands the power of a minimalist, well-configured system
#     where every component is chosen for a reason, championing the Linux philosophy of 'do one thing well.'
#     """
#     print(f"Source Text (The Team Story):\n---\n{team_story.strip()}\n---")
    
#     documents = text_splitter.create_documents([team_story], [{"source": "team_philosophy.md"}])
#     print(f"\n📄 Document split into {len(documents)} chunks.")

#     # --- STEP 3: INDEXING THE STORY ON AZURE (WITH VERBOSE LOGGING) ---
#     print_header("Step 3: Indexing with Azure AI (Embedding in Detail)")
    
#     # We'll manually call the embedding function here to log the output,
#     # even though the `add_documents` method would do this internally.
#     # This is purely for demonstration visibility.
#     texts_to_embed = [doc.page_content for doc in documents]
#     print(f"Sending {len(texts_to_embed)} text chunks to Azure OpenAI for embedding...")
    
#     embedded_vectors = azure_embeddings.embed_documents(texts_to_embed)
    
#     for i, vec in enumerate(embedded_vectors):
#         print(f"\n--- Embedding for Chunk {i+1} ---")
#         print(f"Text: '{texts_to_embed[i]}'")
#         print_vector_preview(vec, prefix="Vector: ")
    
#     print("\n--- Uploading documents and vectors to Azure AI Search ---")
#     vector_store.add_documents(documents)
    
#     print("\nWaiting 5 seconds for indexing to complete...")
#     time.sleep(5)
#     print("✅ Story indexed in Azure AI Search.")

#     # --- STEP 4: RETRIEVAL (WITH VERBOSE LOGGING) ---
#     print_header("Step 4: Retrieval from Azure AI Search")
    
#     question = "What makes Amin's experience so valuable to the project?"
#     print(f"❓ Query: '{question}'")

#     print("\n--- Embedding the query with Azure OpenAI ---")
#     query_vector = azure_embeddings.embed_query(question)
#     print("Query Text: '{}'".format(question))
#     print_vector_preview(query_vector, prefix="Query Vector: ")

#     print("\n--- Performing similarity search in Azure AI Search ---")
#     retrieved_docs = vector_store.similarity_search(question, k=1)
    
#     print(f"🔍 Found {len(retrieved_docs)} relevant chunk(s):")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"  [{i+1}]: '{doc.page_content}' (Similarity Score: {doc.metadata.get('score'):.4f})")

#     # --- STEP 5: AUGMENTATION (BUILDING THE PROMPT) ---
#     print_header("Step 5: Augmenting Prompt for Azure OpenAI Model")

#     context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])
#     rag_prompt = PromptTemplate(
#         template="""
# You are an AI assistant telling the story of the Mini-Chain project team.
# Answer the user's question based ONLY on the provided context.
# Capture the spirit and essence of the person mentioned.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#     )
#     final_prompt = rag_prompt.format(context=context_string, question=question)
#     print("✨ Final prompt prepared for Azure OpenAI:\n---")
#     print(final_prompt.strip())
#     print("---")

#     # --- STEP 6: GENERATION (GETTING THE ANSWER) ---
#     print_header("Step 6: Generating the Answer via Azure")
    
#     final_answer = azure_chat_model.invoke(final_prompt)
    
#     print("\n" + "*" * 70)
#     print(" THE STORY OF AMIN ".center(70, " "))
#     print("*" * 70)
#     print(f"Question: {question}")
#     print(f"Answer: {final_answer}")
#     print("*" * 70)
    
#     # --- STEP 7: CLEANUP ---
#     print_header("Step 7: Cleaning Up Cloud Resources")
#     try:
#         vector_store.index_client.delete_index(index_name)
#         print(f"✅ Successfully deleted temporary index '{index_name}'.")
#     except Exception as e:
#         print(f"⚠️ Failed to delete index '{index_name}': {e}")
#         print("   Please delete it manually from the Azure Portal to avoid costs.")


# if __name__ == "__main__":
#     main()