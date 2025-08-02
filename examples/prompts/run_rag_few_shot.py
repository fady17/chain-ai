"""
RAG Pipeline with a FewShotPromptTemplate for Structured Data Extraction.

Purpose: This script demonstrates an advanced RAG pattern where a few-shot
prompt is used as a first step to understand and parse a user's query into
a structured format. This structured data then leads to a more precise
vector search.

Reviewer Focus: Note the two-step LLM call. The first call uses the
FewShotPromptTemplate to extract entities. The second call uses the retrieved
context to answer the question. This shows how prompts can be chained for
more sophisticated logic.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import socket
from minichain.core.types import Document
from minichain.text_splitters.token_splitter import TokenTextSplitter
from minichain.embeddings.local import LocalEmbeddings
from minichain.chat_models.local import LocalChatModel
from minichain.vector_stores.faiss import FAISSVectorStore
from minichain.prompts.implementations import PromptTemplate, FewShotPromptTemplate



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
    chat_model = LocalChatModel(temperature=0.0) # Zero temp for predictable extraction
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

    # --- Step 2: User Query & Few-Shot Extraction ---
    print_header("Step 2: Structured Extraction with FewShotPromptTemplate")
    user_question = "Tell me about the FAISS feature in the Mini-Chain library."
    print(f"User's Vague Question: '{user_question}'")
    
    # "Teach" the model how to extract structured data
    examples = [
        {"query": "How does the cloud search work in Mini-Chain?", "output": "{'project_name': 'Mini-Chain', 'feature': 'AzureAISearchVectorStore'}"},
        {"query": "I want to run a model locally with Mini-Chain.", "output": "{'project_name': 'Mini-Chain', 'feature': 'LocalChatModel'}"},
    ]
    example_template = PromptTemplate("Query: {query}\nOutput: {output}")
    
    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_template,
        suffix="Query: {input}\nOutput:",
        input_variables=["input"]
    )

    extraction_prompt = few_shot_template.format(input=user_question)
    print("\n--- Few-Shot Prompt for Extraction ---")
    print(extraction_prompt)
    print("------------------------------------")

    # First LLM call: Extract structured data
    extracted_data_str = chat_model.invoke(extraction_prompt)
    print(f"\nExtracted Data (as string): {extracted_data_str}")
    
    # A simple, unsafe eval for demo purposes. In production, use json.loads or a Pydantic parser.
    try:
        extracted_data = eval(extracted_data_str)
        search_query = f"{extracted_data.get('project_name', '')}: {extracted_data.get('feature', '')}"
        print(f"✅ Structured data parsed. Created precise search query: '{search_query}'")
    except:
        print("⚠️ Could not parse extracted data. Using original question for search.")
        search_query = user_question

    # --- Step 3: Precise RAG with Extracted Data ---
    print_header("Step 3: Precise RAG using Extracted Data")
    
    # Use the precise query for a better search
    retrieved_docs = vector_store.similarity_search(search_query, k=1)
    context = retrieved_docs[0].page_content
    print(f"Retrieved Context:\n---\n{context}\n---")
    
    # Second LLM call: Generate the final answer
    final_prompt = f"Using the following context, answer the user's question.\n\nContext: {context}\n\nQuestion: {user_question}\n\nAnswer:"
    final_answer = chat_model.invoke(final_prompt)
    
    print("\n" + "*" * 70)
    print(" FINAL ANSWER (FEW-SHOT PIPELINE) ".center(70, " "))
    print("*" * 70)
    print(final_answer)
    print("*" * 70)

if __name__ == "__main__":
    main()