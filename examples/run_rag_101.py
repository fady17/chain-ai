"""
The simplest possible Retrieval-Augmented Generation (RAG) pipeline
built with the Mini-Chain framework.

This script demonstrates the five core steps of RAG in a minimal,
easy-to-understand way:
1.  Load & Split: A source text is loaded and split into chunks.
2.  Embed & Store: The chunks are converted to vectors and stored in memory.
3.  Retrieve: A user question is used to find the most relevant chunks.
4.  Augment: The question and retrieved context are combined into a prompt.
5.  Generate: An LLM answers the question based on the prompt.

This entire process runs locally, using models served from LM Studio and
an in-memory FAISS vector store.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import socket


# Import the essential components from our framework
from minichain.text_splitters import TokenTextSplitter
from minichain.embeddings import LocalEmbeddings
from minichain.memory import FAISSVectorStore
from minichain.chat_models import LocalChatModel
from minichain.prompts import PromptTemplate

def print_header(title: str):
    """Prints a formatted header for clear sections."""
    print("\n" + "=" * 70)
    print(f" {title.upper()} ".center(70, " "))
    print("=" * 70)

def check_server(port: int) -> bool:
    """Verifies that the local model server is running."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        return s.connect_ex(('localhost', port)) == 0

def main():
    """Executes the simple RAG pipeline."""

    # --- Pre-flight Check ---
    print_header("Step 0: Pre-flight Check")
    if not check_server(1234):
        print("❌ ERROR: Local server not detected on port 1234.")
        print("Please start the server in LM Studio and load a chat model and an embedding model.")
        return
    print("✅ Local server detected. The pipeline is a go.")

    # --- Step 1: Initialize Core Components ---
    print_header("Step 1: Initializing Mini-Chain Components")
    
    # All components are the simplest, local-first options.
    embeddings = LocalEmbeddings()
    text_splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=15)
    vector_store = FAISSVectorStore(embeddings=embeddings)
    chat_model = LocalChatModel()
    
    print("✅ Components Initialized (Embeddings, Splitter, Vector Store, Chat Model)")

    # --- Step 2: Load, Split, and Index the Data ---
    print_header("Step 2: Loading and Indexing Knowledge")

    # A simple, fictional knowledge base. The model will not know this information.
    source_text = """
    The Chrono-Camera is a fictional device invented by Dr. Aris Thorne in 2042.
    It does not take pictures of the present, but rather captures faint temporal
    echoes from the past. The camera requires a rare crystal, "Aethelred," to
    focus these echoes. Its primary limitation is that it cannot capture images
    from before the year 1600 due to quantum instability. For support, contact
    support@thornetechnologies.ai.
    """
    
    print("Splitting the source text into manageable chunks...")
    documents = text_splitter.create_documents(texts=[source_text])
    
    print(f"Indexing {len(documents)} chunks into the FAISS vector store...")
    vector_store.add_documents(documents)
    
    print("✅ Knowledge base is loaded and ready.")

    # --- Step 3: Define the RAG Prompt ---
    print_header("Step 3: Defining the RAG Prompt")

    # This prompt template is the core of our RAG logic.
    rag_template = PromptTemplate(
        template="""
You are a helpful assistant. Answer the user's question based ONLY on the
following context. If the context does not contain the answer, say that you
do not have enough information.

CONTEXT:
{{ context }}

QUESTION:
{{ question }}

ANSWER:
"""
    )
    print("✅ RAG prompt template created.")

    # --- Step 4: The Interactive Q&A Loop ---
    print_header("Step 4: Interactive Q&A")
    print("The RAG system is ready. Ask a question about the Chrono-Camera.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_question = input("\nYour Question: ")
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting RAG session. Goodbye!")
                break
            
            # 1. RETRIEVE
            print("-> Retrieving relevant documents from the vector store...")
            retrieved_results = vector_store.similarity_search(query=user_question, k=2)
            # We extract just the Document objects from the (Document, score) tuples
            retrieved_docs = [doc for doc, score in retrieved_results]
            
            # 2. AUGMENT
            context_string = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 3. GENERATE
            print("-> Generating an answer with the LLM...")
            final_prompt = rag_template.format(context=context_string, question=user_question)
            
            answer = chat_model.invoke(final_prompt)
            
            print("\n" + "-"*70)
            print("AI Answer:")
            print(answer.strip())
            print("-"*70)

        except KeyboardInterrupt:
            print("\nExiting RAG session. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()