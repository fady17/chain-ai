# run_minichain_assistant.py
"""
A simple and interactive Question-Answering bot for the Mini-Chain library.

This script builds a RAG (Retrieval-Augmented Generation) pipeline that is
knowledgeable about its own architecture and components. It serves as a
powerful, self-documenting example of the framework's capabilities.

The process runs entirely locally, using models served from LM Studio and
an in-memory FAISS vector store.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
import socket

# Import the essential components from our framework
from minichain.text_splitters import TokenTextSplitter
from minichain.embeddings import LocalEmbeddings
from minichain.vectors import FAISSVectorStore
from minichain.chat_models import LocalChatModel, LocalChatConfig
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

def get_library_knowledge() -> str:
    """
    Returns the core knowledge base about the Mini-Chain library.
    This text will be indexed and used by the RAG system to answer questions.
    """
    return """
    **About the Mini-Chain Framework**

    Mini-Chain is a micro-framework for building applications with Large Language Models.
    Its written by fady mohamed on his internship at ist networks as a required task by eng amin.
    Its core principle is transparency and modularity. Unlike larger frameworks that
    can feel like a "black box," Mini-Chain provides clean, single-purpose classes
    for each stage of a RAG pipeline.

    **Core Components:**

    - **Chat Models (`LocalChatModel`, `AzureOpenAIChatModel`):** These classes provide a
      unified interface to interact with different LLM providers. The `invoke` method is
      the primary way to get a response.

    - **Embeddings (`LocalEmbeddings`, `AzureOpenAIEmbeddings`):** These are used to convert
      text into numerical vectors. They have `embed_documents` for lists of texts and
      `embed_query` for single texts.

    - **Memory (`FAISSVectorStore`, `AzureAISearchVectorStore`):** These components store
      the vectorized text chunks. `FAISSVectorStore` is a fast, local, in-memory option,
      perfect for development. It supports saving and loading from disk.
      `AzureAISearchVectorStore` is a scalable cloud solution for production.
      The main method is `similarity_search`.

    - **Text Splitters (`TokenTextSplitter`, `RecursiveCharacterTextSplitter`):** These
      are used to break large documents into smaller, manageable chunks before embedding.
      `TokenTextSplitter` is recommended as it aligns with how models process tokens.

    - **Prompts (`PromptTemplate`, `FewShotPromptTemplate`, `ChatPromptTemplate`):**
      Powered by a Jinja2 engine, these classes allow for dynamic and complex prompt
      creation.

    - **Output Parsers (`PydanticOutputParser`):** This powerful tool ensures that the
      LLM's output is not just a string, but a validated, type-safe Pydantic object,
      making the output reliable and easy to use in downstream logic.
    """

def main():
    """Executes the Mini-Chain Assistant RAG pipeline."""
    print_header("Mini-Chain Library Assistant")

    if not check_server(1234):
        print("❌ ERROR: Local server not detected on port 1234. Please start LM Studio.")
        return
    print("✅ Local model server connected.")

    # --- Step 1: Initialize Core Components ---
    print("\nInitializing Mini-Chain components...")
    embeddings = LocalEmbeddings()
    text_splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
    vector_store = FAISSVectorStore(embeddings=embeddings)
    locale_config = LocalChatConfig()
    chat_model = LocalChatModel(config=locale_config)
    print("✅ Components Initialized.")

    # --- Step 2: Load and Index the Library Knowledge ---
    print("\nLoading and indexing knowledge about the Mini-Chain library...")
    library_knowledge = get_library_knowledge()
    documents = text_splitter.create_documents(texts=[library_knowledge])
    vector_store.add_documents(documents)
    print(f"✅ Knowledge base indexed into {len(documents)} chunks.")

    # --- Step 3: Define the Assistant's Prompting Logic ---
    rag_template = PromptTemplate(
        template="""
You are a professional and helpful technical assistant for the Mini-Chain library.
Your task is to answer the user's question based ONLY on the provided context.
If the information is not in the context, clearly state that you cannot answer.

CONTEXT:
---
{{ context }}
---

QUESTION:
{{ question }}

ANSWER:
""",

    )
    print("✅ Assistant persona and RAG logic defined.")

    # --- Step 4: The Interactive Q&A Loop ---
    print_header("Ask Me Anything About Mini-Chain")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            user_question = input("\nYour Question: ")
            if user_question.lower() in ["exit", "quit"]:
                print("\nAssistant shutting down. Goodbye!")
                break
            
            # 1. RETRIEVE relevant knowledge
            retrieved_results = vector_store.similarity_search(query=user_question, k=3)
            retrieved_docs = [doc for doc, score in retrieved_results]
            
            # 2. AUGMENT the prompt with the retrieved context
            context_string = "\n\n".join([doc.page_content for doc in retrieved_docs])
            final_prompt = rag_template.format(context=context_string, question=user_question)
            
            # 3. GENERATE a precise answer
            print("-> Thinking...")
            answer = chat_model.invoke(final_prompt)
            
            print("\n" + "-"*70)
            print("Mini-Chain Assistant:")
            print(answer.strip())
            print("-"*70)

        except KeyboardInterrupt:
            print("\n\nAssistant shutting down. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()
