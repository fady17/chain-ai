"""
The definitive, end-to-end RAG pipeline for the Mini-Chain project,
showcasing its own history with detailed, step-by-step logging.
"""
import os
import sys
import numpy as np
from dotenv import load_dotenv

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './src')))

# Import all the components from our framework
from minichain.core.types import Document
from minichain.text_splitters.implementations import RecursiveCharacterTextSplitter
from minichain.embeddings.azure import AzureOpenAIEmbeddings
from minichain.vector_stores.faiss import FAISSVectorStore
from minichain.chat_models.azure import AzureOpenAIChatModel
from minichain.prompts.implementations import PromptTemplate

# --- Enhanced Helper Functions ---
def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title.upper()} ".center(70, " "))
    print("=" * 70)

def print_vector_preview(vector: list, prefix=""):
    """Prints a preview of a vector (first 3 and last 3 elements)."""
    if not vector:
        print(f"{prefix}Vector Preview: [Empty Vector]")
        return
    preview = f"[{', '.join(map(str, np.round(vector[:3], 4)))}, ..., {', '.join(map(str, np.round(vector[-3:], 4)))}]"
    print(f"{prefix}Vector Preview: {preview} | Dimensions: {len(vector)}")

def main():
    """Main function to run the RAG pipeline."""
    
    print_header("Step 1: Initializing Mini-Chain Components")
    load_dotenv()
    print("✅ Environment variables loaded.")

    # Initialize components
    chat_model = AzureOpenAIChatModel(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") # type: ignore
    )
    embeddings = AzureOpenAIEmbeddings(
        deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") # type: ignore
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    vector_store = FAISSVectorStore(embeddings=embeddings)
    print("✅ All components initialized successfully.")

    # --- STEP 2: DEFINE & PROCESS SOURCE DOCUMENT ---
    print_header("Step 2: The 'Mini-Chain' Origin Story")
    
    # The real, authentic source text.
    source_text = """
    The 'Mini-Chain' project is a lightweight, LangChain-inspired framework developed
    in-house. The primary development was led by Fady, a dedicated intern, who built
    the core package from the ground up. The project's development was guided by insights
    and examples from large language models like Claude and Gemini.

    The architecture and code quality are under review by two senior figures.
    The first is Amin, a legendary AI engineer who was building advanced computer vision
    models back in 2017, long before the current LLM boom. His expertise ensures the
    system is robust and well-engineered.

    The second reviewer is Yehya, a respected open-source contributor and an Arch Linux user,
    known for his rigorous approach to software architecture and maintainability. His perspective
    is crucial for ensuring the framework is both powerful and easy to extend.
    """
    print(f"Source Text:\n---\n{source_text.strip()}\n---")
    
    documents = text_splitter.create_documents([source_text], [{"source": "project_readme.md"}])
    
    print(f"\n📄 Document split into {len(documents)} chunks:")
    for i, doc in enumerate(documents):
        print(f"  Chunk {i+1}: '{doc.page_content}'")

    # --- STEP 3: INDEX DOCUMENTS IN THE VECTOR STORE ---
    print_header("Step 3: Indexing the Story (Embedding in Detail)")
    
    texts_to_embed = [doc.page_content for doc in documents]
    print(f"Embedding {len(texts_to_embed)} text chunks...")
    
    embedded_vectors = embeddings.embed_documents(texts_to_embed)
    
    for i, vec in enumerate(embedded_vectors):
        print(f"\n--- Embedding for Chunk {i+1} ---")
        print(f"Text: '{texts_to_embed[i]}'")
        print_vector_preview(vec)
    
    vector_store.add_documents(documents)
    
    # --- STEP 4: ASK A QUESTION & RETRIEVE CONTEXT ---
    print_header("Step 4: Retrieval (Asking About the Team)")
    
    question = "Who is reviewing the code for the Mini-Chain project?"
    print(f"❓ User Question: '{question}'")
    
    print("\n--- Embedding the User Query ---")
    query_vector = embeddings.embed_query(question)
    print_vector_preview(query_vector)
    
    print("\n--- Performing Similarity Search ---")
    retrieved_docs = vector_store.similarity_search(question, k=2)
    
    print(f"\n🔍 Retrieved {len(retrieved_docs)} most relevant document chunks:")
    for i, doc in enumerate(retrieved_docs):
        print(f"  Result {i+1}: '{doc.page_content}'")
        
    # --- STEP 5: AUGMENT THE PROMPT ---
    print_header("Step 5: Augmentation")

    context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])
    rag_prompt_template = PromptTemplate(
        template="""
You are an AI assistant with perfect knowledge of the 'Mini-Chain' project's history.
Answer the user's question based ONLY on the context provided below.
Be clear and concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    )
    final_prompt = rag_prompt_template.format(context=context_string, question=question)
    
    print("✨ Final prompt being sent to the LLM:")
    print("---")
    print(final_prompt.strip())
    print("---")

    # --- STEP 6: GENERATE THE FINAL ANSWER ---
    print_header("Step 6: Generation")
    
    final_answer = chat_model.invoke(final_prompt)
    
    print("\n" + "*"*70)
    print(" FINAL ANSWER ".center(70, " "))
    print("*"*70)
    print(f"Question: {question}")
    print(f"Answer: {final_answer}")
    print("*"*70)
    
    # --- Let's ask another, more specific question! ---
    print_header("Bonus Round: Another Question")
    question_2 = "What is Amin known for?"
    print(f"❓ User Question: '{question_2}'")
    retrieved_docs_2 = vector_store.similarity_search(question_2, k=1)
    context_string_2 = "\n---\n".join([doc.page_content for doc in retrieved_docs_2])
    final_prompt_2 = rag_prompt_template.format(context=context_string_2, question=question_2)
    final_answer_2 = chat_model.invoke(final_prompt_2)
    print("\n" + "*"*70)
    print(f"Question: {question_2}")
    print(f"Answer: {final_answer_2}")
    print("*"*70)

if __name__ == "__main__":
    main()