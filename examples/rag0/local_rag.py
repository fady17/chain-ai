import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from minichain.core import Document
from minichain.chat_models import LocalChatModel, LocalChatConfig
from minichain.embeddings import LocalEmbeddings
from minichain.vectors import FAISSVectorStore
from minichain.prompts import PromptTemplate
from minichain.text_splitters import RecursiveCharacterTextSplitter

class PipelineLogger:
    """A simple, reusable logging utility for RAG pipeline visualization."""
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    def header(self, title: str):
        """Prints a main header."""
        print(f"\n{self.CYAN}{self.BOLD}{'='*80}{self.RESET}")
        print(f"{self.CYAN}{self.BOLD}{title.center(80)}{self.RESET}")
        print(f"{self.CYAN}{self.BOLD}{'='*80}{self.RESET}")

    def log(self, step: str, details: str):
        """Logs a specific step with details."""
        print(f"\n{self.YELLOW}STEP: {step}{self.RESET}")
        print(f"{self.GREEN}{details}{self.RESET}")
        print("-" * 80)

# --- Initialize Logger ---
logger = PipelineLogger()
logger.header("STARTING RAG PIPELINE DIAGNOSTIC RUN")

# --- 1. Define Source Material ---
long_text = (
    "Mini-Chain is a micro-framework for building applications with Large Language Models inspired by langchain."
    "Its written by fady mohamed on his internship at ist networks as a required task by eng amin."
    "Mini-Chain provides single-purpose classes for each stage of a RAG pipeline."
   
  
)

# --- 2. Initialize Core Components ---
logger.log("COMPONENT INITIALIZATION", "Initializing Embeddings, Splitter, VectorStore, and LLM.")
embeddings = LocalEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
vector_store = FAISSVectorStore(embeddings=embeddings)
llm = LocalChatModel(config=LocalChatConfig(model_name="lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"))

# --- 3. Process Documents: Create -> Split -> Index ---

# Step 3a: Create Document objects
source_documents = [
    Document(page_content=long_text, metadata={"source": "history_of_ai.txt"})
]
logger.log("DOCUMENT LOADING", f"Loaded {len(source_documents)} source document(s).")

# Step 3b: Split documents into chunks
split_docs = text_splitter.split_documents(source_documents)
chunk_details = "\n---\n".join([f'Chunk {i+1}: "{doc.page_content}"' for i, doc in enumerate(split_docs)])
logger.log("TEXT SPLITTING", f"Source document split into {len(split_docs)} chunks.\n\n{chunk_details}")

# Step 3c: Add chunks to the vector store
vector_store.add_documents(split_docs)
# This is a critical check: does the FAISS index have vectors in it?
index_size = vector_store.index.ntotal if vector_store.index else 0
logger.log("INDEXING", f"Vector store now contains {index_size} indexed document chunks.")

# --- 4. Define the RAG Prompt Template ---
rag_template = PromptTemplate(
    template="""You are a helpful Q&A assistant. Answer based ONLY on the context.

CONTEXT:
---
{{ context }}
---

QUESTION: {{ question }}"""
)

# --- 5. Execute the RAG Pipeline ---
question = "Who created minichain and why?"
logger.header(f"EXECUTING PIPELINE FOR QUESTION: \"{question}\"")

# Step A: RETRIEVE relevant documents
logger.log("RETRIEVAL", "Performing similarity search in the vector store.")
retrieved_results = vector_store.similarity_search(query=question, k=2)

# This is the MOST IMPORTANT log. It shows us what the retriever found.
if not retrieved_results:
    retrieval_log_details = f"{logger.RED}❌ FAILED: No documents were retrieved. The context will be empty.{logger.RESET}"
else:
    retrieval_log_details = f"✅ SUCCESS: Retrieved {len(retrieved_results)} document(s).\n"
    for i, (doc, score) in enumerate(retrieved_results):
        retrieval_log_details += f'\nDoc {i+1} (Score: {score:.4f}): "{doc.page_content}"'
logger.log("RETRIEVAL RESULTS", retrieval_log_details)

# Step B: AUGMENT the prompt by creating the context string.
context_string = "\n\n".join([doc.page_content for doc, score in retrieved_results])
logger.log("AUGMENTATION", f"Created context string from retrieved documents.\n\n---\n{context_string}\n---")

# Step C: GENERATE the final prompt.
final_prompt = rag_template.format(context=context_string, question=question)
logger.log("PROMPT GENERATION", f"Final prompt to be sent to LLM:\n\n---\n{final_prompt}\n---")

# Step D: INVOKE the model to get the answer.
logger.log("LLM INVOCATION", "Sending prompt to the language model...")
answer = llm.invoke(final_prompt)

# --- 6. Display the Final Result ---
logger.header("FINAL ANSWER")
print(answer.strip())