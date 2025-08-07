"""
Example 4: The simplest complete RAG pipeline with educational logging.

This script demonstrates a full, local RAG process with detailed step-by-step logging:
Load -> Split -> Embed -> Store -> Retrieve -> Generate.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.text_splitters import TokenTextSplitter
from minichain.embeddings import LocalEmbeddings
from minichain.vectors import FAISSVectorStore
from minichain.chat_models import LocalChatModel, LocalChatConfig
from minichain.prompts import PromptTemplate


class PipelineLogger:
    """A simple, reusable logging utility for educational RAG pipeline visualization."""
    
    # ANSI color codes for terminal formatting
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    
    def log_step(self, step_title: str, details: str):
        """Log a pipeline step with structured, colorful formatting."""
        separator = "=" * 80
        print(f"\n{self.CYAN}{self.BOLD}{separator}{self.RESET}")
        print(f"{self.CYAN}{self.BOLD}{step_title}{self.RESET}")
        print(f"{self.CYAN}{self.BOLD}{separator}{self.RESET}")
        print(f"{self.GREEN}{details}{self.RESET}")
        print()


# Initialize the logger
logger = PipelineLogger()

# 1. Professional Source Data
MINI_CHAIN_PHILOSOPHY = """
Mini-Chain is a micro-framework for building applications with Large Language Models.
It was born from a simple yet powerful idea: what if developers could use the core
components of a modern AI stack without the heavy abstractions? Our core principle
is transparency. We provide clean, single-purpose classes for each stage of the
RAG pipeline: Loaders, Splitters, Embedders, Memory, and Models. This modular,
"glass-box" design allows developers to understand, debug, and swap out any component
with ease.
"""

print(f"{logger.BLUE}{logger.BOLD}🚀 Starting Educational RAG Pipeline Demonstration{logger.RESET}")
print(f"{logger.YELLOW}This pipeline will show you every step of the RAG process in detail.{logger.RESET}")

# 2. Initialize components and create documents
embeddings = LocalEmbeddings()
text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=10) # type: ignore

# Split the text into documents
documents = text_splitter.create_documents([MINI_CHAIN_PHILOSOPHY])

# Log the text splitting step
logger.log_step(
    "1. TEXT SPLITTING",
    f"📄 Source text has been split into {len(documents)} document chunks.\n"
    f"📏 Chunk size: 100 tokens, Overlap: 10 tokens\n\n"
    f"📋 Example - First chunk content:\n"
    f"---\n{documents[0].page_content}\n---"
)

# Generate embeddings for the first document (for educational purposes)
sample_embedding = embeddings.embed_query(documents[0].page_content)

# Create vector store with documents
vector_store = FAISSVectorStore.from_documents(documents, embeddings)

# Log the indexing step with embeddings information
logger.log_step(
    "2. INDEXING",
    f"🔍 All {len(documents)} document chunks have been successfully embedded using local embeddings.\n"
    f"📐 Embedding dimensions: {len(sample_embedding)} (vector length)\n"
    f"💾 Documents are now indexed and stored in the FAISS vector database.\n\n"
    f"🔢 Example - Embedding vector for first chunk (first 10 dimensions):\n"
    f"[{', '.join([f'{x:.4f}' for x in sample_embedding[:10]])}...]\n\n"
    f"📊 Full embedding shape: {len(sample_embedding)} dimensions\n"
    f"✅ Knowledge base is ready for similarity-based retrieval."
)

# Initialize chat model
locale_config = LocalChatConfig()
chat_model = LocalChatModel(config=locale_config)

# 3. Ask a Question
question = "What is the core principle of Mini-Chain?"

# 4. RAG Process - Retrieval
retrieved_docs, scores = zip(*vector_store.similarity_search(query=question, k=1))
most_relevant_doc = retrieved_docs[0]
context = most_relevant_doc.page_content

# Generate embedding for the question to show the similarity process
question_embedding = embeddings.embed_query(question)

# Log the retrieval step with embedding information
logger.log_step(
    "3. DOCUMENT RETRIEVAL",
    f"❓ User Question: \"{question}\"\n\n"
    f"🔢 Question Embedding (first 10 dimensions):\n"
    f"[{', '.join([f'{x:.4f}' for x in question_embedding[:10]])}...]\n\n"
    f"🎯 Retrieved the most relevant document from the vector store:\n"
    f"📊 Similarity score: {scores[0]:.4f}\n"
    f"🔍 This score represents the cosine similarity between question and document embeddings.\n\n"
    f"📝 Retrieved Document Content:\n"
    f"---\n{context}\n---"
)

# 5. Engineer a professional prompt following best practices
prompt_template = PromptTemplate(
    template="""
You are a helpful assistant for the Mini-Chain library.
Answer the user's question based ONLY on the provided context.

Context:
{{ context }}

Question:
{{ question }}

Answer:
"""
)
final_prompt = prompt_template.format(context=context, question=question)

# Log the prompt engineering step
logger.log_step(
    "4. PROMPT ENGINEERING",
    f"🛠️  The retrieved context and user question have been formatted into a structured prompt.\n"
    f"📋 This prompt provides clear instructions and context boundaries to the LLM.\n\n"
    f"📄 Complete Prompt Being Sent to LLM:\n"
    f"---\n{final_prompt.strip()}\n---"
)

# 6. Generate the answer
answer = chat_model.invoke(final_prompt)

# Log the generation step
logger.log_step(
    "5. GENERATION",
    f"🤖 The Large Language Model has processed the prompt and generated a response.\n"
    f"✨ The model used only the provided context to answer the question.\n\n"
    f"💬 Generated Answer:\n"
    f"---\n{answer.strip()}\n---"
)

# Final summary
print(f"\n{logger.MAGENTA}{logger.BOLD}{'=' * 80}{logger.RESET}")
print(f"{logger.MAGENTA}{logger.BOLD}🎉 RAG PIPELINE COMPLETE - SUMMARY{logger.RESET}")
print(f"{logger.MAGENTA}{logger.BOLD}{'=' * 80}{logger.RESET}")
print(f"{logger.BLUE}❓ Question: {logger.RESET}{question}")
print(f"{logger.GREEN}💡 Final Answer: {logger.RESET}{answer.strip()}")
print(f"\n{logger.YELLOW}🔄 The RAG pipeline successfully retrieved relevant context and generated an informed response!{logger.RESET}")
