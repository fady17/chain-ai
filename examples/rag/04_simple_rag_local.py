# examples/04_simple_rag_local.py
"""
Example 4: The simplest complete RAG pipeline.

This script demonstrates a full, local RAG process:
Load -> Split -> Embed -> Store -> Retrieve -> Generate.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.text_splitters import TokenTextSplitter
from minichain.embeddings import LocalEmbeddings
from minichain.memory import FAISSVectorStore
from minichain.chat_models import LocalChatModel
from minichain.prompts import PromptTemplate

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

# 2. Initialize local components
embeddings = LocalEmbeddings()
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
vector_store = FAISSVectorStore.from_documents(
    text_splitter.create_documents([MINI_CHAIN_PHILOSOPHY]), embeddings
)
chat_model = LocalChatModel()
print("✅ Knowledge base indexed.")

# 3. Ask a Question
question = "What is the core principle of Mini-Chain?"

# 4. RAG Process
# Retrieve the most relevant context
retrieved_docs, _ = zip(*vector_store.similarity_search(query=question, k=1))
context = retrieved_docs[0].page_content

# Engineer a professional prompt following best practices
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

# Generate the answer
answer = chat_model.invoke(final_prompt)

print(f"\nQuestion: {question}")
print(f"Answer: {answer.strip()}")