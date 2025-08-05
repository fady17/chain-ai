# examples/06_production_rag_azure.py
"""
Example 6: A production-grade, fully cloud-native RAG pipeline.

This script uses the robust, scalable Azure stack for all components:
- Azure OpenAI for embeddings and chat.
- Azure AI Search for the vector store.
"""

import time
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from minichain.text_splitters import TokenTextSplitter
from minichain.embeddings import AzureOpenAIEmbeddings
from minichain.memory import AzureAISearchVectorStore
from minichain.chat_models import AzureOpenAIChatModel, AzureChatConfig
from minichain.prompts import PromptTemplate

# 1. Professional Source Data
MINI_CHAIN_PHILOSOPHY = """
Mini-Chain is a micro-framework for building applications with Large Language Models.
Our core principle is transparency. This modular, "glass-box" design allows developers
to understand, debug, and swap out any component with ease. For example, a user can
seamlessly switch from a local FAISS vector store for prototyping to a scalable
Azure AI Search instance for production with a single line of code change.
This project is built for engineers who value control and clarity.
"""

# 2. Load credentials and initialize Azure components
load_dotenv()
print("✅ Azure credentials loaded.")
embeddings = AzureOpenAIEmbeddings(deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", ""))
text_splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=15)
azure_config = AzureChatConfig(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""),
    temperature=0.7, # config field
    max_tokens=100   # another config field
)

# 3. Pass the single config object
chat_model = AzureOpenAIChatModel(config=azure_config)

# chat_model = AzureOpenAIChatModel(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""))
index_name = f"prod-demo-{int(time.time())}"
vector_store = AzureAISearchVectorStore(embeddings, index_name)
print(f"✅ Azure components initialized. Using temporary index: {index_name}")

try:
    # 3. Ingest Data
    docs = text_splitter.create_documents(texts=[MINI_CHAIN_PHILOSOPHY])
    vector_store.add_documents(docs)
    print("Waiting for cloud indexing..."); time.sleep(5)
    print("✅ Knowledge base indexed in Azure AI Search.")
    
    # 4. RAG Process
    question = "How does Mini-Chain support moving from prototyping to production?"
    retrieved_docs, _ = zip(*vector_store.similarity_search(query=question, k=1))
    context = retrieved_docs[0].page_content
    
    prompt_template = PromptTemplate(
        template="""
You are a professional technical assistant for the Mini-Chain library.
Your goal is to provide clear and concise answers based ONLY on the provided context.

Context:
{{ context }}

Question:
{{ question }}

Answer:
"""
    )
    final_prompt = prompt_template.format(context=context, question=question)
    answer = chat_model.invoke(final_prompt)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer.strip()}")

finally:
    # 5. Clean up the cloud resource
    print(f"\nCleaning up index '{index_name}'...")
    vector_store.index_client.delete_index(index_name)
    print("✅ Cleanup complete.")