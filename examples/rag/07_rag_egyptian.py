
# examples/08_advanced_rag_azure_arabic.py
"""
Example 8: The Ultimate Showcase - An Advanced, Multi-Agent RAG Pipeline on Azure.

This script demonstrates the full power of the Mini-Chain framework by
executing a sophisticated, multi-step workflow entirely on Azure services.

The pipeline operates as follows:
1.  **Query Analysis**: An AI agent receives a user's question in Arabic. It uses
    a Pydantic model to analyze the question and extract a precise, optimized
    search query in English for better retrieval performance.
2.  **Retrieval**: The optimized query is used to search for the most relevant
    information in a professional knowledge base stored in Azure AI Search.
3.  **Generation**: The retrieved context is passed to a final AI agent with a
    professional Egyptian persona ("المساعد التقني الفهلوي") who formulates a
    clear, concise answer in Arabic.

This represents a production-grade, cross-lingual, agentic pattern.
"""
# -*- coding: utf-8 -*-
import  time, json
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
# Import all necessary Mini-Chain components
from minichain.text_splitters import TokenTextSplitter
from minichain.embeddings import AzureOpenAIEmbeddings
from minichain.memory import AzureAISearchVectorStore
from minichain.chat_models import AzureOpenAIChatModel
from minichain.prompts import PromptTemplate
from minichain.output_parsers import PydanticOutputParser

def print_header(title: str):
    print("\n" + "#" * 70)
    print(f"## {title.upper()} ".ljust(68) + "##")
    print("#" * 70)

class AnalyzedQuery(BaseModel):
    """A structured representation of the user's intent."""
    optimized_search_query: str = Field(
        description="A concise, keyword-rich search query in English, derived from the user's original question."
    )
    user_intent: str = Field(
        description="A brief summary of what the user is asking about."
    )

MINI_CHAIN_PHILOSOPHY = """
Mini-Chain is a micro-framework for building with LLMs. Its core principle is
transparency. The modular, "glass-box" design allows developers to easily
understand, debug, and swap components like vector stores. For example, a user can
seamlessly switch from a local FAISS store for prototyping to a scalable
Azure AI Search instance for production with a single line of code change. This
is for engineers who value control and clarity.
"""

def main():
    print_header("Step 1: Initializing Azure-Native Pipeline")
    load_dotenv()
    # ... (credential check) ...

    # Initialize all Azure components
    embeddings = AzureOpenAIEmbeddings(deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", ""))
    chat_model = AzureOpenAIChatModel(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", ""), temperature=0.2)
    text_splitter = TokenTextSplitter(chunk_size=150, chunk_overlap=15)
    
    index_name = f"advanced-arabic-demo-{int(time.time())}"
    vector_store = AzureAISearchVectorStore(embeddings, index_name)
    print(f"✅ Azure components initialized. Using temporary index: {index_name}")

    try:
        # ... (Step 2: Ingest English Knowledge Base) ...
        docs = text_splitter.create_documents(texts=[MINI_CHAIN_PHILOSOPHY])
        vector_store.add_documents(docs)
        print("Waiting for cloud indexing..."); time.sleep(5)
        print("✅ English knowledge base indexed in Azure AI Search.")

        # --- Step 3: Advanced Query Analysis ---
        print_header("Step 3: AI Query Analysis Agent")
        user_question_arabic = "إيه هي أهم ميزة في تصميم 'ميني تشين'؟"
        print(f"User Question (Arabic): '{user_question_arabic}'")

        parser = PydanticOutputParser(pydantic_object=AnalyzedQuery)
        analysis_prompt = PromptTemplate(
            template="""
You are an expert multilingual query analyst. Your task is to understand the user's
Arabic question and convert it into a structured JSON object containing an optimized
English search query and a summary of the intent.

{{ format_instructions }}

User Question (Arabic):
{{ arabic_question }}
"""
        )
        
        prompt_text = analysis_prompt.format(
            format_instructions=parser.get_format_instructions(),
            arabic_question=user_question_arabic
        )
        llm_output = chat_model.invoke(prompt_text)
        analyzed_query = parser.parse(llm_output)
        
        print("\n--- AI Analysis Result ---")
        
        # --- FIX: Use the two-step method for full control ---
        # 1. Dump the Pydantic model to a Python dictionary.
        analyzed_query_dict = analyzed_query.model_dump()
        # 2. Use the standard json.dumps to print with ensure_ascii=False.
        print(json.dumps(analyzed_query_dict, indent=2, ensure_ascii=False))
        
        print("--------------------------")

        # ... (Rest of the script: Retrieval, Generation, Cleanup) ...
        print_header("Step 4: Precision Retrieval from Azure AI Search")
        print(f"Using optimized English query: '{analyzed_query.optimized_search_query}'")
        retrieved_docs, _ = zip(*vector_store.similarity_search(query=analyzed_query.optimized_search_query, k=1))
        context = retrieved_docs[0].page_content
        print(f"\nRetrieved Context (English):\n---\n{context}\n---")

        print_header("Step 5: Handing Off to the Egyptian Technical Assistant")
        final_prompt_template = PromptTemplate(
            template="""
أنت "المساعد التقني الفهلوي" لمكتبة 'ميني تشين'.
مهمتك هي الإجابة على سؤال المستخدم باللغة العربية بوضوح واحترافية،
معتمدًا فقط على المعلومات الإنجليزية المقدمة لك.

المعلومات (Context):
{{ context }}

سؤال المستخدم الأصلي:
{{ original_question }}

الإجابة النهائية باللغة العربية:
"""
        )
        final_prompt = final_prompt_template.format(
            context=context,
            original_question=user_question_arabic
        )
        final_answer = chat_model.invoke(final_prompt)
        
        print("\n" + "*" * 70)
        print(" الإجابة النهائية من خط الأنابيب المتقدم ".center(70, " "))
        print("*" * 70)
        print(f"السؤال الأصلي: {user_question_arabic}")
        print("\nالإجابة النهائية:")
        print(final_answer.strip())
        print("*" * 70)
        
    finally:
        print_header("Step 6: Cleaning Up Cloud Resources")
        try:
            vector_store.index_client.delete_index(index_name)
            print(f"✅ Successfully deleted temporary index '{index_name}'.")
        except Exception as e:
            print(f"⚠️ Failed to delete index '{index_name}'. Please clean up manually in Azure.")

if __name__ == "__main__":
    main()