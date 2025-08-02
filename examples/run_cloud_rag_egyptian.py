# -*- coding: utf-8 -*-
"""
The Advanced Cloud RAG Pipeline: Egyptian Edition.

This script executes a robust, multi-step RAG pipeline using a full suite
of managed Azure services. It's designed to test Arabic language handling
on a production-grade cloud stack and serve as a direct comparison to the
local implementation.
"""

import time
import numpy as np
import logging
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.text_splitters.token_splitter import TokenTextSplitter
# Import the necessary components from our Mini-Chain framework
from minichain.core.types import Document
from minichain.text_splitters.implementations import RecursiveCharacterTextSplitter
from minichain.embeddings.azure import AzureOpenAIEmbeddings
from minichain.chat_models.azure import AzureOpenAIChatModel
from minichain.vector_stores.azure_ai_search import AzureAISearchVectorStore
from minichain.prompts.implementations import PromptTemplate

# --- Setup for robust logging ---

def setup_logging():
    """Sets up a comprehensive logging system for the pipeline."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('rag_pipeline_cloud_arabic.log', encoding='utf-8', mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

# --- Helper Functions ---

def print_header(title: str):
    """Prints a formatted header to the console."""
    print("\n" + "#" * 70)
    print(f"## {title.upper()} ".ljust(68) + "##")
    print("#" * 70)

def clean_arabic_text(text: str) -> str:
    """A simple function to normalize Arabic text from LLM outputs."""
    return ' '.join(text.split()).strip()

def main():
    """
    Executes the advanced, cloud-native, and robust Egyptian RAG pipeline.
    """
    logger = setup_logging()
    
    try:
        # --- PRE-FLIGHT CHECK ---
        print_header("خطوة استكشافية: فحص الاتصال بسحابة Azure")
        load_dotenv()
        
        required_vars = [
            "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_OPENAI_ENDPOINT_EMBEDDINGS", "AZURE_OPENAI_EMBEDDINGS_API_KEY", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME",
            "AZURE_AI_SEARCH_ENDPOINT", "AZURE_AI_SEARCH_ADMIN_KEY"
        ]
        if not all(os.getenv(var) for var in required_vars):
            logger.error("❌ ERROR: Missing one or more required Azure environment variables.")
            print("❌ ERROR: Missing required Azure credentials in .env file.")
            return

        logger.info("✅ All Azure credentials found. Proceeding with the cloud pipeline.")

        # --- STEP 1: INITIALIZE CLOUD COMPONENTS ---
        print_header("الخطوة 1: تجميع العقل السحابي (Azure)")
        azure_chat_model = AzureOpenAIChatModel(deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"), temperature=0.7) # type: ignore
        azure_embeddings = AzureOpenAIEmbeddings(deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")) # type: ignore
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=30)
        
        index_name = f"minichain-egyptian-demo-{int(time.time())}"
        vector_store = AzureAISearchVectorStore(embeddings=azure_embeddings, index_name=index_name)
        logger.info("✅ All Azure-native components initialized successfully.")

        # --- STEP 2: THE TEAM'S ORIGIN STORY ---
        print_header("الخطوة 2: استيعاب البرديات المقدسة (قصة الفريق)")
        team_story = """
        فلسفة 'Mini-Chain' متجذرة في خبرات فريقه الفريدة. المشروع بيطوره فادي،
        متدرب مجتهد وشقيان، لكن روحه بتتشكل من الأساطير اللي بيراجعوا شغله.

        أمين، واحد من كبار المراجعين، عنده خبرة من زمن تاني قبل عصر الـ LLMs.
        ده يعتبر من الرواد الحقيقيين، كان بيبني موديلز Computer Vision سنة 2017.
        في الأيام دي، النجاح كان معناه إنك "بتنحت في الصخر على Stack Overflow" لساعات،
        وبتتخانق مع وحوش قديمة زي Caffe و Theano. موافقته معناها إن الكود ده مش بس ذكي، لأ ده قوي ومتين.

        المراجع التاني، يحيى، هو تجسيد لروح المصادر المفتوحة والسيطرة الحقيقية على السيستم.
        كمستخدم قديم لـ Arch Linux، هو فاهم قوة النظام البسيط والمتظبط صح، اللي كل حتة فيه
        مختارة لهدف. تأثيره هو السبب إن Mini-Chain بيتجنب الحشو الزائد وبيحترم فلسفة لينكس:
        "اعمل حاجة واحدة، بس اعملها صح الصح".
        """
        print(f"النص الأصلي (قصة الفريق):\n---\n{team_story.strip()}\n---")
        documents = text_splitter.create_documents([team_story], [{"source": "asateer_el_fareeq.md"}])
        logger.info(f"✅ The story has been parsed into {len(documents)} chunks.")

        # --- STEP 3: INDEXING THE STORY ON AZURE ---
        print_header("الخطوة 3: الفهرسة السحابية (Azure AI)")
        vector_store.add_documents(documents)
        logger.info("Waiting for Azure AI Search to index documents...")
        time.sleep(5)
        logger.info("✅ Story indexed in Azure AI Search.")

        # --- ADVANCED STEP 4: QUERY TRANSFORMATION ---
        print_header("الخطوة 4: إطلاق العنان لعقل Azure (تحويل السؤال)")
        original_question = "احكيلي عن شقى أمين وتعب السنين في الشغلانة دي."
        print(f"سؤال المستخدم الأصلي: '{original_question}'")
        
        query_creation_prompt = PromptTemplate(
            template="أنت خبير بحث فهلوي. مهمتك تحوّل سؤال المستخدم لسؤال بحث مختصر وغني بالكلمات المفتاحية. ركز على المفاهيم والأسماء الأساسية.\n\nسؤال المستخدم: {question}\n\nصيغة البحث المحسّنة:"
        )
        optimized_query = azure_chat_model.invoke(query_creation_prompt.format(question=original_question))
        optimized_query = clean_arabic_text(optimized_query)
        print(f"سؤال البحث اللي طلعه Azure: '{optimized_query.strip()}'")
        logger.info(f"Azure AI refined the search query to: '{optimized_query.strip()}'")

        # --- STEP 5: RETRIEVAL FROM AZURE AI SEARCH ---
        print_header("الخطوة 5: البحث عن الحقيقة في سحابة Azure")
        retrieved_docs = vector_store.similarity_search(optimized_query, k=1)
        logger.info(f"🔍 Found {len(retrieved_docs)} relevant chunks from Azure AI Search.")
        print(f"🔍 Azure AI Search لقى أكتر حتة ذاكرة ليها علاقة بالموضوع:")
        for doc in retrieved_docs:
            print(f"  -> '{doc.page_content}'")

        # --- ADVANCED STEP 6: PERSONALITY-DRIVEN GENERATION ---
        print_header("الخطوة 6: صياغة الحكاية (إجابة بشخصية مصرية من Azure)")
        context_string = "\n---\n".join([doc.page_content for doc in retrieved_docs])
        storyteller_prompt = PromptTemplate(
            template="""أنت حكواتي مصري أسطوري ودمك خفيف. مهمتك تجاوب على السؤال بأسلوب شعبي ودرامي.
### المعلومات اللي عندك:
{context}
### السؤال:
{original_question}
### احكي الحكاية يا فنان:"""
        )
        final_prompt = storyteller_prompt.format(context=context_string, original_question=original_question)
        print("✨ الصيغة النهائية للـ prompt، جاهزة عشان تروح لـ Azure:\n---")
        print(final_prompt.strip())
        print("---")
        
        final_answer = azure_chat_model.invoke(final_prompt)
        final_answer = clean_arabic_text(final_answer)
        logger.info(f"Final answer generated by Azure, length: {len(final_answer)}")
        
        # --- FINAL OUTPUT ---
        print("\n" + "*" * 70)
        print(" حكاية من قلب سحابة Azure ".center(70, " "))
        print("*" * 70)
        print(f"السؤال: {original_question}")
        print("\nرد الحكواتي السحابي:")
        print(final_answer)
        print("*" * 70)

        # --- STEP 7: CLEANUP ---
        print_header("الخطوة 7: تنظيف الموارد السحابية")
        try:
            vector_store.index_client.delete_index(index_name)
            logger.info(f"✅ Successfully deleted temporary index '{index_name}'.")
        except Exception as e:
            logger.error(f"⚠️ Failed to delete index '{index_name}'. Please delete manually.", exc_info=True)

    except Exception as e:
        logger.error("❌ An unexpected error occurred in the main cloud pipeline.", exc_info=True)
        print(f"\n❌ حدث خطأ غير متوقع: {e}")

if __name__ == "__main__":
    main()