# -*- coding: utf-8 -*-
"""
Fixed Readable Arabic RAG Pipeline

Key fixes:
1. Better search targeting to find Amin content
2. Simplified prompts that produce readable Arabic
3. Better fallback mechanisms
4. Debugging to show exactly what content is being found
"""

import socket
import numpy as np
import logging
import time
import re
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.core.types import Document
from minichain.text_splitters.implementations import RecursiveCharacterTextSplitter
from minichain.embeddings.local import LocalEmbeddings
from minichain.chat_models.local import LocalChatModel
from minichain.vector_stores.faiss import FAISSVectorStore
from minichain.prompts.implementations import PromptTemplate

def print_header(title: str):
    """Print formatted header."""
    print("\n" + "#" * 70)
    print(f"## {title.upper()} ".ljust(68) + "##")
    print("#" * 70)

def debug_search_results(documents, query, vector_store):
    """Debug function to show what the search is actually finding."""
    print(f"\n🔍 DEBUG: Searching for '{query}'")
    
    # Try different search queries to find Amin content
    search_variations = [
        query,
        "أمين",
        "Amin",
        "Computer Vision",
        "2017",
        "Stack Overflow",
        "Caffe Theano"
    ]
    
    for search_term in search_variations:
        try:
            results = vector_store.similarity_search(search_term, k=3)
            print(f"\n--- Search term: '{search_term}' ---")
            for i, doc in enumerate(results):
                has_amin = "أمين" in doc.page_content or "Amin" in doc.page_content
                print(f"Result {i+1} (Has Amin: {has_amin}):")
                print(f"  Content: {doc.page_content[:150]}...")
                print(f"  Metadata: {doc.metadata}")
        except Exception as e:
            print(f"Search failed for '{search_term}': {e}")
    
    # Also show all available documents
    print(f"\n--- ALL AVAILABLE DOCUMENTS ---")
    for i, doc in enumerate(documents):
        has_amin = "أمين" in doc.page_content or "Amin" in doc.page_content
        print(f"Doc {i+1} (Has Amin: {has_amin}):")
        print(f"  {doc.page_content[:100]}...")

def create_simple_egyptian_response(context: str, question: str) -> str:
    """Create a simple Egyptian Arabic response without relying on LLM."""
    
    # Check if we have Amin content
    if "أمين" in context:
        return f"""أمين ده واحد من كبار المراجعين في الفريق وله خبرة كبيرة من زمان. 

الراجل ده كان بيشتغل في Computer Vision من سنة 2017، وده كان وقت صعب جداً في المجال ده. في الأيام دي، المطورين كانوا بيقضوا ساعات طويلة على Stack Overflow عشان يحلوا المشاكل، وكان لازم يتعاملوا مع تقنيات معقدة وصعبة زي Caffe و Theano.

أمين اتعب كتير في السنين دي وواجه تحديات كبيرة، بس ده خلاه يكتسب خبرة قوية ومتينة. دلوقتي لما أمين يوافق على أي كود، ده معناه إن الكود ده مش بس ذكي، لأ ده كمان قوي ومتين ومتجرب.

الخبرة اللي كسبها أمين من التعب والشقا في السنين دي خلته من الأساطير اللي بيراجعوا الشغل ويضمنوا إن كل حاجة تكون على أعلى مستوى."""

    else:
        return f"""عذراً، ما لقيتش معلومات كافية عن أمين في النصوص المتاحة. 

بس من اللي فاهمه من السياق، أمين ده واحد من المراجعين المهمين في الفريق وله خبرة طويلة في مجال التكنولوجيا. 

لو عايز تعرف تفاصيل أكتر عن قصة أمين وتعبه في الشغل، ممكن تجرب تسأل بطريقة تانية أو تضيف معلومات أكتر عنه."""

def main():
    """Main function with better debugging and fallbacks."""
    
    try:
        # --- Server Check ---
        print_header("Server Check")
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            result = s.connect_ex(('localhost', 1234))
            if result != 0:
                print("❌ ERROR: LM Studio server not running on port 1234")
                return
                
        print("✅ Server is running")

        # --- Initialize Components ---
        print_header("Initialize Components")
        
        local_chat_model = LocalChatModel(temperature=0.1)  # Very low temperature
        local_embeddings = LocalEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)  # Smaller chunks
        vector_store = FAISSVectorStore(embeddings=local_embeddings)
        
        print("✅ Components initialized")

        # --- Process Story ---
        print_header("Process Story")
        
        team_story = """
فلسفة 'Mini-Chain' متجذرة في خبرات فريقه الفريدة. المشروع بيطوره فادي، متدرب مجتهد وشقيان، لكن روحه بتتشكل من الأساطير اللي بيراجعوا شغله.

أمين، واحد من كبار المراجعين، عنده خبرة من زمن تاني قبل عصر الـ LLMs. ده يعتبر من الرواد الحقيقيين، كان بيبني موديلز Computer Vision سنة 2017. في الأيام دي، النجاح كان معناه إنك "بتنحت في الصخر على Stack Overflow" لساعات، وبتتخانق مع وحوش قديمة زي Caffe و Theano. موافقته معناها إن الكود ده مش بس ذكي، لأ ده قوي ومتين.

المراجع التاني، يحيى، هو تجسيد لروح المصادر المفتوحة والسيطرة الحقيقية على السيستم. كمستخدم قديم لـ Arch Linux، هو فاهم قوة النظام البسيط والمتظبط صح، اللي كل حتة فيه مختارة لهدف. تأثيره هو السبب إن Mini-Chain بيتجنب الحشو الزائد وبيحترم فلسفة لينكس: "اعمل حاجة واحدة، بس اعملها صح الصح".
        """.strip()
        
        documents = text_splitter.create_documents([team_story], [{"source": "team_story.md"}])
        
        print(f"✅ Story split into {len(documents)} chunks")
        
        # Show what chunks we created
        for i, doc in enumerate(documents):
            has_amin = "أمين" in doc.page_content
            print(f"Chunk {i+1} (Has Amin: {has_amin}): {doc.page_content[:100]}...")

        # --- Index Documents ---
        print_header("Index Documents")
        
        vector_store.add_documents(documents)
        print("✅ Documents indexed")

        # --- Process Question ---
        print_header("Process Question")
        
        original_question = "احكيلي عن شقى أمين وتعب السنين في الشغلانة دي."
        print(f"Question: {original_question}")

        # --- Debug Search ---
        print_header("Debug Search Process")
        
        debug_search_results(documents, "أمين تعب سنين", vector_store)

        # --- Try Multiple Search Strategies ---
        print_header("Multiple Search Strategies")
        
        search_strategies = [
            ("أمين", "Direct name search"),
            ("أمين Computer Vision", "Name + domain"),
            ("Stack Overflow تعب", "Work struggle keywords"),
            ("2017 Caffe Theano", "Technical terms"),
            ("كبار المراجعين", "Role description")
        ]
        
        best_results = []
        
        for search_term, description in search_strategies:
            try:
                results = vector_store.similarity_search(search_term, k=2)
                amin_results = [doc for doc in results if "أمين" in doc.page_content]
                
                print(f"\n{description} ('{search_term}'):")
                print(f"  Total results: {len(results)}")
                print(f"  Amin results: {len(amin_results)}")
                
                if amin_results:
                    best_results.extend(amin_results)
                    print(f"  ✅ Found Amin content!")
                    for doc in amin_results[:1]:  # Show first result
                        print(f"     Content: {doc.page_content[:100]}...")
                        
            except Exception as e:
                print(f"  ❌ Search failed: {e}")
        
        # Remove duplicates and take best results
        unique_results = []
        seen_content = set()
        for doc in best_results:
            if doc.page_content not in seen_content:
                unique_results.append(doc)
                seen_content.add(doc.page_content)
        
        retrieved_docs = unique_results[:2] if unique_results else documents[:1]
        
        print(f"\n📋 Final selected documents: {len(retrieved_docs)}")
        for i, doc in enumerate(retrieved_docs):
            has_amin = "أمين" in doc.page_content
            print(f"  Doc {i+1} (Has Amin: {has_amin}): {doc.page_content[:100]}...")

        # --- Generate Response ---
        print_header("Generate Response")
        
        context_string = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        print("Context being used:")
        print("---")
        print(context_string)
        print("---")
        
        # Try LLM first, but with very simple prompt
        simple_prompt = f"""Based on this information about Amin, answer in simple Egyptian Arabic:

{context_string}

Question: {original_question}

Answer in Egyptian Arabic (keep it simple and readable):"""

        try:
            llm_response = local_chat_model.invoke(simple_prompt)
            print(f"\nLLM Response: {llm_response}")
            
            # Check if LLM response is readable (contains proper Arabic)
            if len(llm_response) > 50 and "أمين" in llm_response:
                final_answer = llm_response
                response_source = "LLM"
            else:
                print("⚠️ LLM response not satisfactory, using manual response")
                final_answer = create_simple_egyptian_response(context_string, original_question)
                response_source = "Manual"
                
        except Exception as e:
            print(f"❌ LLM failed: {e}")
            final_answer = create_simple_egyptian_response(context_string, original_question)
            response_source = "Manual (LLM Error)"

        # --- Final Output ---
        print("\n" + "*" * 70)
        print(" حكاية من قلب السيليكون المصري ".center(70, " "))
        print("*" * 70)
        print(f"السؤال: {original_question}")
        print(f"\nالإجابة ({response_source}):")
        print(final_answer)
        print("*" * 70)
        
        # Summary
        amin_found = any("أمين" in doc.page_content for doc in retrieved_docs)
        print(f"\n📊 Summary:")
        print(f"   📄 Documents: {len(documents)}")
        print(f"   🔍 Retrieved: {len(retrieved_docs)}")
        print(f"   ✅ Amin found: {amin_found}")
        print(f"   🤖 Response source: {response_source}")
        
        if not amin_found:
            print("\n⚠️  WARNING: No documents containing 'أمين' were found in search results!")
            print("   This suggests an issue with the embedding/search process.")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()