"""
ChainForge End-to-End RAG Pipeline Demonstration (Production-Ready).

This script showcases the assembly and asynchronous execution of a full
Retrieval-Augmented Generation (RAG) chain. It is configured to use
production-grade components suitable for a cloud-native deployment: Qdrant
for persistent, scalable vector storage and Azure OpenAI for managed LLM and
embedding services.
"""
import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List

from chainforge.config import settings
from chainforge.core import Runnable, RunnableLambda
from chainforge.embeddings import OpenAIEmbeddings
from chainforge.llms import ChatOpenAI
from chainforge.prompts import StringPromptTemplate
from chainforge.vectorstores import QdrantVectorStore, BaseVectorStore

# Configure logging to see the framework's operational output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Quieten down verbose loggers from dependencies for a cleaner demo output.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def main():
    """
    Main execution function to set up and run the RAG chain.
    """
    print("--- ChainForge Production RAG Demo ---")
    
    # --- 1. Component Instantiation ---
    # This setup demonstrates a realistic production configuration.
    # We use Qdrant for vector storage and Azure for AI models.
    print("\n[Step 1] Instantiating production-grade components...")
    
    try:
        # The core AI models are configured to use Azure by default.
        # The components will fail fast if the required Azure env vars are missing.
        llm = ChatOpenAI(settings=settings)
        embedding_model = OpenAIEmbeddings(settings=settings)
        print("✅ Successfully instantiated Azure LLM and Embedding components.")

        # Instantiate the client for our Qdrant vector database.
        # For this demo, we'll use an in-memory instance for simplicity,
        # but in production, you would provide the URL to your hosted Qdrant.
        vector_store = QdrantVectorStore(
            embeddings=embedding_model,
            collection_name="chainforge_prod_demo",
            in_memory=True # In production: url="<your-qdrant-url>", port=6333
        )
        print("✅ Successfully instantiated QdrantVectorStore (in-memory).")

    except ValueError as e:
        print(f"\n❌ Configuration Error: Could not instantiate components.")
        print(f"   Please ensure all required environment variables are set in the .env file.")
        print(f"   Error details: {e}")
        return

    rag_prompt = StringPromptTemplate(
        "You are a helpful AI assistant. Answer the user's question based *only* on the "
        "following context. If the context does not contain the answer, state that clearly.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}"
    )
    print("All components instantiated.")

    # --- 2. Vector Store Population ---
    print("\n[Step 2] Populating Qdrant vector store with documents...")
    documents = [
        "The GigaWave 5000 is our flagship 5G home internet router with a max speed of 1 Gbps.",
        "To troubleshoot the GigaWave 5000, first try unplugging it for 30 seconds, then plugging it back in.",
        "The Unlimited Max mobile plan includes 100GB of premium high-speed data for your hotspot.",
        "You can check your data usage by dialing *3282# from your mobile device.",
        "Our return policy allows for returns of undamaged equipment within 30 days of purchase."
    ]
    await vector_store.aadd_documents(documents)
    print("Vector store populated.")

    # --- 3. RAG Chain Construction ---
    # This is the core of the declarative pipeline. Because the output of one
    # component does not perfectly match the input of the next, we use
    # `RunnableLambda` to perform the necessary data transformations.
    print("\n[Step 3] Constructing the RAG chain with data transformers...")
    
    # The chain now explicitly shows the data flow:
    # 1. `retriever`: Takes a query string, returns a list of doc strings.
    # 2. `RunnableLambda`: Takes the list of docs, formats it into a single context string.
    # 3. `prompt_builder`: Takes the context, combines it with the original question.
    # 4. `rag_prompt`: Takes the final dict, formats it into a single prompt string.
    # 5. `llm`: Takes the prompt string, returns the final answer string.
    
    retriever = vector_store # Our vector store is a Runnable[str, List[str]]
    
    # This lambda takes the output of the retriever (a list of docs) and prepares
    # it for the next step. This is a common pattern in data pipelines.
    def format_context(docs: List[str]) -> str:
        return "\n- ".join(docs)

    # We need to pass the original question along with the newly created context.
    # This lambda takes the original input (question) and the processed input (context)
    # and combines them into the dictionary required by our prompt template.
    def build_prompt_dict(inputs: tuple[str, str]) -> Dict[str, Any]:
        question, context = inputs
        return {"context": context, "question": question}

    # The `Runnable.abatch` is used implicitly here by `RunnableParallel` which is
    # not yet implemented, so for now we will create a more explicit chain.
    # This highlights the need for a parallel composition tool.
    
    # A more explicit chain:
    async def full_retrieval_step(question: str) -> Dict[str, Any]:
        """A single async function that encapsulates the full retrieval logic."""
        docs = await retriever.ainvoke(question)
        context = format_context(docs)
        return {"context": context, "question": question}

    # Now we can build the chain cleanly.
    rag_chain = RunnableLambda(full_retrieval_step) | rag_prompt | llm
    print("RAG chain constructed.")

    # --- 4. Chain Execution ---
    user_question = "How do I check how much hotspot data I have used?"
    print(f"\n[Step 4] Executing chain with question: '{user_question}'")
    
    # The input to the chain is now a simple string, as expected by our `full_retrieval_step`.
    final_answer = await rag_chain.ainvoke(user_question)

    print("\n--- FINAL ANSWER ---")
    print(final_answer)
    print("--------------------\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"The demo failed with an error: {e}", exc_info=True)