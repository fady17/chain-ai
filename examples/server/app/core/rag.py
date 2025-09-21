# core/rag.py
from qdrant_client import QdrantClient
from chain.embeddings.local import LocalEmbeddings

# Reuse the same instances
# --- Use the same updated model name ---
embedding_model = LocalEmbeddings(model_name="mlx-community/Qwen3-Embedding-8B-4bit-DWG")

qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "legal_documents"

def query_knowledge_base(query_text: str, n_results: int = 5):
    """
    Queries the Qdrant collection for relevant document chunks.
    """
    query_embedding = embedding_model.embed_query(query_text)
    
    # In Qdrant, this is called a "search" operation
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=n_results,
        with_payload=True # Ensure we get our metadata back
    )
    
    # The payload contains our original metadata
    return [hit.payload for hit in search_results]