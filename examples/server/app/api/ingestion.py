# api/ingestion.py
import sys
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from fastapi import APIRouter, UploadFile, BackgroundTasks
from pathlib import Path
import shutil
from unstructured.partition.auto import partition
from chain.text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models

from chain.embeddings.local import LocalEmbeddings

# --- RAG PIPELINE SETUP (Updated for Arabic and a new model) ---

# --- FIX 1: Update the model name to match your powerful MLX model ---
# This identifier MUST match what your LM Studio server expects.
embedding_model = LocalEmbeddings(model_name="mlx-community/Qwen3-Embedding-8B-4bit-DWG")

qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "legal_documents_arabic" # It's good practice to use a new collection for a different model

try:
    # We must get the embedding size for the NEW model.
    print("Determining embedding size for the new model...")
    embedding_size = len(embedding_model.embed_query("test"))
    print(f"Detected embedding size: {embedding_size}")
except Exception as e:
    print(f"Could not connect to embedding model to get embedding size. Error: {e}")
    # Qwen models often have a different size, e.g., 4096. A fallback is less reliable here.
    embedding_size = 4096 # A common size for larger models, but connecting is better.

try:
    qdrant_client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except Exception:
    print(f"Collection '{COLLECTION_NAME}' not found. Creating a new one.")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=models.Distance.COSINE
        ),
    )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

SOURCE_DOCS_PATH = Path("./persistent_storage/source_documents")
router = APIRouter()

def process_document_in_background(filepath: Path):
    """
    The RAG pipeline function updated for Arabic.
    """
    print(f"BACKGROUND TASK: Parsing '{filepath.name}' with Arabic language support...")
    
    # --- FIX 2: Specify the language for unstructured ---
    # The `languages` parameter takes a list of language codes.
    try:
        elements = partition(filename=str(filepath), languages=['ara'])
    except Exception as e:
        print(f"Error during document parsing: {e}")
        # If parsing fails, we should stop processing this file.
        return
    
    print(f"BACKGROUND TASK: Chunking '{filepath.name}'...")
    full_text = "\n\n".join([e.text for e in elements])
    chunks = text_splitter.split_text(full_text)
    
    if not chunks:
        print(f"BACKGROUND TASK: No content found in '{filepath.name}'. Skipping.")
        return

    print(f"BACKGROUND TASK: Embedding and Indexing '{filepath.name}' with Qwen2 into Qdrant...")
    embeddings = embedding_model.embed_documents(chunks)
    
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"source": filepath.name, "text": chunk}
        )
        for embedding, chunk in zip(embeddings, chunks)
    ]
    
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )
    print(f"BACKGROUND TASK: Finished indexing '{filepath.name}'. Added {len(chunks)} points to Qdrant.")

@router.post("/ingest")
async def ingest_documents(files: list[UploadFile], background_tasks: BackgroundTasks):
    # This endpoint remains unchanged.
    saved_files = []
    for file in files:
        filepath = SOURCE_DOCS_PATH / file.filename # type: ignore
        with filepath.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        background_tasks.add_task(process_document_in_background, filepath)
        saved_files.append(file.filename)
        
    return {
        "message": f"Successfully started processing {len(saved_files)} file(s) in the background.",
        "filenames": saved_files
    }