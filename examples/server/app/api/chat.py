import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
import tiktoken 
import json
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.rag import query_knowledge_base
# --- Import your chat model and its config ---
from chain.chat_models.local import LocalChatModel
from chain.chat_models.base import LocalChatConfig
from chain.core.types import HumanMessage, SystemMessage

# --- CHAT MODEL SETUP ---
# Create an instance of your local chat model.
# The model identifier should match what you've loaded in LM Studio.
# This assumes you have a chat/instruction-tuned model loaded, not an embedding model.
chat_config = LocalChatConfig(model="qwen/qwen3-8b")
chat_model = LocalChatModel(config=chat_config)


router = APIRouter()

# --- TOKENIZER and CONTEXT LIMIT SETUP ---
# It's a good practice to define the model's limits.
# We'll be conservative and assume a context limit, leaving room for the answer.
CONTEXT_LIMIT = 4096 # Let's assume a safe context limit
try:
    # Use a generic tokenizer that works well for many models.
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    # Fallback for older versions
    tokenizer = tiktoken.get_encoding("p50k_base")

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

async def rag_stream_generator(question: str) -> AsyncGenerator[str, None]:
    print(f"Retrieving context for question: {question}")
    retrieved_chunks = query_knowledge_base(question)
    
    # --- FIX: Intelligent Context Truncation ---
    context = ""
    current_token_count = 0
    
    # Build the context string chunk by chunk, checking the token count
    for chunk in retrieved_chunks:
        chunk_text = chunk.get('text', '') # pyright: ignore[reportOptionalMemberAccess]
        chunk_token_count = len(tokenizer.encode(chunk_text))
        
        # Add the chunk if it fits within our defined limit
        if current_token_count + chunk_token_count < CONTEXT_LIMIT:
            context += f"\n\n---\n\n{chunk_text}"
            current_token_count += chunk_token_count
        else:
            # Stop adding chunks once we're near the limit
            print("Context limit reached. Truncating further context.")
            break
    
    if not context:
        context = "No relevant context found in the knowledge base."

    # Define the messages to be sent to the model
    messages = [
        SystemMessage(content=(
            "You are an expert legal AI assistant... (rest of your system prompt)"
        )),
        HumanMessage(content=f"Context:\n{context}\n\n---\n\nQuestion: {question}")
    ]
    
    print(f"Streaming response from local chat model: {chat_model.model_name}...")
    try:
        for chunk in chat_model.stream(messages):
            sse_chunk = {"type": "text", "delta": chunk}
            yield f"{json.dumps(sse_chunk)}\n"
    except Exception as e:
        print(f"Error during streaming from local model: {e}")
        error_chunk = {"type": "error", "message": str(e)}
        yield f"{json.dumps(error_chunk)}\n"

    print("Finished streaming.")

@router.post("/stream_rag_query")
async def stream_rag_query_endpoint(request: ChatRequest):
    return StreamingResponse(
        rag_stream_generator(request.question), 
        media_type="text/plain"
    )