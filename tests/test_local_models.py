# tests/test_local_models.py
"""
Test suite for local models (Chat & Embeddings) served via LM Studio.
"""
import os
import sys
import socket
import numpy as np # Import numpy for vector preview
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.chat_models.local import LocalChatModel
from minichain.embeddings.local import LocalEmbeddings
from minichain.core.types import SystemMessage, HumanMessage

# --- Helper Functions ---
def is_port_in_use(port: int) -> bool:
    """Check if a port is in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def print_header(title):
    print("\n" + "="*60)
    print(f" {title.upper()} ".center(60, " "))
    print("="*60)

# --- NEW: Helper function to display vector previews ---
def print_vector_preview(vector: list, prefix=""):
    """Prints a preview of a vector (first 3 and last 3 elements)."""
    if not vector:
        print(f"{prefix}Vector Preview: [Empty Vector]")
        return
    preview = f"[{', '.join(map(str, np.round(vector[:3], 4)))}, ..., {', '.join(map(str, np.round(vector[-3:], 4)))}]"
    print(f"{prefix}Vector Preview: {preview}")

# --- Test Data ---
SAMPLE_DOCS = [
    "This is the first document for embedding.",
    "Here is the second one, slightly different."
]
SAMPLE_QUERY = "A query to be embedded."

# --- Test Functions ---
def test_local_chat_model():
    """Tests the LocalChatModel by connecting to an LM Studio server."""
    print_header("Testing Local Chat Model via LM Studio")
    server_port = 1234
    if not is_port_in_use(server_port):
        print(f"⚠️ Skipping test: LM Studio server not found on port {server_port}.")
        return True
    print(f"✅ LM Studio server detected on port {server_port}.")
    try:
        local_model = LocalChatModel(model_name="qwen2-7b-instruct-gguf")
        response = local_model.invoke("In one sentence, what is a Large Language Model?")
        print(f"   Response: '{response}'")
        assert isinstance(response, str) and len(response) > 10
        return True
    except Exception as e:
        print(f"\n❌ CHAT TEST FAILED: {e}")
        return False

# --- UPDATED TEST FUNCTION ---
def test_local_embeddings_model():
    """Tests the LocalEmbeddings model with detailed output."""
    print_header("Testing Local Embeddings Model via LM Studio")
    server_port = 1234
    if not is_port_in_use(server_port):
        print(f"⚠️ Skipping test: LM Studio server not found on port {server_port}.")
        return True
    print(f"✅ LM Studio server detected on port {server_port}.")
    
    try:
        print("\nSTEP 1: INITIALIZING `LocalEmbeddings`")
        local_embeddings = LocalEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
        print("✅ Embeddings model initialized successfully.")
        
        print("\nSTEP 2: TESTING `embed_query`")
        print(f"   Query Text: '{SAMPLE_QUERY}'")
        query_vector = local_embeddings.embed_query(SAMPLE_QUERY)
        
        print("   --- Output from LM Studio ---")
        print_vector_preview(query_vector, prefix="   ") # <-- VISUALIZE THE VECTOR
        
        expected_dimension = 768
        print(f"\n   Vector Dimensions: {len(query_vector)} (Expected: {expected_dimension})")
        assert len(query_vector) == expected_dimension
        print("✅ Received a valid query vector with correct dimensions.")
        
        print("\nSTEP 3: TESTING `embed_documents`")
        print(f"   Document Texts: {SAMPLE_DOCS}")
        doc_vectors = local_embeddings.embed_documents(SAMPLE_DOCS)
        
        print("   --- Output from LM Studio ---") # <-- VISUALIZE THE VECTORS
        for i, vec in enumerate(doc_vectors):
            print(f"\n   Vector for Document [{i}]:")
            print(f"     Text: '{SAMPLE_DOCS[i]}'")
            print_vector_preview(vec, prefix="     ")
            
        print(f"\n   Returned {len(doc_vectors)} vectors in total.")
        assert len(doc_vectors) == len(SAMPLE_DOCS)
        assert all(len(vec) == expected_dimension for vec in doc_vectors)
        print("✅ Received valid document vectors with correct dimensions.")
        
        print_header("Local Embeddings Test Passed")
        return True
        
    except Exception as e:
        print(f"\n❌ EMBEDDING TEST FAILED: {e}")
        return False

# --- Main Test Runner ---
if __name__ == "__main__":
    chat_result = test_local_chat_model()
    embeddings_result = test_local_embeddings_model()
    
    print_header("Local Model Test Summary")
    print(f"Chat Model Test:    {'PASSED' if chat_result else 'FAILED'}")
    print(f"Embeddings Test:  {'PASSED' if embeddings_result else 'FAILED'}")
    
    if chat_result and embeddings_result:
        print("\n🎉 All local model components are working!")
        print("✅ Ready to build a fully local RAG pipeline.")
# # tests/test_local_models.py
# """
# Test suite for local models served via LM Studio.
# """
# import os
# import sys
# import socket
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# from minichain.chat_models.local import LocalChatModel
# from minichain.core.types import SystemMessage, HumanMessage

# # --- Helper to check if the local server is running ---
# def is_port_in_use(port: int) -> bool:
#     """Check if a port is in use on localhost."""
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#         return s.connect_ex(('localhost', port)) == 0

# def print_header(title):
#     print("\n" + "="*60)
#     print(f" {title.upper()} ".center(60, " "))
#     print("="*60)

# # --- Test Function ---
# def test_local_chat_model():
#     """
#     Tests the LocalChatModel by connecting to an LM Studio server.
#     """
#     print_header("Testing Local Chat Model via LM Studio")
    
#     # --- Step 1: Check for LM Studio Server ---
#     print("STEP 1: CHECKING FOR LM STUDIO SERVER")
#     server_port = 1234
#     if not is_port_in_use(server_port):
#         print(f"⚠️ Skipping test: LM Studio server not found on port {server_port}.")
#         print("   Please start the server in LM Studio and load a model.")
#         return True
    
#     print(f"✅ LM Studio server detected on port {server_port}.")
    
#     try:
#         # --- Step 2: Initialization ---
#         print("\nSTEP 2: INITIALIZING `LocalChatModel`")
#         # The model_name can be a descriptive placeholder
#         local_model = LocalChatModel(model_name="qwen2-7b-instruct-gguf")
#         print("✅ Model initialized successfully.")
        
#         # --- Step 3: Simple Invoke with a String ---
#         print("\nSTEP 3: TESTING `invoke` WITH A STRING")
#         prompt = "In one sentence, what is a Large Language Model?"
#         print(f"   Prompt: '{prompt}'")
        
#         response = local_model.invoke(prompt)
        
#         print(f"   Response: '{response}'")
#         assert isinstance(response, str) and len(response) > 10
#         print("✅ Received a valid string response.")

#         # --- Step 4: Invoke with a Message List ---
#         print("\nSTEP 4: TESTING `invoke` WITH SYSTEM & HUMAN MESSAGES")
#         messages = [
#             SystemMessage("You are a helpful pirate assistant. You always answer in character."),
#             HumanMessage("What is the capital of France?")
#         ]
#         print(f"   System: '{messages[0].content}'")
#         print(f"   Human: '{messages[1].content}'")

#         pirate_response = local_model.invoke(messages)
        
#         print(f"   Pirate Response: '{pirate_response}'")
#         assert "arr" in pirate_response.lower() or "matey" in pirate_response.lower() or "paris" in pirate_response.lower()
#         print("✅ Received a valid, in-character response.")

#         print_header("Local Chat Model Test Passed")
#         return True

#     except Exception as e:
#         print(f"\n❌ TEST FAILED: An error occurred: {e}")
#         import traceback
#         traceback.print_exc()
#         print("\n   TROUBLESHOOTING:")
#         print("   - Is a model fully loaded in LM Studio?")
#         print("   - Is the server running on the correct port (1234)?")
#         print("   - Does the model support the chatml prompt format (most do)?")
#         return False

# if __name__ == "__main__":
#     test_local_chat_model()