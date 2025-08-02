# tests/test_embeddings.py
"""
Test suite for Azure embedding models with detailed logging.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from dotenv import load_dotenv
from minichain.embeddings.azure import AzureOpenAIEmbeddings

load_dotenv()

# --- Test Data ---
SAMPLE_DOCS = [
    "Built a full computer vision system in 2017 — honestly, we should take a picture with him.",
    "Uses Arch btw — only those who’ve faced 3AM kernel panics can understand."
]
SAMPLE_QUERY = "I use arch btw."

# --- Helper Functions ---
def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title.upper()} ".center(60, " "))
    print("=" * 60)

def print_vector_preview(vector: list, prefix=""):
    """Prints a preview of a vector (first 3 and last 3 elements)."""
    preview = f"[{', '.join(map(str, vector[:3]))}, ..., {', '.join(map(str, vector[-3:]))}]"
    print(f"{prefix}Vector Preview: {preview}")

# --- Main Test ---
def test_azure_openai_embeddings_verbose():
    """Test AzureOpenAIEmbeddings functionality with detailed output."""
    
    print_header("Azure OpenAI Embeddings Test")

    # --- Step 1: Configuration Check & Initialization ---
    print("STEP 1: CONFIGURATION & INITIALIZATION")
    
    required_vars = {
        "Endpoint": "AZURE_OPENAI_ENDPOINT_EMBEDDINGS", 
        "API Key": "AZURE_OPENAI_EMBEDDINGS_API_KEY", 
        "Deployment Name": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    }
    
    config_valid = True
    for name, var in required_vars.items():
        value = os.getenv(var)
        if not value:
            print(f"❌ {name}: NOT SET ({var})")
            config_valid = False
        else:
            # Mask the API key for security
            display_value = "****" + value[-4:] if name == "API Key" else value
            print(f"✅ {name}: {display_value}")

    if not config_valid:
        print("\n⚠️ Skipping test - required environment variables not set.")
        return True

    try:
        deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        embeddings = AzureOpenAIEmbeddings(
            deployment_name=deployment_name, # type: ignore
            model_name="text-embedding-3-small"
        )
        print("✅ Initialization successful.")
        
        # --- Step 2: Testing `embed_query` (Single Text) ---
        print("\nSTEP 2: TESTING `embed_query` (SINGLE TEXT)")
        print(f"Input Query: '{SAMPLE_QUERY}'")
        
        query_vector = embeddings.embed_query(SAMPLE_QUERY)
        
        print("--- Output Analysis ---")
        print(f"Vector Type: {type(query_vector)}")
        print(f"Vector Length (Dimensions): {len(query_vector)}")
        print_vector_preview(query_vector)
        
        # Verification
        expected_dimension = 1536
        assert isinstance(query_vector, list)
        assert len(query_vector) == expected_dimension
        print("✅ Verification PASSED: Vector has correct type and dimensions.")
        
        # --- Step 3: Testing `embed_documents` (Batch of Texts) ---
        print("\nSTEP 3: TESTING `embed_documents` (BATCH OF TEXTS)")
        print(f"Input Documents ({len(SAMPLE_DOCS)} total):")
        for i, doc in enumerate(SAMPLE_DOCS):
            print(f"  [{i}]: '{doc}'")
            
        doc_vectors = embeddings.embed_documents(SAMPLE_DOCS)
        
        print("\n--- Output Analysis ---")
        print(f"Result Type: {type(doc_vectors)}")
        print(f"Result Shape: ({len(doc_vectors)}, {len(doc_vectors[0]) if doc_vectors else 0})")
        
        for i, vec in enumerate(doc_vectors):
            print(f"\nDocument [{i}] Vector:")
            print(f"  - Dimensions: {len(vec)}")
            print_vector_preview(vec, prefix="  - ")

        # Verification
        assert isinstance(doc_vectors, list)
        assert len(doc_vectors) == len(SAMPLE_DOCS)
        assert all(len(vec) == expected_dimension for vec in doc_vectors)
        print("\n✅ Verification PASSED: All document vectors have correct shape and dimensions.")
        
        print_header("Test Result: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Test Runner ---
def run_embedding_tests():
    print("🧪 Testing Phase 5: Embeddings (Azure-Only)")
    result = test_azure_openai_embeddings_verbose()
    
    print("-" * 60)
    if result:
        print("\n🎉 Phase 5 Complete - Azure embedding model working!")
        print("✅ Ready to move to Phase 6: Vector Stores")
    else:
        print("\n⚠️ Azure embedding test failed. Please check your configuration and code.")
        
    return result

if __name__ == "__main__":
    run_embedding_tests()
# # tests/test_embeddings.py
# """
# Test suite for Azure embedding models
# """
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# from dotenv import load_dotenv
# from minichain.embeddings.azure import AzureOpenAIEmbeddings

# load_dotenv()

# SAMPLE_DOCS = ["Document one for testing.", "Document two, also for testing."]
# SAMPLE_QUERY = "What is the topic?"

# def test_azure_openai_embeddings():
#     """Test AzureOpenAIEmbeddings functionality"""
#     print("☁️ Testing Azure OpenAI Embeddings...")
#     print("=" * 60)
    
#     required_vars = [
#         "AZURE_OPENAI_ENDPOINT_EMBEDDINGS", 
#         "AZURE_OPENAI_EMBEDDINGS_API_KEY", 
#         "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME" # Specific for embeddings
#     ]
    
#     missing_vars = [var for var in required_vars if not os.getenv(var)]
#     if missing_vars:
#         print("⚠️ Skipping Azure embeddings test - required environment variables not set:")
#         for var in missing_vars:
#             print(f"   - {var}")
#         return True # Skip but don't fail the entire suite

#     try:
#         # Use a specific deployment name for embeddings
#         deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        
#         # Test 1: Initialization
#         embeddings = AzureOpenAIEmbeddings(
#             deployment_name=deployment_name, # type: ignore
#             model_name="text-embedding-3-small"
#         )
#         assert embeddings.deployment_name == deployment_name
#         print(f"✅ Initialization successful for deployment: '{deployment_name}'")
        
#         # Test 2: embed_query
#         query_vector = embeddings.embed_query(SAMPLE_QUERY)
#         # For text-embedding-3-small, the dimension is 1536
#         expected_dimension = 1536
#         print(f"✅ embed_query returned vector of length: {len(query_vector)} (Expected: {expected_dimension})")
#         assert isinstance(query_vector, list)
#         assert len(query_vector) == expected_dimension
        
#         # Test 3: embed_documents
#         doc_vectors = embeddings.embed_documents(SAMPLE_DOCS)
#         print(f"✅ embed_documents returned {len(doc_vectors)} vectors")
#         assert len(doc_vectors) == len(SAMPLE_DOCS)
#         assert all(len(vec) == expected_dimension for vec in doc_vectors)
        
#         print("\n🎉 Azure OpenAI Embeddings test passed!")
#         return True
        
#     except Exception as e:
#         print(f"❌ Azure OpenAI Embeddings test failed: {e}")
#         return False

# def run_embedding_tests():
#     """Run all embedding model tests"""
#     print("🧪 Testing Phase 5: Embeddings (Azure-Only)")
#     print("=" * 60)
    
#     result = test_azure_openai_embeddings()
    
#     print("-" * 60)
#     if result:
#         print("\n🎉 Phase 5 Complete - Azure embedding model working!")
#         print("✅ Ready to move to Phase 6: Vector Stores")
#     else:
#         print("\n⚠️ Azure embedding test failed. Please check your configuration and code.")
        
#     return result

# if __name__ == "__main__":
#     run_embedding_tests()