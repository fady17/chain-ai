# tests/test_chat_models.py
"""
Test suite for chat models
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from minichain.core.types import SystemMessage, HumanMessage, AIMessage
from minichain.chat_models.implementations import (
    
    AzureOpenAIChatModel, 
   
)

from dotenv import load_dotenv
load_dotenv()
# tests/test_azure_chat_models.py
"""
Test suite for Azure OpenAI chat models only
"""

import os
import sys
sys.path.append('..')





def test_azure_openai_chat_model():
    """Test Azure OpenAI chat model with comprehensive scenarios"""
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these in your environment or .env file")
        return False
        
    print("☁️ Testing Azure OpenAI Chat Model...")
    print("=" * 50)
    
    try:
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Test 1: Basic model initialization
        print("1️⃣ Testing model initialization...")
        model = AzureOpenAIChatModel(
            deployment_name=deployment_name, # type: ignore
            model_name="gpt-4",  # This can be different from deployment name
            temperature=0,
            max_tokens=50  # Limit tokens for testing
        )
        print("✅ Model initialized successfully")
        
        # Test 2: String input
        print("\n2️⃣ Testing string input...")
        response1 = model.invoke("Say exactly: 'Azure test 1 passed'")
        print(f"✅ Response: {response1}")
        
        # Test 3: Message list input
        print("\n3️⃣ Testing message list input...")
        messages = [
            SystemMessage("You are a helpful assistant that answers briefly."),
            HumanMessage("What is the capital of France? Answer with just the city name.")
        ]
        response2 = model.invoke(messages)
        print(f"✅ Response: {response2}")
        
        # Test 4: Conversation flow
        print("\n4️⃣ Testing conversation flow...")
        conversation = [
            SystemMessage("You are a math tutor."),
            HumanMessage("What is 5 + 3?"),
            AIMessage("5 + 3 = 8"),
            HumanMessage("What about 8 + 2?")
        ]
        response3 = model.invoke(conversation)
        print(f"✅ Response: {response3}")
        
        print("\n🎉 All Azure OpenAI tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def test_azure_parameters():
    """Test Azure-specific parameter handling"""
    print("\n🔧 Testing Azure parameter handling...")
    
    try:
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Test with different parameters
        model = AzureOpenAIChatModel(
            deployment_name=deployment_name, # type: ignore
            temperature=0.7,
            max_tokens=30,
            top_p=0.9,
            frequency_penalty=0.1
        )
        
        response = model.invoke("Count from 1 to 5")
        print(f"✅ Response with custom parameters: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        return False


def run_azure_tests():
    """Run all Azure OpenAI tests"""
    print("🧪 Testing Azure OpenAI Implementation")
    print("=" * 60)
    
    print("Required Environment Variables:")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- AZURE_OPENAI_API_KEY") 
    print("- AZURE_OPENAI_DEPLOYMENT_NAME")
    print("- AZURE_OPENAI_API_VERSION (optional, defaults to 2024-12-01-preview)")
    print()
    
    # Check environment
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        print(f"🔗 Endpoint: {endpoint}")
    
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    if deployment:
        print(f"🚀 Deployment: {deployment}")
        
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    print(f"📅 API Version: {api_version}")
    print()
    
    # Run tests
    basic_test = test_azure_openai_chat_model()
    param_test = test_azure_parameters()
    
    if basic_test and param_test:
        print("\n🎉 Phase 2 Complete - Azure OpenAI chat model working perfectly!")
        print("✅ Ready to move to Phase 3: Prompt Templates")
    else:
        print("\n⚠️ Some tests failed - please check your Azure configuration")
    
    return basic_test and param_test


if __name__ == "__main__":
    run_azure_tests()