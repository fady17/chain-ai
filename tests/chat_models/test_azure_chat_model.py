# tests/chat_models/test_azure_chat_model.py
"""
Unit tests for the `AzureOpenAIChatModel`, validating invoke and stream methods.
"""
import sys
import os
import pytest
import json
import re
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from chain.chat_models import AzureOpenAIChatModel, AzureChatConfig
from chain.core.types import HumanMessage, SystemMessage

load_dotenv()

# --- Test Fixtures and Configuration ---

AZURE_CREDS_AVAILABLE = all(
    os.getenv(var) for var in [
        "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
)
TEST_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

requires_azure_creds = pytest.mark.skipif(
    not AZURE_CREDS_AVAILABLE,
    reason="Azure chat model credentials not found in environment variables."
)

@pytest.fixture
def azure_chat_config():
    """Provides a default AzureChatConfig for tests."""
    return AzureChatConfig(
        deployment_name=TEST_DEPLOYMENT_NAME,
        temperature=0.0
    )

# --- Test Functions ---

@requires_azure_creds
def test_azure_model_initialization(azure_chat_config):
    """
    Tests that the AzureOpenAIChatModel initializes without errors using a config object.
    """
    AzureOpenAIChatModel(config=azure_chat_config)

@requires_azure_creds
def test_azure_model_invoke_with_string(azure_chat_config):
    """
    Tests the blocking `invoke` method with a simple string prompt.
    """
    model = AzureOpenAIChatModel(config=azure_chat_config)
    prompt = "What is the capital of France? Respond with only the name of the city."
    
    response = model.invoke(prompt)
    
    assert "Paris" in response
    assert len(response.split()) < 5

@requires_azure_creds
def test_azure_model_invoke_with_messages_and_json_output(azure_chat_config):
    """
    Tests the `invoke` method's ability to follow a system prompt for JSON output.
    """
    model = AzureOpenAIChatModel(config=azure_chat_config)
    messages = [
        SystemMessage(content="You are a helpful assistant that always responds in JSON format. The JSON should contain the country and its capital."),
        HumanMessage(content="Provide the capital of Canada.")
    ]
    
    response = model.invoke(messages)
    
    # Use a robust regex to extract the JSON blob
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    assert json_match, "Model response did not contain a JSON object."
    
    data = json.loads(json_match.group(0))
    assert data.get("capital", data.get("city", "")).lower() == "ottawa"

@requires_azure_creds
def test_azure_model_streams_response(azure_chat_config):
    """
    Tests the new `stream` method to ensure it yields response chunks from Azure.
    """
    # ARRANGE
    model = AzureOpenAIChatModel(config=azure_chat_config)
    prompt = "Briefly, what is Mini-Chain?"
    
    # ACT
    stream = model.stream(prompt)
    chunks = list(stream)
    full_response = "".join(chunks)
    
    # ASSERT
    assert len(chunks) > 1, "A successful stream should yield multiple chunks."
    assert isinstance(chunks[0], str)
    assert "framework" in full_response.lower() or "library" in full_response.lower()