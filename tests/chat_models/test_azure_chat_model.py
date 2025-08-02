
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
# tests/chat_models/test_azure_chat_model.py
"""
Unit tests for the `AzureOpenAIChatModel` class.
"""

import pytest
import json
import re # Import the regular expression module
from dotenv import load_dotenv
from minichain.chat_models import AzureOpenAIChatModel
from minichain.core.types import HumanMessage, SystemMessage

# Load environment variables for local testing
load_dotenv()

# --- Test Fixtures and Configuration ---

AZURE_CREDS_AVAILABLE = all(
    os.getenv(var) for var in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
    ]
)
TEST_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")

requires_azure_creds = pytest.mark.skipif(
    not AZURE_CREDS_AVAILABLE,
    reason="Azure chat model credentials not found in environment variables."
)

# --- Test Functions ---

@requires_azure_creds
def test_azure_model_initialization_succeeds():
    """
    Tests that the AzureOpenAIChatModel initializes without errors when
    credentials are provided.
    """
    AzureOpenAIChatModel(deployment_name=TEST_DEPLOYMENT_NAME)

def test_azure_model_initialization_fails_without_creds():
    """
    Tests that a ValueError is raised if the class is initialized
    without the necessary environment variables.
    """
    original_key = os.environ.pop("AZURE_OPENAI_API_KEY", None)
    with pytest.raises(ValueError, match="environment variables must be set"):
        AzureOpenAIChatModel(deployment_name="test")
    if original_key:
        os.environ["AZURE_OPENAI_API_KEY"] = original_key

@requires_azure_creds
def test_azure_model_invoke_with_string_input():
    """
    Tests the model's ability to generate a response from a simple string prompt.
    """
    model = AzureOpenAIChatModel(deployment_name=TEST_DEPLOYMENT_NAME, temperature=0)
    prompt = "What is the capital of France? Respond with only the name of the city."
    response = model.invoke(prompt)
    assert isinstance(response, str)
    assert "Paris" in response
    assert len(response.split()) < 5

@requires_azure_creds
def test_azure_model_invoke_with_message_list():
    """
    Tests the model's ability to process a list of Pydantic Message objects
    and respect the system prompt's instructions.
    """
    # ARRANGE
    model = AzureOpenAIChatModel(deployment_name=TEST_DEPLOYMENT_NAME, temperature=0)
    messages = [
        SystemMessage(content="You are a helpful assistant that always responds in JSON format. The JSON should contain the country and its capital."),
        HumanMessage(content="Provide the capital of Canada.")
    ]
    
    # ACT
    response = model.invoke(messages)
    
    # ASSERT
    assert isinstance(response, str)
    
    # --- FIX: Use a robust method to parse the JSON from the response ---
    # 1. Extract the JSON blob using regex, ignoring surrounding text/fences.
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    assert json_match is not None, "The model did not return a JSON object."
    
    # 2. Parse the extracted string into a Python dictionary.
    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        pytest.fail("The model's output was not valid JSON.")
        
    # 3. Assert against the dictionary's contents.
    # This is immune to whitespace, newlines, and key order.
    assert "capital" in data or "city" in data
    assert data.get("capital", data.get("city")).lower() == "ottawa"
    assert data.get("country", "").lower() == "canada"