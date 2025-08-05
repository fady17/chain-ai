# tests/chat_models/test_local_chat_model.py
"""
Unit tests for the `LocalChatModel` class, validating both invoke and stream methods.
"""
import sys
import os
import pytest
import socket

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.chat_models import LocalChatModel, LocalChatConfig
from minichain.core.types import HumanMessage, SystemMessage

# --- Test Fixtures and Configuration ---

def is_server_running(host='localhost', port=1234):
    """Checks if a local server is running on the specified host and port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

requires_local_server = pytest.mark.skipif(
    not is_server_running(),
    reason="Local chat model server (e.g., LM Studio) not running on port 1234."
)

@pytest.fixture
def local_chat_config():
    """Provides a default LocalChatConfig for tests."""
    return LocalChatConfig(temperature=0.0)

# --- Test Functions ---

@requires_local_server
def test_local_model_initialization(local_chat_config):
    """
    Tests that the LocalChatModel class can be initialized with a config object.
    """
    LocalChatModel(config=local_chat_config)

@requires_local_server
def test_local_model_invoke_with_string(local_chat_config):
    """
    Tests the blocking `invoke` method with a simple string prompt.
    """
    model = LocalChatModel(config=local_chat_config)
    prompt = "What are the three primary colors? Answer with a comma-separated list."
    
    response = model.invoke(prompt)
    
    assert isinstance(response, str)
    response_lower = response.lower()
    assert "red" in response_lower
    assert "blue" in response_lower
    assert "yellow" in response_lower

@requires_local_server
def test_local_model_invoke_with_messages(local_chat_config):
    """
    Tests the blocking `invoke` method with a list of Pydantic Message objects.
    """
    # ARRANGE
    local_chat_config.temperature = 0.7
    model = LocalChatModel(config=local_chat_config)
    messages = [
        SystemMessage(content="You are a pirate. Answer all questions in character."),
        HumanMessage(content="What is a computer mouse?")
    ]
    
    # ACT
    response = model.invoke(messages)
    
    # ASSERT
    assert isinstance(response, str)
    
    # --- FIX: Make the assertion more robust and less brittle ---
    # Create a set of possible pirate-themed words to check for.
    pirate_keywords = {"arr", "matey", "ye", "landlubber", "treasure", "shiver"}
    response_lower = response.lower()
    
    # Check if any of the keywords are present in the response.
    assert any(keyword in response_lower for keyword in pirate_keywords), \
        f"Response did not contain expected pirate keywords. Response: '{response}'"

@requires_local_server
def test_local_model_streams_response(local_chat_config):
    """
    Tests the new `stream` method to ensure it yields response chunks.
    """
    model = LocalChatModel(config=local_chat_config)
    prompt = "Write a two-sentence story about a robot."
    
    stream = model.stream(prompt)
    chunks = list(stream)
    full_response = "".join(chunks)
    
    assert len(chunks) > 1
    assert isinstance(chunks[0], str)
    assert "robot" in full_response.lower()