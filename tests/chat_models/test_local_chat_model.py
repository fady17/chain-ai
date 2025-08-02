# tests/chat_models/test_local_chat_model.py
"""
Unit tests for the `LocalChatModel` class.

These tests validate the ability to connect to a local, OpenAI-compatible
server (like LM Studio) and generate chat completions.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import pytest
import socket
from minichain.chat_models import LocalChatModel
from minichain.core.types import HumanMessage, SystemMessage

# --- Test Fixtures and Configuration ---

def is_server_running(host='localhost', port=1234):
    """Checks if a server is running on the specified host and port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0

# Pytest marker to skip tests if the local server is not running
requires_local_server = pytest.mark.skipif(
    not is_server_running(),
    reason="Local chat model server (e.g., LM Studio) not running on port 1234."
)

# --- Test Functions ---

@requires_local_server
def test_local_model_initialization_succeeds():
    """
    Tests that the LocalChatModel class can be initialized without errors.
    """
    # No assert needed, the test passes if it doesn't raise an exception
    LocalChatModel()

@requires_local_server
def test_local_model_invoke_with_string_input():
    """
    Tests the model's ability to generate a response from a simple string prompt.
    """
    # ARRANGE
    model = LocalChatModel(temperature=0)
    prompt = "What are the three primary colors?"
    
    # ACT
    response = model.invoke(prompt)
    
    # ASSERT
    assert isinstance(response, str)
    response_lower = response.lower()
    assert "red" in response_lower
    assert "blue" in response_lower
    assert "yellow" in response_lower

@requires_local_server
def test_local_model_invoke_with_message_list():
    """
    Tests the model's ability to process a list of Pydantic Message objects
    and respect the system prompt's persona.
    """
    # ARRANGE
    model = LocalChatModel(temperature=0.7) # Higher temp for more creative persona
    messages = [
        SystemMessage(content="You are a pirate. You answer all questions in character."),
        HumanMessage(content="What is the main purpose of a CPU in a computer?")
    ]
    
    # ACT
    response = model.invoke(messages)
    
    # ASSERT
    assert isinstance(response, str)
    # Check for pirate-like words
    response_lower = response.lower()
    assert "arrr" in response_lower or "matey" in response_lower or "treasure" in response_lower or "ship" in response_lower