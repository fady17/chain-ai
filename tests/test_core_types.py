# tests/test_core_types.py
"""
Unit tests for the Pydantic-based core data structures in `minichain.core.types`.
These tests validate the behavior of the fundamental building blocks of the framework.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import json
from pydantic import ValidationError

# The components we are testing
from minichain.core.types import Document, HumanMessage, SystemMessage, AIMessage

def test_document_creation_with_all_fields():
    """
    Tests successful Document creation when both page_content and metadata are provided.
    """
    meta = {"source": "test.txt", "page_number": 1}
    doc = Document(page_content="This is the content of the document.", metadata=meta)
    
    assert doc.page_content == "This is the content of the document."
    assert doc.metadata == meta
    assert doc.metadata["source"] == "test.txt"

def test_document_metadata_is_optional_and_defaults_correctly():
    """
    Tests that a Document's metadata attribute defaults to an empty dictionary
    if it is not provided during instantiation.
    """
    doc = Document(page_content="This document has no metadata.")
    
    assert isinstance(doc.metadata, dict)
    assert doc.metadata == {}

def test_document_serializes_to_valid_json():
    """
    Tests that a Document object can be correctly serialized to a JSON string
    using Pydantic's `.model_dump_json()` method.
    """
    doc = Document(page_content="serialize me", metadata={"id": 123, "active": True})
    json_output = doc.model_dump_json()
    
    data = json.loads(json_output)
    assert data["page_content"] == "serialize me"
    assert data["metadata"]["id"] == 123
    assert data["metadata"]["active"] is True

def test_message_creation_succeeds_with_keyword_arguments():
    """
    Tests that all Message types can be successfully instantiated using
    the required 'content' keyword argument.
    """
    human_msg = HumanMessage(content="Hello AI.")
    system_msg = SystemMessage(content="You are a helpful assistant.")
    ai_msg = AIMessage(content="Hello! How can I help you today?")
    
    assert human_msg.content == "Hello AI."
    assert system_msg.content == "You are a helpful assistant."
    assert ai_msg.content == "Hello! How can I help you today?"

def test_message_creation_fails_without_keyword_argument():
    """

    Tests that instantiating a Pydantic BaseModel with positional arguments
    correctly raises a TypeError.
    """
    # --- FIX: Pydantic V2 raises TypeError for positional args, not ValidationError ---
    with pytest.raises(TypeError, match="takes 1 positional argument but 2 were given"):
        # This is the incorrect way and must fail.
        # The error message is about __init__ getting 'self' + your argument.
        HumanMessage("This will raise an error.") # type: ignore

def test_message_type_property_returns_correct_class_name():
    """
    Tests that the `.type` property of each message class instance
    correctly returns the name of its class as a string.
    """
    human_msg = HumanMessage(content="...")
    system_msg = SystemMessage(content="...")
    ai_msg = AIMessage(content="...")
    
    assert human_msg.type == "HumanMessage"
    assert system_msg.type == "SystemMessage"
    assert ai_msg.type == "AIMessage"