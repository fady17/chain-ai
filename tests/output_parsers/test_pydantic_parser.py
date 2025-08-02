# tests/output_parsers/test_pydantic_parser.py
"""
Unit tests for the generic `PydanticOutputParser`.

These tests validate the parser's ability to generate correct formatting
instructions and to safely parse various forms of LLM string output into
a structured Pydantic model.
"""
import pytest
from pydantic import BaseModel, Field, ValidationError
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# The component we are testing
from minichain.output_parsers.pydantic_parser import PydanticOutputParser

# Define a Pydantic model that will be the target for parsing in our tests.
class Actor(BaseModel):
    """A simple Pydantic model representing an actor."""
    name: str = Field(description="The full name of the actor.")
    film_count: int = Field(description="The number of films they have appeared in.")

def test_parser_generates_clear_format_instructions():
    """
    Tests that the parser can generate a string of instructions for an LLM
    that clearly describes the desired JSON format based on the Pydantic model.
    """
    # ARRANGE
    parser = PydanticOutputParser(pydantic_object=Actor)
    
    # ACT
    instructions = parser.get_format_instructions()
    
    # ASSERT
    assert isinstance(instructions, str)
    assert '"name"' in instructions  # Check for field names
    assert '"film_count"' in instructions
    assert 'JSON' in instructions   # Check for the keyword "JSON"

def test_parser_succeeds_on_valid_json_string():
    """
    Tests the primary success case where the LLM provides a clean, valid
    JSON string that matches the Pydantic schema.
    """
    # ARRANGE
    parser = PydanticOutputParser(pydantic_object=Actor)
    valid_json_string = '{"name": "Tom Hanks", "film_count": 92}'
    
    # ACT
    result = parser.parse(valid_json_string)
    
    # ASSERT
    assert isinstance(result, Actor)
    assert result.name == "Tom Hanks"
    assert result.film_count == 92

def test_parser_handles_llm_output_with_markdown_fences():
    """
    Tests a common failure mode where LLMs wrap their JSON output in markdown
    code fences (```json ... ```). The parser should be robust enough to strip these.
    """
    # ARRANGE
    parser = PydanticOutputParser(pydantic_object=Actor)
    text_with_fences = '```json\n{"name": "Meryl Streep", "film_count": 85}\n```'
    
    # ACT
    result = parser.parse(text_with_fences)
    
    # ASSERT
    assert result.name == "Meryl Streep"
    assert result.film_count == 85

def test_parser_handles_llm_output_with_leading_text():
    """
    Tests another common failure mode where LLMs add conversational text
    before the JSON object. The parser should be able to find and parse the JSON.
    """
    # ARRANGE
    parser = PydanticOutputParser(pydantic_object=Actor)
    text_with_leading_words = 'Sure, here is the JSON you requested:\n{"name": "Denzel Washington", "film_count": 65}'
    
    # ACT
    result = parser.parse(text_with_leading_words)

    # ASSERT
    assert result.name == "Denzel Washington"
    assert result.film_count == 65

def test_parser_raises_valueerror_for_malformed_json():
    """
    Tests that the parser raises a descriptive `ValueError` when the input
    string is not structurally valid JSON.
    """
    # ARRANGE
    parser = PydanticOutputParser(pydantic_object=Actor)
    malformed_json = '{"name": "Invalid", "film_count": }'  # Missing value
    
    # ACT & ASSERT
    with pytest.raises(ValueError) as exc_info:
        parser.parse(malformed_json)
    
    # Check that the error message is helpful for debugging
    assert "Failed to parse LLM output" in str(exc_info.value)
    assert "Raw output" in str(exc_info.value)

def test_parser_raises_valueerror_for_schema_mismatch():
    """
    Tests that the parser raises a `ValueError` when the input is valid JSON
    but does not conform to the Pydantic model's schema (e.g., missing a required field).
    """
    # ARRANGE
    parser = PydanticOutputParser(pydantic_object=Actor)
    json_with_missing_field = '{"name": "Leonardo DiCaprio"}' # Missing film_count
    
    # ACT & ASSERT
    with pytest.raises(ValueError) as exc_info:
        parser.parse(json_with_missing_field)
        
    assert "Field required" in str(exc_info.value) # Pydantic's validation error message
    assert "film_count" in str(exc_info.value)