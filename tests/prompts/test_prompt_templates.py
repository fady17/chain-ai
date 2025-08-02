# tests/prompts/test_prompt_template.py
"""
Unit tests for the Jinja2-powered `PromptTemplate`.
"""
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.prompts import PromptTemplate

def test_template_handles_simple_substitution():
    """
    Tests that the template correctly performs basic variable replacement,
    which is its most common use case.
    """
    # ARRANGE
    template = PromptTemplate(template="Hello, {{ name }}! Welcome to {{ project }}.")
    
    # ACT
    result = template.format(name="Amin", project="Mini-Chain")
    
    # ASSERT
    assert result == "Hello, Amin! Welcome to Mini-Chain."

def test_template_infers_variables_from_string():
    """
    Tests that the template can automatically and correctly identify all
    input variables from the template string using the Jinja2 parser.
    """
    # ARRANGE
    template = PromptTemplate(template="Context: {{ context }}\nQuestion: {{ question }}")
    
    # ACT & ASSERT
    assert set(template.input_variables) == {"context", "question"}

def test_template_renders_jinja_for_loop():
    """
    Tests the template's ability to handle advanced Jinja2 syntax,
    specifically a for-loop, confirming the engine upgrade was successful.
    """
    # ARRANGE
    template_string = "Here are the documents:\n{% for doc in documents %}- {{ doc }}\n{% endfor %}"
    template = PromptTemplate(template=template_string)
    
    # ACT
    result = template.format(documents=["Doc A", "Doc B", "Doc C"])
    
    # ASSERT
    assert "- Doc A" in result
    assert "- Doc B" in result
    assert "- Doc C" in result
    assert "Here are the documents:" in result

def test_template_raises_error_for_missing_variable():
    """
    Tests that the internal validation correctly raises a ValueError when
    a required variable is not provided during formatting.
    """
    # ARRANGE
    template = PromptTemplate(template="Hello, {{ name }}!")
    
    # ACT & ASSERT
    with pytest.raises(ValueError, match="Missing required input variables: \\['name'\\]"):
        template.format() # Missing the 'name' variable