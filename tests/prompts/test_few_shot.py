# tests/prompts/test_few_shot.py
"""
Unit tests for the `FewShotPromptTemplate`.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.prompts import PromptTemplate, FewShotPromptTemplate

def test_few_shot_template_assembles_prompt_correctly():
    """
    Tests that the template correctly assembles the final prompt string from
    the prefix, formatted examples, and suffix.
    """
    # ARRANGE
    examples = [
        {"query": "How are you?", "answer": "I am fine."},
        {"query": "What's up?", "answer": "Not much."}
    ]
    
    example_prompt = PromptTemplate(template="User: {{ query }}\nAI: {{ answer }}")
    
    # The `input_variables` should only contain the variables for the final suffix
    fs_template = FewShotPromptTemplate(
        prefix="The following is a conversation with an AI.",
        examples=examples,
        example_prompt=example_prompt,
        suffix="User: {{ user_input }}\nAI:",
        input_variables=["user_input"],
        example_separator="\n---\n"
    )
    
    # ACT
    result = fs_template.format(user_input="Hello?")
    
    # ASSERT
    expected_string = (
        "The following is a conversation with an AI.\n---\n"
        "User: How are you?\nAI: I am fine.\n---\n"
        "User: What's up?\nAI: Not much.\n---\n"
        "User: Hello?\nAI:"
    )
    assert result == expected_string

def test_few_shot_template_works_with_no_prefix():
    """
    Tests that the template functions correctly even when an optional
    prefix is not provided.
    """
    # ARRANGE
    examples = [{"text": "some example"}]
    example_prompt = PromptTemplate(template="Example: {{ text }}")
    fs_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="Input: {{ user_input }}",
        input_variables=["user_input"]
        # No prefix provided
    )
    
    # ACT
    result = fs_template.format(user_input="new input")
    
    # ASSERT
    expected_string = "Example: some example\n\nInput: new input"
    assert result == expected_string
    assert not result.startswith("\n\n")

def test_few_shot_template_with_jinja_in_suffix():
    """
    Tests that the suffix can also contain Jinja2 logic.
    """
    # ARRANGE
    examples = [] # No examples needed for this test
    example_prompt = PromptTemplate(template="")
    fs_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix="The user is an {% if is_admin %}admin{% else %}guest{% endif %}.",
        input_variables=["is_admin"]
    )
    
    # ACT
    result_admin = fs_template.format(is_admin=True)
    result_guest = fs_template.format(is_admin=False)
    
    # ASSERT
    assert result_admin == "The user is an admin."
    assert result_guest == "The user is an guest."