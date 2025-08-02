# tests/prompts/test_chat.py
"""
Unit tests for the `ChatPromptTemplate`.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from minichain.prompts import ChatPromptTemplate

def test_chat_template_formats_messages_with_variables():
    """
    Tests that the template correctly substitutes variables and produces a
    structured list of message dictionaries.
    """
    # ARRANGE
    template = ChatPromptTemplate(
        messages=[
            {"role": "system", "content": "You are a helpful {{ persona }}."},
            {"role": "user", "content": "Can you tell me about the {{ component }} in Mini-Chain?"}
        ]
    )
    
    # ACT
    result = template.format(persona="technical assistant", component="PydanticOutputParser")
    
    # ASSERT
    expected = [
        {"role": "system", "content": "You are a helpful technical assistant."},
        {"role": "user", "content": "Can you tell me about the PydanticOutputParser in Mini-Chain?"}
    ]
    assert result == expected

def test_chat_template_infers_variables_from_multiple_messages():
    """
    Tests that the template can correctly identify all unique input variables
    spread across multiple messages in the template structure.
    """
    # ARRANGE
    template = ChatPromptTemplate(
        messages=[
            {"role": "system", "content": "Your persona is {{ persona }}."},
            {"role": "user", "content": "My question is about {{ topic }}."}
        ]
    )
    
    # ACT & ASSERT
    assert set(template.input_variables) == {"persona", "topic"}

def test_chat_template_handles_jinja_logic():
    """

    Tests that the Jinja2 engine is active within the message content,
    allowing for advanced logic like conditionals.
    """
    # ARRANGE
    template_string = "The user is an {% if is_admin %}admin{% else %}end-user{% endif %}."
    template = ChatPromptTemplate(
        messages=[
            {"role": "system", "content": template_string}
        ]
    )
    
    # ACT
    result_admin = template.format(is_admin=True)
    result_user = template.format(is_admin=False)
    
    # ASSERT
    assert result_admin[0]['content'] == "The user is an admin."
    assert result_user[0]['content'] == "The user is an end-user."