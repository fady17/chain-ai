# src/minichain/prompts/chat.py
"""
Implementation of a chat prompt template for conversational models.
"""
from typing import Dict, Any, List, Optional
from jinja2 import Environment, meta
from .base import BasePromptTemplate

class ChatPromptTemplate(BasePromptTemplate):
    """
    A prompt template designed for chat-based language models.

    Instead of producing a single string, this template generates a structured
    list of messages (e.g., `[{'role': 'system', 'content': '...'}, ...]`).
    This is the standard format for interacting with chat models, allowing for
    clear separation of system instructions, user queries, and AI responses.
    """
    
    def __init__(self, messages: List[Dict[str, Any]], input_variables: Optional[List[str]] = None):
        """
        Initializes the ChatPromptTemplate.

        Args:
            messages (List[Dict[str, Any]]): A list of message dictionaries. Each
                dictionary must have a 'role' key and a 'content' key. The 'content'
                can be a string with Jinja2 variables.
            input_variables (Optional[List[str]]): A list of expected variable names.
                If None, variables are inferred from the message content.
        """
        self.messages = messages
        self.jinja_env = Environment()

        # Pre-compile Jinja templates for each message's content for efficiency
        self.message_templates: List[Dict[str, Any]] = [
            {"role": msg["role"], "template": self.jinja_env.from_string(msg.get("content", ""))}
            for msg in messages
        ]
            
        if input_variables is None:
            input_variables = self._extract_variables_from_messages(messages)
        
        super().__init__(input_variables=input_variables)
    
    def _extract_variables_from_messages(self, messages: List[Dict[str, str]]) -> List[str]:
        """Extracts all unique Jinja2 variables from all message templates."""
        all_vars = set()
        for msg in messages:
            content = msg.get("content", "")
            ast = self.jinja_env.parse(content)
            all_vars.update(meta.find_undeclared_variables(ast))
        return list(all_vars)

    def format(self, **kwargs: Any) -> List[Dict[str, str]]:
        """
        Formats the chat messages with the provided variables.

        Returns:
            A list of formatted message dictionaries, ready to be sent to a
            chat model API.
        """
        self._validate_variables(kwargs)
        
        return [
            {"role": msg["role"], "content": msg["template"].render(**kwargs)}
            for msg in self.message_templates
        ]