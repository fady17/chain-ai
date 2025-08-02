# src/minichain/prompts/__init__.py
"""
This module provides a flexible and powerful system for creating and formatting
prompts for language models.

The key components exposed are:
    - PromptTemplate: For general-purpose string templating using the Jinja2 engine.
    - FewShotPromptTemplate: For constructing prompts that include examples to guide
      the model's behavior (in-context learning).
    - ChatPromptTemplate: For creating structured, multi-turn conversations suitable
      for chat-based models.
"""
from .base import BasePromptTemplate
from .prompt_template import PromptTemplate
from .few_shot import FewShotPromptTemplate
from .chat import ChatPromptTemplate

__all__ = [
    "BasePromptTemplate",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "ChatPromptTemplate",
]