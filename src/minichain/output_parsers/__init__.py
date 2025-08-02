# src/minichain/output_parsers/__init__.py
"""
This module provides classes for parsing the string output of language models
into structured, type-safe data formats.

Parsing is a critical step in making LLM outputs reliable and usable in
downstream application logic.

The key component exposed is:
    - PydanticOutputParser: A powerful and robust parser that leverages Pydantic
      models to define the desired output schema and validate the LLM's response.
"""
from .pydantic_parser import PydanticOutputParser

# The __all__ variable explicitly defines the public API of this module.
# When a user does `from minichain.output_parsers import *`, only the names
# listed here will be imported.
__all__ = [
    "PydanticOutputParser",
]