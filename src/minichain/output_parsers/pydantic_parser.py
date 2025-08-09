# src/minichain/output_parsers/pydantic_parser.py

import json
from typing import Any, Type, TypeVar, Generic
from pydantic import BaseModel, ValidationError

from .base import BaseOutputParser
from ..utils.json_utils import parse_json_markdown

T = TypeVar("T", bound=BaseModel)

class PydanticOutputParser(BaseOutputParser, Generic[T]):
    """
    Parses LLM output into a Pydantic model using a robust JSON extractor.
    """
    pydantic_object: Type[T]

    def __init__(self, pydantic_object: Type[T]):
        self.pydantic_object = pydantic_object

    def get_format_instructions(self) -> str:
        """Generates clear instructions for the LLM, including the JSON schema."""
        schema = self.pydantic_object.model_json_schema()
        # Clean up the schema for a clearer prompt
        if "title" in schema: del schema["title"]
        if "type" in schema: del schema["type"]
        schema_str = json.dumps(schema)

        return (
            f"Your response MUST be a valid JSON object conforming to the following JSON schema.\n"
            f"Enclose the JSON in a single ```json markdown code block.\n"
            f"Schema:\n{schema_str}"
        )

    def parse(self, text: str) -> T:
        """
        Parses the string output and validates it with the Pydantic model.
        """
        try:
            # Use our new, robust utility to get the JSON dictionary
            json_dict = parse_json_markdown(text)
            
            # Validate the dictionary with the Pydantic model
            return self.pydantic_object.model_validate(json_dict)
        
        except (ValueError, ValidationError) as e:
            # If parsing or validation fails, raise a detailed error
            raise ValueError(
                f"Failed to parse Pydantic model '{self.pydantic_object.__name__}'. Error: {e}\n"
                f"Raw LLM output:\n---\n{text}\n---"
            )
