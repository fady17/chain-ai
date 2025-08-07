# src/minichain/output_parsers/pydantic_parser.py
"""
An output parser that uses Pydantic for type-safe parsing.
"""
import json
import re
from typing import Any, Type, TypeVar, Generic
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

class PydanticOutputParser(Generic[T]):
    """
    A generic class that parses LLM string output into a specific
    Pydantic model instance, T.
    """
    pydantic_object: Type[T]

    def __init__(self, pydantic_object: Type[T]):
        self.pydantic_object = pydantic_object

    def get_format_instructions(self) -> str:
        """
        Generates clear, human-readable instructions for the LLM on how to
        format its output as JSON, focusing on the required keys and their purpose.
        """
        schema = self.pydantic_object.model_json_schema()

        # Create a dictionary of { "field_name": "field_description" }
        # This is much clearer for the LLM than a full JSON schema.
        field_descriptions = {
            k: v.get("description", "")
            for k, v in schema.get("properties", {}).items()
        }

        # Build a robust instruction string that is less likely to be misinterpreted.
        instructions = [
            "Your response must be a single, valid JSON object.",
            "Do not include any other text, explanations, or markdown code fences.",
            "The JSON object must have the following keys:",
        ]
        for name, desc in field_descriptions.items():
            instructions.append(f'- "{name}": (Description: {desc})')
        
        instructions.append("\nPopulate the string values for these keys based on the user's query.")
        return "\n".join(instructions)


    def parse(self, text: str) -> T:
        """
        Parses the string output from an LLM into an instance of the target
        Pydantic model (T).
        """
        try:
            # Use regex to find the first '{' and last '}' to isolate the JSON blob.
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                raise json.JSONDecodeError("No JSON object found in the output.", text, 0)

            json_string = match.group(0)
            json_object = json.loads(json_string)
            
            return self.pydantic_object.model_validate(json_object)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(
                f"Failed to parse LLM output into {self.pydantic_object.__name__}. Error: {e}\n"
                f"Raw output:\n---\n{text}\n---"
            )
    def invoke(self, input: str, **kwargs: Any) -> T:
        return self.parse(input)
