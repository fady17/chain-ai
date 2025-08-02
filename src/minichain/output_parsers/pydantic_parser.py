# src/minichain/output_parsers/pydantic_parser.py
"""
An output parser that uses Pydantic for robust, type-safe parsing.
"""
import json
import re
from typing import Type, TypeVar, Generic
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
# # src/minichain/output_parsers/pydantic_parser.py
# """
# An output parser that uses Pydantic for robust, type-safe parsing.
# """
# import json
# import re # Import the regular expression module
# from typing import Type, TypeVar, Generic
# from pydantic import BaseModel, ValidationError

# # Define a TypeVar 'T' that is bound to Pydantic's BaseModel.
# T = TypeVar("T", bound=BaseModel)

# class PydanticOutputParser(Generic[T]):
#     """
#     A generic class that parses LLM string output into a specific
#     Pydantic model instance, T.
#     """
#     pydantic_object: Type[T]

#     def __init__(self, pydantic_object: Type[T]):
#         self.pydantic_object = pydantic_object

#     def get_format_instructions(self) -> str:
#         """
#         Generates instructions for the LLM on how to format its output as JSON
#         that conforms to the Pydantic model's schema.
#         """
#         schema = self.pydantic_object.model_json_schema()
        
#         # Simplify the schema for the prompt to make it clearer for the LLM
#         properties = {
#             k: {"description": v.get("description"), "type": v.get("type")}
#             for k, v in schema.get("properties", {}).items()
#         }

#         return (
#             "Please respond with a JSON object that strictly adheres to the following schema. "
#             "Do not include any other text, explanations, or code fences around the response.\n"
#             f"JSON Schema: {json.dumps(properties, indent=2)}"
#         )

#     def parse(self, text: str) -> T:
#         """
#         Parses the string output from an LLM into an instance of the target
#         Pydantic model (T).

#         This method is designed to be robust against common LLM formatting issues,
#         such as conversational text and markdown code fences.
#         """
#         try:
#             # --- FIX: Use a regex to find the first '{' and last '}' ---
#             # This will extract the JSON blob from a larger string.
#             # re.DOTALL makes '.' match newlines as well.
#             match = re.search(r"\{.*\}", text, re.DOTALL)
#             if not match:
#                 raise json.JSONDecodeError("No JSON object found in the output.", text, 0)

#             json_string = match.group(0)
            
#             # Now parse the extracted JSON string
#             json_object = json.loads(json_string)
            
#             # Validate and create the Pydantic model instance
#             return self.pydantic_object.model_validate(json_object)
#         except (json.JSONDecodeError, ValidationError) as e:
#             # Raise a more informative error for easier debugging
#             raise ValueError(
#                 f"Failed to parse LLM output into {self.pydantic_object.__name__}. Error: {e}\n"
#                 f"Raw output:\n---\n{text}\n---"
#             )