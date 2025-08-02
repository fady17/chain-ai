# # src/minichain/output_parsers/pydantic_parser.py
# """
# An output parser that uses Pydantic for robust, type-safe parsing.
# """
# import json
# from typing import Type, TypeVar
# from pydantic import BaseModel, ValidationError

# T = TypeVar("T", bound=BaseModel)

# class PydanticOutputParser(BaseModel):
#     """
#     Parses LLM string output into a Pydantic model instance.
#     """
#     pydantic_object: Type[T]

#     def get_format_instructions(self) -> str:
#         """
#         Generates instructions for the LLM on how to format its output as JSON
#         that conforms to the Pydantic model's schema.
#         """
#         schema = self.pydantic_object.model_json_schema()
        
#         # Remove properties that are not useful for the LLM
#         if "title" in schema:
#             del schema["title"]
#         if "description" in schema:
#             del schema["description"]
        
#         # Simple schema representation for the prompt
#         reduced_schema = {
#             k: v.get("type", v.get("anyOf", "unknown")) for k, v in schema.get("properties", {}).items()
#         }

#         return (
#             "Please respond with a JSON object formatted according to the following schema.\n"
#             "Ensure your response is a single, valid JSON blob, and nothing else.\n"
#             f"JSON Schema: {json.dumps(reduced_schema)}"
#         )

#     def parse(self, text: str) -> T:
#         """

#         Parses the string output from an LLM into an instance of the target
#         Pydantic model.
#         """
#         try:
#             # Attempt to find a valid JSON blob in the text
#             match = json.loads(text.strip())
#             return self.pydantic_object.model_validate(match)
#         except (json.JSONDecodeError, ValidationError) as e:
#             raise ValueError(f"Failed to parse LLM output. Error: {e}\nRaw output:\n---\n{text}\n---")