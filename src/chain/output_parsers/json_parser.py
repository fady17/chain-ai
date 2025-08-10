# src/chain/output_parsers/json_parser.py

from ..utils.json_utils import parse_json_markdown
from ..core import PrettyDict
from .base import BaseOutputParser

class JsonOutputParser(BaseOutputParser):
    """
    Parses LLM output for a JSON object, robustly handling markdown fences.
    Returns a pretty-printable dictionary.
    """

    def parse(self, text: str) -> PrettyDict:
        """Parses the string output into a PrettyDict."""
        try:
            parsed_json = parse_json_markdown(text)
            return PrettyDict(parsed_json)
        except ValueError as e:
            # Re-raise with the raw text for better debugging
            raise ValueError(
                f"Failed to parse JSON from LLM output. Error: {e}\n"
                f"Raw output:\n---\n{text}\n---"
            )

    def get_format_instructions(self) -> str:
        """Provides instructions for the LLM to return JSON."""
        return (
            "Your response MUST be a valid JSON object enclosed in a single ```json markdown code block. "
            "Do not include any other text or explanations."
        )