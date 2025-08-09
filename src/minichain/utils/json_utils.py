# src/minichain/utils/json_utils.py

import re
import json
from typing import Any, Dict

def parse_json_markdown(text: str) -> Dict[str, Any]:
    """
    Parses a JSON object from a string, tolerating markdown code fences and
    other extraneous text. This version intelligently finds the boundaries of
    the first complete JSON object.
    """
    # 1. Try to find JSON within ```json ... ``` blocks first.
    # This is the most reliable method.
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from markdown block. Content: {json_str}") from e

    # 2. If no markdown, find the first '{' and extract a balanced object.
    try:
        start_index = text.index('{')
        brace_count = 0
        for i in range(start_index, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            
            if brace_count == 0:
                # We found the end of the first complete JSON object.
                json_str = text[start_index : i + 1]
                return json.loads(json_str)
        
        # If the loop finishes and brace_count is not 0, the JSON is incomplete.
        raise ValueError("Incomplete JSON object found in the output.")

    except ValueError:
        # This catches both text.index('{') failing and the incomplete JSON error.
        raise ValueError("No valid JSON object found in the output.")
    except json.JSONDecodeError as e: # type: ignore
        # This catches errors from the final json.loads() call
        raise ValueError(f"Failed to decode extracted JSON. Content: {json_str}") from e # type: ignore
