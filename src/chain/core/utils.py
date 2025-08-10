# src/chain/core/utils.py

import json

class PrettyDict(dict):
    """
    A dictionary subclass that provides a pretty-printed representation.

    When this object is displayed in a REPL or Jupyter notebook, it will be
    formatted as a multi-line JSON string, making it much more readable.
    """
    def __repr__(self) -> str:
        """
        Overrides the default representation to pretty-print the dictionary.
        """
        # Use json.dumps with an indent of 2 for nice formatting.
        return json.dumps(self, indent=2)