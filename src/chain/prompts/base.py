# src/chain/prompts/base.py
"""
Defines the abstract base class for all prompt templates in the Mini-Chain framework.
This ensures a consistent interface for formatting and validation across different
prompting strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""
    def __init__(self, input_variables: List[str]):
        self.input_variables = input_variables

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        pass

    def invoke(self, variables: Dict[str, Any], **kwargs: Any) -> str:
        """A convenience method to format the prompt using a dictionary."""
        return self.format(**variables)

    # Make the method signature consistent. We won't use the second parameter here.
    def _validate_variables(self, variables: Dict[str, Any]) -> None:
        """Internal method to ensure all required variables are provided for formatting."""
        missing = set(self.input_variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required input variables: {sorted(list(missing))}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(input_variables={self.input_variables})"
    
    def to_string(self) -> str:
        raise NotImplementedError("This prompt template must implement `to_string()` explicitly.")

