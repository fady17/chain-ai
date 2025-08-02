# mini_langchain/prompts/base.py
"""
Base prompt template interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BasePromptTemplate(ABC):
    """Abstract base class for all prompt templates"""
    
    def __init__(self, input_variables: List[str]):
        self.input_variables = input_variables
    
    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the prompt with given variables"""
        pass
    
    def invoke(self, variables: Dict[str, Any]) -> str:
        """Alternative method name for consistency with LangChain"""
        return self.format(**variables)
    
    def _validate_variables(self, variables: Dict[str, Any]) -> None:
        """Validate that all required variables are provided"""
        missing = set(self.input_variables) - set(variables.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(input_variables={self.input_variables})"