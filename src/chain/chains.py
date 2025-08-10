# src/chain/chains.py

from typing import Any, Callable, List

class Chain:
    """A sequence of runnable components that are invoked in order."""
    
    def __init__(self, first_step: Callable[..., Any]):
        self.steps: List[Callable[..., Any]] = [first_step]

    def __or__(self, next_step: Callable[..., Any]) -> 'Chain':
        """Appends a new step to the chain using the `|` operator."""
        self.steps.append(next_step)
        return self

    def invoke(self, input_data: Any, **kwargs: Any) -> Any:
        """Executes the chain by passing the output of each step to the next."""
        # The first step might take a dictionary
        current_result = self.steps[0].invoke(input_data, **kwargs) # type: ignore
        
        # Subsequent steps take the output of the previous one
        for step in self.steps[1:]:
            current_result = step.invoke(current_result) # type: ignore    
        return current_result

def _or_magic(self: Callable[..., Any], other: Callable[..., Any]) -> Chain:
    """The function that will become the __or__ method for our classes."""
    return Chain(self) | other

# This is the list of base classes we want to make "chainable"
# We will patch them directly. This is more reliable than a complex decorator.
from chain.prompts.base import BasePromptTemplate
from chain.chat_models.base import BaseChatModel
from chain.output_parsers.base import BaseOutputParser

BasePromptTemplate.__or__ = _or_magic # type: ignore
BaseChatModel.__or__ = _or_magic # type: ignore
BaseOutputParser.__or__ = _or_magic # type: ignore