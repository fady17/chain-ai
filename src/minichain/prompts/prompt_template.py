# src/minichain/prompts/prompt_template.py
"""
Implementation of a flexible and powerful prompt template using Jinja2.
"""
from typing import Any, List, Optional
from jinja2 import Environment, meta
from .base import BasePromptTemplate

class PromptTemplate(BasePromptTemplate):
    """
    A prompt template that uses the Jinja2 templating engine for formatting.

    This class serves as the standard for creating prompts from strings. It can
    handle simple variable substitutions (e.g., "Hello, {name}!") as well as
    more complex logic like loops and conditionals, thanks to the power of
    the underlying Jinja2 engine.
    """
    
    def __init__(self, template: str, input_variables: Optional[List[str]] = None):
        """
        Initializes the PromptTemplate.

        Args:
            template (str): The template string. Can contain Jinja2 syntax.
            input_variables (Optional[List[str]]): A list of expected variable
                names. If None, variables will be automatically inferred from
                the template string.
        """
        self.template_string = template
        self.jinja_env = Environment()
        self.template = self.jinja_env.from_string(template)
        
        if input_variables is None:
            # Auto-detect variables using Jinja2's Abstract Syntax Tree parser
            ast = self.jinja_env.parse(template)
            input_variables = list(meta.find_undeclared_variables(ast))
        
        super().__init__(input_variables)
    
    def format(self, **kwargs: Any) -> str:
        """Renders the template with the provided variables to produce a final string."""
        self._validate_variables(kwargs)
        return self.template.render(**kwargs)
    
    @classmethod
    def from_template(cls, template: str) -> 'PromptTemplate':
        """A convenience class method to create a PromptTemplate from a string."""
        return cls(template=template)