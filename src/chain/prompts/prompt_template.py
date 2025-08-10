# src/chain/prompts/prompt_template.py

from typing import Any, List, Optional, Dict
from jinja2 import Environment
from .base import BasePromptTemplate

class PromptTemplate(BasePromptTemplate):
    """
    A prompt template that uses the Jinja2 templating engine and correctly
    handles partial variables.
    """
    
    def __init__(
        self, 
        template: str, 
        input_variables: List[str],
        partial_variables: Optional[Dict[str, Any]] = None
    ):
        # The input_variables are the keys the user provides at runtime.
        super().__init__(input_variables=input_variables)
        self.template_string = template
        self.jinja_env = Environment()
        self.template = self.jinja_env.from_string(template)
        self.partial_variables = partial_variables or {}

    def format(self, **kwargs: Any) -> str:
        """
        Renders the template with the provided and partial variables.
        The `kwargs` here will be the dictionary like `{"query": "..."}`.
        """
        # Combine the pre-defined partials with the runtime kwargs from invoke
        all_kwargs = {**self.partial_variables, **kwargs}
        return self.template.render(**all_kwargs)
    
    @classmethod
    def from_template(cls, template: str, **kwargs) -> 'PromptTemplate':
        return cls(template=template, **kwargs)
