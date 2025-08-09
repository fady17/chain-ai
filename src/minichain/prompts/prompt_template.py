# src/minichain/prompts/prompt_template.py

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
# # src/minichain/prompts/prompt_template.py

# from typing import Any, List, Optional, Dict
# from jinja2 import Environment, meta
# from .base import BasePromptTemplate

# class PromptTemplate(BasePromptTemplate):
#     """A prompt template that uses the Jinja2 templating engine."""
    
#     def __init__(
#         self, 
#         template: str, 
#         input_variables: Optional[List[str]] = None,
#         partial_variables: Optional[Dict[str, Any]] = None
#     ):
#         self.template_string = template
#         self.jinja_env = Environment()
#         self.template = self.jinja_env.from_string(template)
#         self.partial_variables = partial_variables or {}

#         # Determine the final input_variables after accounting for partials
#         if input_variables is None:
#             ast = self.jinja_env.parse(template)
#             all_vars = meta.find_undeclared_variables(ast)
#             # The real inputs are all variables minus the ones we've pre-filled
#             final_input_vars = list(all_vars - set(self.partial_variables.keys()))
#         else:
#             final_input_vars = input_variables
            
#         super().__init__(input_variables=final_input_vars)
    
#     def format(self, **kwargs: Any) -> str:
#         """Renders the template with the provided variables."""
#         # Combine runtime kwargs with the pre-filled partials
#         all_kwargs = {**self.partial_variables, **kwargs}
#         self._validate_variables(all_kwargs)
#         return self.template.render(**all_kwargs)
    
#     # This override is now compatible with the base class
#     def _validate_variables(self, variables: Dict[str, Any]) -> None:
#         # For this class, we need to validate against ALL expected variables
#         # (both input and partial)
#         all_expected_vars = self.input_variables + list(self.partial_variables.keys())
#         missing = set(all_expected_vars) - set(variables.keys())
#         if missing:
#             raise ValueError(f"Missing required input variables for template: {sorted(list(missing))}")

#     @classmethod
#     def from_template(cls, template: str, **kwargs) -> 'PromptTemplate':
#         return cls(template=template, **kwargs)
# # """
# # Implementation of a flexible and powerful prompt template using Jinja2.
# # """
# # from typing import Any, Dict, List, Optional
# # from jinja2 import Environment, meta
# # from .base import BasePromptTemplate

# # class PromptTemplate(BasePromptTemplate):
# #     """
# #     A prompt template that uses the Jinja2 templating engine for formatting.

# #     This class serves as the standard for creating prompts from strings. It can
# #     handle simple variable substitutions (e.g., "{{ name }}") as well as
# #     more complex logic like loops and conditionals.
# #     """
    
# #     def __init__(self, template: str, input_variables: Optional[List[str]] = None, partial_variables: Optional[Dict[str, Any]] = None ):
# #         """
# #         Initializes the PromptTemplate.

# #         Args:
# #             template (str): The template string. Must use Jinja2 syntax,
# #                             e.g., "{{ variable_name }}".
# #             input_variables (Optional[List[str]]): A list of expected variable
# #                 names. If None, variables will be automatically inferred from
# #                 the template string.
# #         """
# #         self.template_string = template
# #         self.jinja_env = Environment()
# #         self.template = self.jinja_env.from_string(template)
# #         self.partial_variables = partial_variables or {} 
        
# #         if input_variables is None:
# #             # Auto-detect variables using Jinja2's Abstract Syntax Tree parser
# #             ast = self.jinja_env.parse(template)
# #             # input_variables = list(meta.find_undeclared_variables(ast))
# #              # Find all variables declared in the template
# #             all_vars = meta.find_undeclared_variables(ast)
# #             # Subtract the partial variables to find the true input variables
# #             input_variables = list(all_vars - set(self.partial_variables.keys()))
        
        
# #         super().__init__(input_variables)
    
# #     # def format(self, **kwargs: Any) -> str:
# #     #     """Renders the template with the provided variables to produce a final string."""
# #     #     self._validate_variables(kwargs)
# #     #     return self.template.render(**kwargs)
# #     def format(self, **kwargs: Any) -> str:
# #         """Renders the template with the provided variables to produce a final string."""
# #         # Combine runtime kwargs with pre-filled partial variables
# #         all_kwargs = {**self.partial_variables, **kwargs}
# #         # We need to validate against the original full set of variables
# #         self._validate_variables(all_kwargs, self.input_variables + list(self.partial_variables.keys()))
# #         return self.template.render(**all_kwargs)
  
# #       # We need to adjust the validation helper slightly to allow passing the variable list
# #     def _validate_variables(self, variables: Dict[str, Any], required_vars: List[str]) -> None: # type: ignore
# #         missing = set(required_vars) - set(variables.keys())
# #         if missing:
# #             raise ValueError(f"Missing required input variables: {sorted(list(missing))}")

# #     @classmethod
# #     def from_template(cls, template: str, **kwargs) -> 'PromptTemplate':
# #         """A convenience class method to create a PromptTemplate from a string."""
# #         return cls(template=template, **kwargs)
# #     # @classmethod
# #     # def from_template(cls, template: str, **kwargs) -> 'PromptTemplate':
# #     #     """A convenience class method to create a PromptTemplate from a string."""
# #     #     return cls(template=template, **kwargs)
