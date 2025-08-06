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
    handle simple variable substitutions (e.g., "{{ name }}") as well as
    more complex logic like loops and conditionals.
    """
    
    def __init__(self, template: str, input_variables: Optional[List[str]] = None):
        """
        Initializes the PromptTemplate.

        Args:
            template (str): The template string. Must use Jinja2 syntax,
                            e.g., "{{ variable_name }}".
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
    def from_template(cls, template: str, **kwargs) -> 'PromptTemplate':
        """A convenience class method to create a PromptTemplate from a string."""
        return cls(template=template, **kwargs)
# from typing import Any, List, Optional
# import re

# # Attempt to import Jinja2, but don't make it a hard requirement.
# try:
#     from jinja2 import Environment, meta
#     _jinja2_installed = True
# except ImportError:
#     _jinja2_installed = False

# from .base import BasePromptTemplate

# class PromptTemplate(BasePromptTemplate):
#     """
#     A flexible prompt template that supports both simple f-string style
#     and powerful Jinja2 style templating.

#     The templating engine is chosen based on the `template_format` argument.
#     """
    
#     def __init__(
#         self,
#         template: str,
#         template_format: str = "f-string", # Default to simple, dependency-free format
#         input_variables: Optional[List[str]] = None,
#     ):
#         """
#         Initializes the PromptTemplate.

#         Args:
#             template (str): The template string.
#             template_format (str): The format of the template. One of "f-string"
#                                    or "jinja2".
#             input_variables (Optional[List[str]]): A list of expected variable
#                 names. If None, variables will be automatically inferred.
#         """
#         self.template = template
#         self.template_format = template_format

#         if template_format == "f-string":
#             # Infer variables for f-string format, e.g., {variable}
#             if input_variables is None:
#                 input_variables = re.findall(r"\{(\w+)\}", template)
        
#         elif template_format == "jinja2":
#             if not _jinja2_installed:
#                 raise ImportError(
#                     "Jinja2 is not installed. Please run `pip install Jinja2` to use "
#                     "the 'jinja2' template format."
#                 )
#             # Create the Jinja2 template object
#             self.jinja2_template = Environment().from_string(template) # type: ignore
#             # Infer variables for Jinja2 format, e.g., {{ variable }}
#             if input_variables is None:
#                 ast = self.jinja2_template.environment.parse(template)
#                 input_variables = list(meta.find_undeclared_variables(ast)) # type: ignore
#         else:
#             raise ValueError(
#                 f"Invalid template_format: '{template_format}'. "
#                 "Must be one of 'f-string' or 'jinja2'."
#             )
        
#         # Use a set to ensure unique variable names
#         super().__init__(input_variables=list(set(input_variables)))
    
#     def format(self, **kwargs: Any) -> str:
#         """Renders the template with the provided variables."""
#         self._validate_variables(kwargs)
        
#         if self.template_format == "f-string":
#             return self.template.format(**kwargs)
#         elif self.template_format == "jinja2":
#             return self.jinja2_template.render(**kwargs)
#         # This line should be unreachable due to the __init__ check.
#         raise RuntimeError("Invalid template format configured.")
    
#     @classmethod
#     def from_template(cls, template: str, **kwargs) -> 'PromptTemplate':
#         """A convenience class method to create a PromptTemplate from a string."""
#         return cls(template=template, **kwargs)
# # # src/minichain/prompts/prompt_template.py
# # """
# # Implementation of a flexible and powerful prompt template using Jinja2.
# # """
# # from typing import Any, List, Optional
# # from jinja2 import Environment, meta
# # from .base import BasePromptTemplate

# # class PromptTemplate(BasePromptTemplate):
# #     """
# #     A prompt template that uses the Jinja2 templating engine for formatting.

# #     This class serves as the standard for creating prompts from strings. It can
# #     handle simple variable substitutions (e.g., "Hello, {name}!") as well as
# #     more complex logic like loops and conditionals, thanks to the power of
# #     the underlying Jinja2 engine.
# #     """
    
# #     def __init__(self, template: str, input_variables: Optional[List[str]] = None):
# #         """
# #         Initializes the PromptTemplate.

# #         Args:
# #             template (str): The template string. Can contain Jinja2 syntax.
# #             input_variables (Optional[List[str]]): A list of expected variable
# #                 names. If None, variables will be automatically inferred from
# #                 the template string.
# #         """
# #         self.template_string = template
# #         self.jinja_env = Environment()
# #         self.template = self.jinja_env.from_string(template)
        
# #         if input_variables is None:
# #             # Auto-detect variables using Jinja2's Abstract Syntax Tree parser
# #             ast = self.jinja_env.parse(template)
# #             input_variables = list(meta.find_undeclared_variables(ast))
        
# #         super().__init__(input_variables)
    
# #     def format(self, **kwargs: Any) -> str:
# #         """Renders the template with the provided variables to produce a final string."""
# #         self._validate_variables(kwargs)
# #         return self.template.render(**kwargs)
    
# #     @classmethod
# #     def from_template(cls, template: str) -> 'PromptTemplate':
# #         """A convenience class method to create a PromptTemplate from a string."""
# #         return cls(template=template)