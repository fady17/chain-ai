# # src/minichain/prompts/jinja.py
# """
# A prompt template implementation using the powerful Jinja2 templating engine.
# """
# from typing import Any, Dict
# from jinja2 import Environment, Template

# class JinjaPromptTemplate:
#     """
#     A prompt template that uses Jinja2 for formatting. Allows for complex logic
#     like loops and conditionals within the prompt string itself.
#     """
#     def __init__(self, template: str):
#         self.template_string = template
#         self.jinja_env = Environment()
#         self.template: Template = self.jinja_env.from_string(template)
        
#     def format(self, **kwargs: Any) -> str:
#         """
#         Renders the template with the provided variables.
#         """
#         return self.template.render(**kwargs)

#     @classmethod
#     def from_template(cls, template: str) -> "JinjaPromptTemplate":
#         """Creates a JinjaPromptTemplate from a template string."""
#         return cls(template)