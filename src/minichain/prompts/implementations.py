# # src/minichain/prompts/implementations.py
# """
# Prompt template implementations, now internally powered by the Jinja2 engine
# for enhanced flexibility and robust parsing.
# """
# from typing import Dict, Any, List, Optional
# from jinja2 import Environment, meta

# from .base import BasePromptTemplate

# class PromptTemplate(BasePromptTemplate):
#     """
#     A powerful prompt template that uses the Jinja2 templating engine
#     for formatting. It supports simple substitutions and complex logic
#     like loops and conditionals within the template string.
#     """
    
#     def __init__(self, template: str, input_variables: Optional[List[str]] = None):
#         self.template_string = template
#         self.jinja_env = Environment()
#         self.template = self.jinja_env.from_string(template)
        
#         if input_variables is None:
#             # Auto-detect variables using Jinja2's Abstract Syntax Tree parser
#             ast = self.jinja_env.parse(template)
#             input_variables = list(meta.find_undeclared_variables(ast))
        
#         super().__init__(input_variables)
    
#     def format(self, **kwargs: Any) -> str:
#         """Renders the template with the provided variables using Jinja2."""
#         self._validate_variables(kwargs)
#         return self.template.render(**kwargs)
    
#     @classmethod
#     def from_template(cls, template: str) -> 'PromptTemplate':
#         """Creates a PromptTemplate from a template string."""
#         return cls(template=template)


# class FewShotPromptTemplate(BasePromptTemplate):
#     """
#     Few-shot prompt template that now supports Jinja2 in its components.
#     It constructs a prompt by combining a prefix, formatted examples, and a suffix.
#     """
    
#     def __init__(self, 
#                  examples: List[Dict[str, str]], 
#                  example_prompt: PromptTemplate,
#                  suffix: str,
#                  input_variables: List[str],
#                  prefix: str = "",
#                  example_separator: str = "\n\n"):
        
#         super().__init__(input_variables)
#         self.examples = examples
#         self.example_prompt = example_prompt
#         self.prefix = prefix
#         self.example_separator = example_separator
        
#         # The suffix is also treated as a Jinja2 template
#         self.suffix_template = Environment().from_string(suffix)
    
#     def format(self, **kwargs: Any) -> str:
#         """Formats the prompt with the given variables."""
#         self._validate_variables(kwargs)
        
#         # Format all examples using the now Jinja-powered example_prompt
#         formatted_examples = [self.example_prompt.format(**ex) for ex in self.examples]
        
#         # Combine prefix, examples, and separator
#         example_str = self.example_separator.join(formatted_examples)
#         prompt_parts = [self.prefix, example_str]
        
#         # Render the suffix with the remaining input variables
#         formatted_suffix = self.suffix_template.render(**kwargs)
#         prompt_parts.append(formatted_suffix)
        
#         # Join all non-empty parts with the separator
#         return self.example_separator.join(filter(None, prompt_parts))


# class ChatPromptTemplate(BasePromptTemplate):
#     """
#     Chat prompt template that formats a list of messages for chat models,
#     with each message content rendered by the Jinja2 engine.
#     """
    
#     def __init__(self, messages: List[Dict[str, Any]], input_variables: Optional[List[str]] = None):
#         super().__init__(input_variables=[] if input_variables is None else input_variables)
#         self.messages = messages
#         self.jinja_env = Environment()

#         # Pre-compile Jinja templates for each message content
#         self.message_templates: List[Dict[str, Any]] = [
#             {"role": msg["role"], "template": self.jinja_env.from_string(msg.get("content", ""))}
#             for msg in messages
#         ]
            
#         if input_variables is None:
#             self.input_variables = self._extract_variables_from_messages(messages)
    
#     def _extract_variables_from_messages(self, messages: List[Dict[str, str]]) -> List[str]:
#         """Extracts all unique variables from all message templates."""
#         all_vars = set()
#         for msg in messages:
#             content = msg.get("content", "")
#             ast = self.jinja_env.parse(content)
#             all_vars.update(meta.find_undeclared_variables(ast))
#         return list(all_vars)

#     def format(self, **kwargs: Any) -> List[Dict[str, str]]:
#         """Formats the chat messages with the provided variables."""
#         self._validate_variables(kwargs)
        
#         return [
#             {"role": msg["role"], "content": msg["template"].render(**kwargs)}
#             for msg in self.message_templates
#         ]
# # # mini_langchain/prompts/implementations.py
# # """
# # Prompt template implementations
# # """

# # import re
# # from typing import Dict, Any, List
# # from .base import BasePromptTemplate


# # class PromptTemplate(BasePromptTemplate):
# #     """Simple prompt template with variable substitution using {variable} syntax"""
    
# #     def __init__(self, template: str, input_variables: List[str] = None): # type: ignore
# #         self.template = template
        
# #         # Auto-detect variables if not provided
# #         if input_variables is None:
# #             input_variables = self._extract_variables(template)
        
# #         super().__init__(input_variables)
    
# #     def _extract_variables(self, template: str) -> List[str]:
# #         """Extract variable names from template using regex"""
# #         pattern = r'\{([^}]+)\}'
# #         variables = re.findall(pattern, template)
# #         return list(set(variables))  # Remove duplicates
    
# #     def format(self, **kwargs) -> str:
# #         """Format the template with provided variables"""
# #         self._validate_variables(kwargs)
        
# #         try:
# #             return self.template.format(**kwargs)
# #         except KeyError as e:
# #             raise ValueError(f"Template variable {e} not found in provided arguments")
    
# #     @classmethod
# #     def from_template(cls, template: str) -> 'PromptTemplate':
# #         """Create PromptTemplate from template string (LangChain compatibility)"""
# #         return cls(template)


# # class FewShotPromptTemplate(BasePromptTemplate):
# #     """Few-shot prompt template with examples"""
    
# #     def __init__(self, 
# #                  examples: List[Dict[str, str]], 
# #                  example_prompt: PromptTemplate,
# #                  suffix: str,
# #                  input_variables: List[str],
# #                  example_separator: str = "\n\n"):
        
# #         self.examples = examples
# #         self.example_prompt = example_prompt
# #         self.suffix = suffix
# #         self.example_separator = example_separator
        
# #         super().__init__(input_variables)
    
# #     def format(self, **kwargs) -> str:
# #         """Format the few-shot prompt with examples and input"""
# #         self._validate_variables(kwargs)
        
# #         # Format all examples
# #         formatted_examples = []
# #         for example in self.examples:
# #             formatted_example = self.example_prompt.format(**example)
# #             formatted_examples.append(formatted_example)
        
# #         # Combine examples
# #         examples_text = self.example_separator.join(formatted_examples)
        
# #         # Format suffix with input variables
# #         formatted_suffix = self.suffix.format(**kwargs)
        
# #         # Combine everything
# #         if examples_text:
# #             return f"{examples_text}{self.example_separator}{formatted_suffix}"
# #         else:
# #             return formatted_suffix
    
# #     def add_example(self, example: Dict[str, str]) -> None:
# #         """Add a new example to the template"""
# #         self.examples.append(example)
    
# #     def __str__(self) -> str:
# #         return f"FewShotPromptTemplate(examples={len(self.examples)}, input_variables={self.input_variables})"


# # class ChatPromptTemplate(BasePromptTemplate):
# #     """Chat prompt template that formats messages for chat models"""
    
# #     def __init__(self, messages: List[Dict[str, str]], input_variables: List[str] = None): # type: ignore
# #         self.messages = messages
        
# #         # Auto-detect variables from all messages if not provided
# #         if input_variables is None:
# #             input_variables = self._extract_variables_from_messages(messages)
        
# #         super().__init__(input_variables)
    
# #     def _extract_variables_from_messages(self, messages: List[Dict[str, str]]) -> List[str]:
# #         """Extract variables from all message contents"""
# #         all_variables = set()
# #         pattern = r'\{([^}]+)\}'
        
# #         for message in messages:
# #             content = message.get('content', '')
# #             variables = re.findall(pattern, content)
# #             all_variables.update(variables)
        
# #         return list(all_variables)
    
# #     def format(self, **kwargs) -> List[Dict[str, str]]: # type: ignore
# #         """Format chat messages with variables - returns list of messages"""
# #         self._validate_variables(kwargs)
        
# #         formatted_messages = []
# #         for message in self.messages:
# #             formatted_message = {
# #                 'role': message['role'],
# #                 'content': message['content'].format(**kwargs)
# #             }
# #             formatted_messages.append(formatted_message)
        
# #         return formatted_messages
    
# #     def format_as_string(self, **kwargs) -> str:
# #         """Format as single string for compatibility with string-based models"""
# #         formatted_messages = self.format(**kwargs)
        
# #         formatted_parts = []
# #         for msg in formatted_messages:
# #             role = msg['role'].title()
# #             content = msg['content']
# #             formatted_parts.append(f"{role}: {content}")
        
# #         return "\n".join(formatted_parts)