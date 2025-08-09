# src/minichain/prompts/few_shot.py
"""
Implementation of a few-shot prompt template for in-context learning.
"""
from typing import Dict, Any, List
from jinja2 import Environment
from .base import BasePromptTemplate
from .prompt_template import PromptTemplate

class FewShotPromptTemplate(BasePromptTemplate):
    """
    A prompt template for creating "few-shot" prompts.

    This template dynamically constructs a prompt that includes several
    examples of a task, helping language models understand complex
    instructions or specific output formats by learning "in-context".
    """
    
    def __init__(self, 
                 examples: List[Dict[str, str]], 
                 example_prompt: PromptTemplate,
                 suffix: str,
                 input_variables: List[str],
                 prefix: str = "",
                 example_separator: str = "\n\n"):
        """
        Initializes the FewShotPromptTemplate.
        """
        super().__init__(input_variables)
        self.examples = examples
        self.example_prompt = example_prompt
        self.prefix = prefix
        self.example_separator = example_separator
        
        # Suffix is also a Jinja2 template
        self.suffix_template = Environment().from_string(suffix)
        # All variables used in the suffix must be in the input_variables
        self.suffix_input_variables = self._get_template_variables(suffix)

    def _get_template_variables(self, template_string: str) -> List[str]:
        """Helper to extract variables from a Jinja2 template string."""
        env = Environment()
        ast = env.parse(template_string)
        from jinja2 import meta
        return list(meta.find_undeclared_variables(ast))
    
    def format(self, **kwargs: Any) -> str:
        """Constructs the final few-shot prompt string."""
   
        # The main input variables are those used in the final suffix.
        # Everything else is assumed to be part of the examples.
        main_input = {k: v for k, v in kwargs.items() if k in self.suffix_input_variables}
        self._validate_variables(main_input)

   
        # Iterate through the examples and apply the .format() method of the
        # example_prompt to each one.
        formatted_examples = [
            self.example_prompt.format(**example) for example in self.examples
        ]
        
        # Join the formatted examples with the separator
        example_str = self.example_separator.join(formatted_examples)
        
        # Assemble the final prompt parts
        prompt_parts = [self.prefix, example_str]
        
        # --- render the suffix with the main input ---
        formatted_suffix = self.suffix_template.render(**main_input)
        prompt_parts.append(formatted_suffix)
        
        # Join all non-empty parts of the prompt
        return self.example_separator.join(filter(None, prompt_parts))
    def invoke(self, variables: Dict[str, Any]) -> 'StringPromptResult': # type: ignore
        formatted = self.format(**variables)
        return StringPromptResult(formatted)

class StringPromptResult:
    def __init__(self, value: str):
        self.value = value

    def to_string(self) -> str:
        return self.value

    def __str__(self):
        return self.value