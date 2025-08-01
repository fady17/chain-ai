"""
Prompt Template Layer for the ChainForge Framework.

This module provides components for creating and formatting prompts
that can be sent to language models. These components act as the first
step in many chains, transforming structured data into a string prompt.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict

from jinja2 import Template

from chainforge.core import Runnable


class BasePromptTemplate(Runnable[Dict[str, Any], str], ABC):
    """
    Abstract base class for all prompt template implementations.

    This class defines the contract for formatting a prompt with a given
    set of input variables. It is a `Runnable` that takes a dictionary of
    values and produces a single formatted string.
    """

    def __init__(self, template: str):
        """
        Args:
            template: The template string that will be formatted.
        """
        self.template = template
        # The `input_variables` attribute is useful for validation and introspection.
        self.input_variables: set[str] = self._extract_variables(template)

    @abstractmethod
    def _extract_variables(self, template: str) -> set[str]:
        """
        Subclasses must implement this to identify input variables in their
        specific template syntax.
        """
        raise NotImplementedError

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """
        Synchronously formats the template with the provided keyword arguments.

        This is the core logic of any prompt template.

        Raises:
            KeyError: If a required variable is not provided in kwargs.
        """
        raise NotImplementedError

    async def ainvoke(self, input: Dict[str, Any], **kwargs) -> str:
        """
        Asynchronously formats the prompt using the input dictionary.

        This method fulfills the `Runnable` contract. As string formatting is a
        fast, synchronous, CPU-bound operation, this is a simple async wrapper
        around the synchronous `format` method.

        Args:
            input: A dictionary where keys correspond to the variables
                   in the prompt template.
            **kwargs: Additional keyword arguments (not used by this component).

        Returns:
            The final, formatted prompt string.
        """
        return self.format(**input)

    async def astream(
        self, input: Dict[str, Any], **kwargs
    ) -> AsyncIterator[str]:
        """
        Yields the single formatted prompt result.

        Prompt formatting is an atomic, non-streaming operation. This method
        fulfills the `Runnable` contract by invoking the model and yielding the
        complete result as a single item in an async iterator.
        """
        result = await self.ainvoke(input, **kwargs)
        yield result


class StringPromptTemplate(BasePromptTemplate):
    """
    A prompt template that uses Python's standard f-string/format syntax.

    This is a lightweight and efficient choice for simple prompts where
    variables are denoted by curly braces, e.g., `{variable}`.
    """

    def _extract_variables(self, template: str) -> set[str]:
        """Extracts variables using a regex for the standard `{variable}` pattern."""
        return set(re.findall(r"{(\w+)}", template))

    def format(self, **kwargs: Any) -> str:
        """
        Formats the template using the built-in `str.format()` method.
        """
        # This check provides a clear error message if the input data is missing
        # keys that are expected by the template string.
        missing_keys = self.input_variables - kwargs.keys()
        if missing_keys:
            raise KeyError(
                f"Missing required variables for prompt template: {missing_keys}"
            )
        return self.template.format(**kwargs)


class JinjaPromptTemplate(BasePromptTemplate):
    """
    A prompt template that uses the powerful Jinja2 templating engine.

    This is ideal for complex prompts that require logic like loops or
    conditionals. Variables are denoted by double curly braces, e.g.,
    `{{ variable }}`.
    """

    def __init__(self, template: str):
        super().__init__(template)
        # We compile the Jinja2 template once at initialization for efficiency.
        # This avoids the cost of parsing the template string on every format call.
        self.jinja_template = Template(template, autoescape=False)

    def _extract_variables(self, template: str) -> set[str]:
        """
        Extracts variables using a simplified regex for the `{{ variable }}` pattern.

        Note: Jinja2's own parser is more robust and can find variables inside
        loops or conditionals. For our purpose of simple validation, this
        regex is a pragmatic and sufficient solution.
        """
        return set(re.findall(r"{{\s*(\w+)\s*}}", template))

    def format(self, **kwargs: Any) -> str:
        """
        Renders the template using the Jinja2 engine's `render()` method.
        """
        missing_keys = self.input_variables - kwargs.keys()
        if missing_keys:
            raise KeyError(
                f"Missing required variables for Jinja2 template: {missing_keys}"
            )
        return self.jinja_template.render(**kwargs)