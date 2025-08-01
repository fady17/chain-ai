"""
Language Model Layer for the ChainForge Framework.

This module provides the core abstraction (`BaseLLM`) and concrete
implementations for interfacing with various language models. Each component
is designed to be independently configurable and resilient.
"""
from __future__ import annotations

import logging
from abc import ABC
from typing import Any, AsyncIterator, List, Optional

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from chainforge.config import Settings
from chainforge.core import Runnable

# Standard library logging provides visibility into component operations.
logger = logging.getLogger(__name__)


class BaseLLM(Runnable[str, str], ABC):
    """
    Abstract base class for all Large Language Model implementations.

    This class defines the `Runnable` contract for a component that takes a
    prompt string as input and produces a response string as output. The streaming
    nature of the `Runnable` interface is particularly important for LLMs to
e    enable low-latency, token-by-token responses.
    """

    async def ainvoke(self, input: str, **kwargs: Any) -> str:
        """
        Invokes the LLM with a single prompt and aggregates the full response.

        This default implementation is a convenience wrapper around the `astream`
        method. It consumes the entire async iterator and joins the chunks.

        Args:
            input: The prompt string to send to the LLM.
            **kwargs: Additional keyword arguments for the invocation.

        Returns:
            The complete, aggregated LLM response string.
        """
        # Aggregate the streaming string chunks into a single response string.
        chunks = [chunk async for chunk in self.astream(input, **kwargs)]
        return "".join(chunks)


class ChatOpenAI(BaseLLM):
    """
    A resilient, asynchronous client for OpenAI-compatible chat models.

    This component is architected for maximum flexibility, supporting both
    standard OpenAI and Azure OpenAI endpoints via component-level configuration.
    """
    def __init__(
        self,
        settings: Settings,
        temperature: float = 0.0,
        # Explicit parameters allow for full programmatic control.
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_deployment: Optional[str] = None,
    ):
        """
        Initializes the ChatOpenAI client.
        """
        self.settings = settings
        self.temperature = temperature

        _azure_endpoint = azure_endpoint or self.settings.AZURE_OPENAI_ENDPOINT

        if _azure_endpoint:
            # --- Azure Configuration Path ---
            _api_key = api_key or self.settings.AZURE_OPENAI_API_KEY
            _api_version = api_version or self.settings.OPENAI_API_VERSION
            self.model_name = azure_deployment or self.settings.AZURE_OPENAI_CHAT_DEPLOYMENT

            if not all([_api_key, _azure_endpoint, self.model_name]):
                raise ValueError(
                    "For Azure, 'api_key', 'azure_endpoint', and 'azure_deployment' must be provided."
                )

            self.client = openai.AsyncAzureOpenAI(
                api_key=_api_key, azure_endpoint=_azure_endpoint, api_version=_api_version
            )
            logger.info(f"ChatOpenAI initialized for Azure. Endpoint: {_azure_endpoint}, Deployment: {self.model_name}")
        else:
            # --- Standard OpenAI Configuration Path ---
            _api_key = api_key or self.settings.OPENAI_API_KEY
            self.model_name = "gpt-4o-mini" # A sensible default.

            if not _api_key:
                raise ValueError("For standard OpenAI, 'api_key' must be provided.")

            self.client = openai.AsyncOpenAI(api_key=_api_key)
            logger.info("ChatOpenAI initialized for standard OpenAI.")

    @retry(
        wait=wait_random_exponential(min=Settings.DEFAULT_RETRY_DELAY, max=60),
        stop=stop_after_attempt(Settings.DEFAULT_RETRY_ATTEMPTS),
        reraise=True
    )
    async def _ainvoke_with_retry(self, **kwargs: Any) -> Any:
        """
        A private helper method that wraps the API call with tenacity's retry logic.

        This ensures that transient network errors are handled gracefully without
        disrupting the application flow.
        """
        return await self.client.chat.completions.create(**kwargs)

    async def astream(self, input: str, **kwargs) -> AsyncIterator[str]:
        """
        Streams the response from the configured OpenAI-compatible API.

        This is the core method for achieving low-latency responses, as it yields
        each piece of the response as soon as it is received.

        Args:
            input: The prompt string to send to the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            An async iterator yielding string chunks of the response.
        """
        if not isinstance(input, str):
            raise TypeError("Input for an LLM must be a string.")

        stream = await self._ainvoke_with_retry(
            model=self.model_name,
            messages=[{"role": "user", "content": input}],
            temperature=self.temperature,
            stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            # The API may send empty content chunks; we must filter these out
            # to provide a clean stream of text to the consumer.
            if content is not None:
                yield content