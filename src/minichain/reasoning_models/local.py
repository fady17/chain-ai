# src/minichain/reasoning_models/local.py
"""
Provides an interface for local, OpenAI-compatible reasoning models.
"""

from openai import OpenAI

from .base import LocalReasoningConfig
from .openai_like import OpenAILikeReasoningModel

class LocalReasoningModel(OpenAILikeReasoningModel):
    """A reasoning model running on a local, OpenAI-compatible server."""
    config: LocalReasoningConfig # type: ignore

    def __init__(self, config: LocalReasoningConfig):
        super().__init__(config=config)
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )