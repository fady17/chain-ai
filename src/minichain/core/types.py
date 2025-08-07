# src/minichain/core/types.py
"""
Core data structures for Mini-Chain Framework, now powered by Pydantic.
"""
from typing import Dict, Any, Optional
import uuid
from pydantic import BaseModel, Field

class Document(BaseModel):
    """Core document structure. Uses Pydantic for validation."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

class BaseMessage(BaseModel):
    """Base class for all Pydantic-based message types."""
    content: str
    
    @property
    def type(self) -> str:
        return self.__class__.__name__

    def __str__(self) -> str:
        return f"{self.type}(content='{self.content}')"

class HumanMessage(BaseMessage):
    """Message from a human user."""
    pass

class AIMessage(BaseMessage):
    """Message from an AI assistant."""
    pass

class SystemMessage(BaseMessage):
    """System instruction message."""
    pass

class ConversationalTurn(BaseModel):
    """
    A Pydantic model representing a single, structured turn in a conversation.
    This explicitly links the user's input to the AI's output and provides
    a unique ID for traceability.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_message: HumanMessage
    ai_message: AIMessage

class TokenUsage(BaseModel):
    """A Pydantic model to represent token usage data from an LLM call."""
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

class ChatResult(BaseModel):
    """
    The structured output of a chat model generation.
    
    This provides not just the text content, but also valuable metadata about
    the generation, such as token usage and the raw provider response.
    """
    content: str = Field(description="The string content of the generated message.")
    model_name: str = Field(description="The name of the model used for the generation.")
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="Token usage statistics.")
    finish_reason: Optional[str] = Field(None, description="The reason the model stopped generating tokens.")
    raw: Optional[Any] = Field(None, description="The raw, original response object from the provider for debugging.")
    
    def __str__(self) -> str:
        """
        When the object is printed, it behaves like a simple string,
        returning only its content.
        """
        return self.content

    def __repr__(self) -> str:
        """Provides a more detailed representation for debugging."""
        return (
            f"ChatResult(content='{self.content[:50]}...', "
            f"model_name='{self.model_name}', "
            f"finish_reason='{self.finish_reason}')"
        )