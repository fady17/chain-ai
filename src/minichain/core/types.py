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