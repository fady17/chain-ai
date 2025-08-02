# src/minichain/core/types.py
"""
Core data structures for Mini-Chain Framework, now powered by Pydantic.
"""
from typing import Dict, Any, Optional
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
