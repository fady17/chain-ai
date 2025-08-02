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
# # mini_langchain/core/types.py
# """
# Core data structures for Mini LangChain Framework
# """

# from typing import Dict, Any
# from dataclasses import dataclass


# @dataclass
# class Document:
#     """Core document structure that holds content and metadata"""
#     page_content: str
#     metadata: Dict[str, Any] = None # type: ignore
    
#     def __post_init__(self):
#         if self.metadata is None:
#             self.metadata = {}
    
#     def __str__(self) -> str:
#         return f"Document(content='{self.page_content[:50]}...', metadata={self.metadata})"


# @dataclass
# class BaseMessage:
#     """Base class for all message types"""
#     content: str
    
#     def __str__(self) -> str:
#         return f"{self.__class__.__name__}(content='{self.content}')"


# @dataclass 
# class HumanMessage(BaseMessage):
#     """Message from human user"""
#     pass


# @dataclass
# class AIMessage(BaseMessage):
#     """Message from AI assistant"""
#     pass


# @dataclass
# class SystemMessage(BaseMessage):
#     """System instruction message"""
#     pass