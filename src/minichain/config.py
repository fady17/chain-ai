"""
Mini LangChain Framework - Core Implementation
Building a lightweight alternative to LangChain with modular components
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests
import json
import os

# ==============================================================================
# PHASE 1: CORE DATA STRUCTURES ✅
# ==============================================================================

@dataclass
class Document:
    """Core document structure that holds content and metadata"""
    page_content: str
    metadata: Dict[str, Any] = None # type: ignore
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def __str__(self) -> str:
        return f"Document(content='{self.page_content[:50]}...', metadata={self.metadata})"


@dataclass
class BaseMessage:
    """Base class for all message types"""
    content: str
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(content='{self.content}')"


@dataclass 
class HumanMessage(BaseMessage):
    """Message from human user"""
    pass


@dataclass
class AIMessage(BaseMessage):
    """Message from AI assistant"""
    pass


@dataclass
class SystemMessage(BaseMessage):
    """System instruction message"""
    pass

