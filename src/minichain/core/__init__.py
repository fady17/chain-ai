# src/minichain/core/__init__.py
"""
The `core` module contains the fundamental data structures and interfaces
that power the Mini-Chain framework.

These Pydantic-based models ensure that data passed between different
components of the library is structured, validated, and type-safe.
"""
from .types import (
    Document,
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ConversationalTurn,
)

__all__ = [
    "Document",
    "BaseMessage",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "ConversationalTurn",
]