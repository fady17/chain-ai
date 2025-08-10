# src/chain/__init__.py
"""
Mini-Chain: A transparent, modular micro-framework for building with LLMs.

This file marks the 'chain' directory as a Python package.
To use the components, please import them from their respective sub-modules.

For example:
    from chain.chat_models import LocalChatModel
    from chain.memory import FAISSVectorStore
    from chain.voice import run_stt
"""
from . import chains 
from pydantic import BaseModel, Field
__all__ = [
    "BaseModel",
    "Field",
]
