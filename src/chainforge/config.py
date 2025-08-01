"""
Centralized Configuration Management for the ChainForge Framework.

This module uses pydantic-settings to load and validate all external
configurations from environment variables. It acts as a passive container,
providing a single, type-safe source of truth for default settings that
components can use as fallbacks.
"""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Defines the application's default configuration settings, loaded from the environment.

    The primary purpose of this class is to make environment variables accessible
    in a structured, type-safe way. Components should use these values as a
    fallback source of configuration if no explicit parameters are provided
    during their instantiation. This class itself makes no decisions about which
    credentials to use; that responsibility lies with the individual components.
    """
    # --- Azure OpenAI Credentials ---
    # These are the credentials for connecting to an Azure-hosted OpenAI service.
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    OPENAI_API_VERSION: str = "2024-02-01"

    # --- Azure OpenAI Deployment Names ---
    # In Azure, models are accessed via "deployments," which are named instances.
    AZURE_OPENAI_CHAT_DEPLOYMENT: Optional[str] = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: Optional[str] = None

    # --- Standard OpenAI Credentials ---
    OPENAI_API_KEY: Optional[str] = None

    # --- Google Credentials ---
    # Included for future components that may use Google's AI services.
    GOOGLE_API_KEY: Optional[str] = None

    # --- Resilience Defaults ---
    # These provide a sensible default retry behavior for all components.
    DEFAULT_RETRY_ATTEMPTS: int = 5
    DEFAULT_RETRY_DELAY: float = 1.0

    # This model_config tells Pydantic to look for a .env file, which is
    # extremely useful for local development without polluting the global environment.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

# This singleton instance ensures that environment variables are read and parsed
# only once upon application startup, providing an efficient and consistent
# source of configuration defaults.
settings = Settings()