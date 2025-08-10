# tests/conftest.py - Pytest configuration
"""
Pytest configuration for chain tests.
"""
import pytest
import asyncio
import os
from unittest.mock import Mock

# Configure asyncio mode for all tests
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_azure_credentials():
    """Provide mock Azure credentials for testing."""
    return {
        "api_key": "test_azure_key_12345",
        "region": "test_region"
    }


@pytest.fixture
def mock_openai_credentials():
    """Provide mock OpenAI credentials for testing."""
    return {
        "api_key": "test_openai_key_12345"
    }


# Skip integration tests if credentials are not available
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: Integration tests requiring real API credentials"
    )
    config.addinivalue_line(
        "markers",
        "slow: Slow running tests that may take several seconds"
    )
    config.addinivalue_line(
        "markers",
        "unit: Fast unit tests with mocked dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip integration tests if credentials are missing.
    """
    skip_integration = pytest.mark.skip(
        reason="Integration test requires AZURE_SPEECH_KEY environment variable"
    )
    
    for item in items:
        if "integration" in item.keywords:
            if not os.getenv('AZURE_SPEECH_KEY'):
                item.add_marker(skip_integration)

