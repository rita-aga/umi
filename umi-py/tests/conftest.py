"""
Pytest configuration for Umi tests.
"""

import pytest


# Markers for integration tests (require API keys/databases)
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys/databases)"
    )
    config.addinivalue_line(
        "markers", "anthropic: marks tests that require Anthropic API key"
    )
    config.addinivalue_line(
        "markers", "openai: marks tests that require OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "lance: marks tests that require LanceDB"
    )
    config.addinivalue_line(
        "markers", "postgres: marks tests that require Postgres database"
    )
