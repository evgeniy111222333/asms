"""Pytest configuration for ACMS tests.

Sets required environment variables for testing to allow old tests
to work with new security requirements.
"""

import os
import pytest

# Set required env vars for testing before any imports
os.environ.setdefault("JWT_SECRET", "test-jwt-secret-for-unit-tests-32chars")
os.environ.setdefault("ENCRYPTION_KEY", "test-encryption-key-for-unit-tests32chars")
os.environ.setdefault("POSTGRES_PASSWORD", "test-password")


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Ensure test environment variables are set."""
    # Already set above via setdefault
    yield
