"""Tests for security fixes - validates all security improvements."""

import os
import pytest


class TestEncryptionKeyValidation:
    """Test CredentialEncryptor requires strong keys."""
    
    def test_rejects_none_key_without_env(self):
        """Must set ENCRYPTION_KEY env var."""
        # Clear env
        env_backup = os.environ.get("ENCRYPTION_KEY")
        if "ENCRYPTION_KEY" in os.environ:
            del os.environ["ENCRYPTION_KEY"]
        
        try:
            from acms.db.encryption import CredentialEncryptor
            with pytest.raises(ValueError, match="ENCRYPTION_KEY environment variable must be set"):
                CredentialEncryptor(key=None)
        finally:
            if env_backup:
                os.environ["ENCRYPTION_KEY"] = env_backup
    
    def test_rejects_short_key(self):
        """Key must be at least 32 characters."""
        from acms.db.encryption import CredentialEncryptor
        with pytest.raises(ValueError, match="at least 32 characters"):
            CredentialEncryptor(key="too-short-key")
    
    def test_rejects_default_key(self):
        """Cannot use default key prefix."""
        from acms.db.encryption import CredentialEncryptor
        default_key = "default-encryption-key-change-in-production"
        assert len(default_key) >= 32  # It's long enough to pass length check
        
        with pytest.raises(ValueError, match="cannot be a default key"):
            CredentialEncryptor(key=default_key)
    
    def test_accepts_strong_key(self):
        """Strong key should work."""
        from acms.db.encryption import CredentialEncryptor
        strong_key = "a" * 32  # 32 chars, not default prefix
        
        # This should NOT raise
        encryptor = CredentialEncryptor(key=strong_key)
        assert encryptor is not None
        
        # Test encrypt/decrypt roundtrip
        original = "my-secret-api-key"
        encrypted = encryptor.encrypt(original)
        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == original


class TestJWTSecretValidation:
    """Test JWT AuthManager requires strong secrets."""
    
    def test_rejects_none_secret_without_env(self):
        """Must set JWT_SECRET env var."""
        env_backup = os.environ.get("JWT_SECRET")
        if "JWT_SECRET" in os.environ:
            del os.environ["JWT_SECRET"]
        
        try:
            from acms.auth import AuthManager
            with pytest.raises(ValueError, match="JWT_SECRET environment variable must be set"):
                AuthManager(secret_key=None)
        finally:
            if env_backup:
                os.environ["JWT_SECRET"] = env_backup
    
    def test_rejects_short_secret(self):
        """Secret must be at least 32 characters."""
        from acms.auth import AuthManager
        with pytest.raises(ValueError, match="at least 32 characters"):
            AuthManager(secret_key="too-short")
    
    def test_rejects_default_secret(self):
        """Cannot use the default 'change-me-in-production' value."""
        from acms.auth import AuthManager
        # Use a string that's long enough to pass length check but contains default prefix
        default_secret = "change-me-in-production" + "x" * 10
        with pytest.raises(ValueError, match="cannot contain"):
            AuthManager(secret_key=default_secret)
    
    def test_accepts_strong_secret(self):
        """Strong secret should work."""
        from acms.auth import AuthManager
        strong_secret = "b" * 32  # 32 chars, not default
        
        # This should NOT raise
        auth = AuthManager(secret_key=strong_secret)
        assert auth is not None
        
        # Test token creation
        token = auth.create_access_token("user123", "test@example.com")
        assert token is not None
        
        # Test token verification
        token_data = auth.verify_token(token)
        assert token_data is not None
        assert token_data.user_id == "user123"


class TestJWTRefreshTokens:
    """Test JWT refresh token functionality."""
    
    @pytest.fixture
    def auth_manager(self):
        """Create auth manager with strong secret."""
        from acms.auth import AuthManager
        return AuthManager(secret_key="test-secret-key-for-jwt-testing-32chars")
    
    def test_creates_refresh_token(self, auth_manager):
        """Should create a refresh token."""
        refresh = auth_manager.create_refresh_token("user123", "test@example.com")
        assert refresh is not None
        
        # Refresh token should NOT work for verify_token
        token_data = auth_manager.verify_token(refresh)
        assert token_data is None  # Refresh tokens are not for API access
    
    def test_refresh_token_exchange(self, auth_manager):
        """Should exchange refresh token for new access token."""
        refresh = auth_manager.create_refresh_token("user123", "test@example.com")
        
        result = auth_manager.refresh_access_token(refresh)
        assert result is not None
        assert "access_token" in result
        assert "refresh_token" in result
        
        # New access token should work
        token_data = auth_manager.verify_token(result["access_token"])
        assert token_data is not None
    
    def test_old_refresh_token_invalid_after_rotation(self, auth_manager):
        """After using refresh token, it should be invalidated (rotation)."""
        refresh = auth_manager.create_refresh_token("user123", "test@example.com")
        
        # First exchange
        result1 = auth_manager.refresh_access_token(refresh)
        assert result1 is not None
        
        # Second exchange with same refresh token should fail
        result2 = auth_manager.refresh_access_token(refresh)
        assert result2 is None  # Token already used


class TestRateLimiterRedis:
    """Test Redis-based rate limiter."""
    
    def test_rate_limiter_is_redis_based(self):
        """API should use RedisRateLimiter, not in-memory."""
        from acms.api.dependencies import RedisRateLimiter, EndpointRateLimiter
        
        # Should be the same class (alias for backward compat)
        assert RedisRateLimiter is EndpointRateLimiter
    
    def test_rate_limiter_class_exists(self):
        """RedisRateLimiter should exist and be importable."""
        from acms.api.dependencies import RedisRateLimiter, get_rate_limiter
        
        limiter = get_rate_limiter()
        assert isinstance(limiter, RedisRateLimiter)
    
    def test_rate_limiter_fallback_without_redis(self):
        """Should work without Redis (fallback to allow)."""
        from acms.api.dependencies import RedisRateLimiter
        
        limiter = RedisRateLimiter(redis_client=None, max_requests=5)
        
        # Should allow when Redis unavailable
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(limiter.is_allowed("test-client"))
            assert result is True
        finally:
            loop.close()


class TestDockerComposeSecrets:
    """Test docker-compose uses env vars, not hardcoded secrets."""
    
    def test_docker_compose_uses_env_vars(self):
        """Verify docker-compose.yml uses ${VAR} syntax."""
        with open("E:/asms/docker/docker-compose.yml", "r") as f:
            content = f.read()
        
        # Should use env vars
        assert "${JWT_SECRET}" in content
        assert "${POSTGRES_PASSWORD}" in content
        assert "${ENCRYPTION_KEY}" in content
        
        # Should NOT have hardcoded values
        assert "change-me-in-production" not in content
        assert 'POSTGRES_PASSWORD: acms' not in content.replace(" ", "")


class TestEnvExampleExists:
    """Test .env.example documents required variables."""
    
    def test_env_example_file_exists(self):
        """Should have .env.example with all required vars."""
        with open("E:/asms/docker/.env.example", "r") as f:
            content = f.read()
        
        assert "JWT_SECRET=" in content
        assert "POSTGRES_PASSWORD=" in content
        assert "ENCRYPTION_KEY=" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
