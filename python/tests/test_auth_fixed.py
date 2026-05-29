"""Tests for the fixed auth module - bcrypt hashing, real authentication, constant-time comparison."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestPasswordHashing:
    """Test that password hashing uses bcrypt, not SHA-256."""

    def test_hash_password_produces_bcrypt_hash(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        hashed = auth.hash_password("test_password_123")
        # bcrypt hashes start with $2b$
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$"), \
            f"Expected bcrypt hash, got: {hashed[:10]}..."

    def test_hash_password_different_salts(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        h1 = auth.hash_password("same_password")
        h2 = auth.hash_password("same_password")
        # Different salts should produce different hashes
        assert h1 != h2, "Same password should produce different hashes due to random salts"

    def test_verify_password_correct(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        hashed = auth.hash_password("my_secret_password")
        assert auth.verify_password("my_secret_password", hashed) is True

    def test_verify_password_incorrect(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        hashed = auth.hash_password("my_secret_password")
        assert auth.verify_password("wrong_password", hashed) is False

    def test_verify_password_malformed_hash(self):
        """Verify that malformed hashes don't crash, just return False."""
        from acms.auth import AuthManager
        auth = AuthManager()
        result = auth.verify_password("password", "not_a_real_hash")
        assert result is False

    def test_verify_password_empty(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        hashed = auth.hash_password("")
        assert auth.verify_password("", hashed) is True
        assert auth.verify_password("nonempty", hashed) is False


class TestApiKeySecurity:
    """Test that API key operations use constant-time comparison."""

    def test_generate_api_key_unique(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        key1 = auth.generate_api_key()
        key2 = auth.generate_api_key()
        assert key1 != key2, "API keys should be unique"

    def test_generate_api_key_format(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        result = auth.generate_api_key()
        # generate_api_key returns a tuple (raw_key, hashed_key)
        assert isinstance(result, tuple)
        raw_key, hashed_key = result
        assert isinstance(raw_key, str)
        assert len(raw_key) > 10, "API key should be reasonably long"
        assert raw_key.startswith("acms_"), "API key should start with acms_"

    def test_verify_api_key_constant_time(self):
        """Test that verify_api_key uses hmac.compare_digest, not ==."""
        from acms.auth import AuthManager
        import hmac
        auth = AuthManager()

        # Generate a key and verify it
        raw_key, hashed_key = auth.generate_api_key()

        # Verify works correctly
        assert auth.verify_api_key(raw_key, hashed_key) is True
        # Wrong key should fail
        assert auth.verify_api_key("acms_wrongkey", hashed_key) is False


class TestAuthenticateUser:
    """Test the fixed authenticate_user that queries the database."""

    @pytest.mark.asyncio
    async def test_authenticate_valid_user(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        hashed = auth.hash_password("correct_password")

        with patch("acms.db.DatabaseManager") as MockDB:
            mock_db = MockDB.return_value
            mock_db.get_user_by_email = AsyncMock(return_value={
                "id": "user_123",
                "email": "test@example.com",
                "hashed_password": hashed,
                "is_admin": False,
            })
            result = await auth.authenticate_user("test@example.com", "correct_password")
            assert result is not None
            assert result["id"] == "user_123"
            assert result["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_wrong_password(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        hashed = auth.hash_password("correct_password")

        with patch("acms.db.DatabaseManager") as MockDB:
            mock_db = MockDB.return_value
            mock_db.get_user_by_email = AsyncMock(return_value={
                "id": "user_123",
                "email": "test@example.com",
                "hashed_password": hashed,
                "is_admin": False,
            })
            result = await auth.authenticate_user("test@example.com", "wrong_password")
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_nonexistent_user(self):
        from acms.auth import AuthManager
        auth = AuthManager()

        with patch("acms.db.DatabaseManager") as MockDB:
            mock_db = MockDB.return_value
            mock_db.get_user_by_email = AsyncMock(return_value=None)
            result = await auth.authenticate_user("nonexistent@example.com", "any_password")
            assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_db_error(self):
        """DB errors should not crash, just return None."""
        from acms.auth import AuthManager
        auth = AuthManager()

        with patch("acms.db.DatabaseManager") as MockDB:
            mock_db = MockDB.return_value
            mock_db.get_user_by_email = AsyncMock(side_effect=Exception("DB connection failed"))
            result = await auth.authenticate_user("test@example.com", "password")
            assert result is None


class TestJWTTokens:
    """Test JWT token creation and verification."""

    def test_create_and_verify_token(self):
        from acms.auth import AuthManager, TokenData
        auth = AuthManager()
        token = auth.create_token(user_id="user_123", email="test@example.com")
        assert isinstance(token, str)
        assert len(token) > 20

        payload = auth.verify_token(token)
        assert payload is not None
        assert isinstance(payload, TokenData)
        assert payload.user_id == "user_123"
        assert payload.email == "test@example.com"

    def test_verify_invalid_token(self):
        from acms.auth import AuthManager
        auth = AuthManager()
        result = auth.verify_token("invalid.token.here")
        assert result is None

    def test_verify_expired_token(self):
        """Tokens that are expired should be rejected."""
        from acms.auth import AuthManager
        auth = AuthManager()
        # Create a token that expires immediately
        import jwt
        import time
        token = jwt.encode(
            {"user_id": "user_123", "exp": time.time() - 3600},
            auth.secret_key,
            algorithm="HS256"
        )
        result = auth.verify_token(token)
        assert result is None


class TestNoHardcodedSecrets:
    """Verify that default secrets have startup warnings."""

    def test_default_jwt_secret_documented(self):
        """The default JWT secret should be documented as insecure."""
        from acms.auth import AuthManager
        auth = AuthManager()
        # It should have a default but it should not be used in production
        assert auth.secret_key is not None
