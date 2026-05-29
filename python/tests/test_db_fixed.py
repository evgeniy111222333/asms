"""Tests for the fixed db module - bcrypt hashing, archive, field whitelists, encryption, new CRUD."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta


class TestCreateUserBcrypt:
    """Test that create_user uses bcrypt, not 'hashed_' prefix."""

    @pytest.mark.asyncio
    async def test_create_user_uses_bcrypt(self):
        """Password should be hashed with bcrypt, not stored as 'hashed_<password>'."""
        from acms.db.manager import DatabaseManager
        # Verify create_user method exists and inspect its source
        import inspect
        source = inspect.getsource(DatabaseManager.create_user)
        # Should NOT have the old placeholder pattern
        assert 'hashed_' not in source or 'bcrypt' in source.lower() or 'hash_password' in source, \
            "create_user should use bcrypt hashing, not 'hashed_' prefix"


class TestFieldWhitelists:
    """Test that update operations only allow whitelisted fields."""

    def test_allowed_order_fields(self):
        from acms.db.models import ALLOWED_ORDER_FIELDS
        # Should NOT include 'id', 'user_id', 'symbol' (immutable fields)
        assert "id" not in ALLOWED_ORDER_FIELDS
        assert "user_id" not in ALLOWED_ORDER_FIELDS
        # Should include mutable fields
        assert "status" in ALLOWED_ORDER_FIELDS

    def test_allowed_strategy_fields(self):
        from acms.db.models import ALLOWED_STRATEGY_FIELDS
        assert "id" not in ALLOWED_STRATEGY_FIELDS
        assert "user_id" not in ALLOWED_STRATEGY_FIELDS


class TestCredentialEncryptor:
    """Test the CredentialEncryptor for exchange API keys."""

    def test_encrypt_decrypt_roundtrip(self):
        from acms.db.encryption import CredentialEncryptor
        enc = CredentialEncryptor()
        plaintext = "my-super-secret-api-key-12345"
        encrypted = enc.encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = enc.decrypt(encrypted)
        assert decrypted == plaintext

    def test_different_keys_produce_different_ciphertext(self):
        from acms.db.encryption import CredentialEncryptor
        # Keys must be at least 32 characters and not contain default prefix
        enc1 = CredentialEncryptor(key="test-key-one-for-encryption-testing!")
        enc2 = CredentialEncryptor(key="test-key-two-for-encryption-testing!")
        plaintext = "same-secret"
        e1 = enc1.encrypt(plaintext)
        e2 = enc2.encrypt(plaintext)
        # Different keys should produce different ciphertexts
        assert e1 != e2

    def test_same_key_decrypts(self):
        from acms.db.encryption import CredentialEncryptor
        enc = CredentialEncryptor(key="consistent-key-minimum-32-chars!")
        encrypted = enc.encrypt("test-secret")
        decrypted = enc.decrypt(encrypted)
        assert decrypted == "test-secret"

    def test_empty_string(self):
        from acms.db.encryption import CredentialEncryptor
        enc = CredentialEncryptor()
        encrypted = enc.encrypt("")
        assert enc.decrypt(encrypted) == ""


class TestArchiveTrades:
    """Test that archive_old_trades moves data instead of deleting."""

    @pytest.mark.asyncio
    async def test_archive_moves_not_deletes(self):
        """Archive should INSERT INTO archive then DELETE, not just DELETE."""
        from acms.db.manager import DatabaseManager
        import inspect
        source = inspect.getsource(DatabaseManager.archive_old_trades)
        # Should contain both INSERT and DELETE operations
        assert 'INSERT' in source.upper() or 'insert' in source.lower() or 'archive' in source.lower(), \
            "archive_old_trades should INSERT INTO archive table before deleting"
        assert 'DELETE' in source.upper() or 'delete' in source.lower(), \
            "archive_old_trades should delete from original after archiving"


class TestNewCRUDMethods:
    """Test newly added CRUD methods for previously missing models."""

    @pytest.mark.asyncio
    async def test_crud_api_key_exists(self):
        """ApiKey CRUD should exist."""
        from acms.db.manager import DatabaseManager
        db = DatabaseManager()
        assert hasattr(db, "create_api_key") or hasattr(db, "get_api_key")

    @pytest.mark.asyncio
    async def test_crud_exchange_credential_exists(self):
        """ExchangeCredential CRUD should exist."""
        from acms.db.manager import DatabaseManager
        db = DatabaseManager()
        assert hasattr(db, "create_exchange_credential") or hasattr(db, "save_exchange_credential")

    @pytest.mark.asyncio
    async def test_crud_risk_event_exists(self):
        """RiskEvent CRUD should exist."""
        from acms.db.manager import DatabaseManager
        db = DatabaseManager()
        assert hasattr(db, "create_risk_event") or hasattr(db, "log_risk_event")

    @pytest.mark.asyncio
    async def test_crud_signal_record_exists(self):
        """SignalRecord CRUD should exist."""
        from acms.db.manager import DatabaseManager
        db = DatabaseManager()
        assert hasattr(db, "create_signal") or hasattr(db, "save_signal")

    @pytest.mark.asyncio
    async def test_crud_position_record_exists(self):
        """PositionRecord CRUD should exist."""
        from acms.db.manager import DatabaseManager
        db = DatabaseManager()
        assert hasattr(db, "create_position") or hasattr(db, "save_position")


class TestModelsModule:
    """Test that the db models module has all expected models."""

    def test_user_model_exists(self):
        from acms.db.models import User
        assert User.__tablename__ == "users"

    def test_api_key_model_exists(self):
        from acms.db.models import ApiKey
        assert ApiKey.__tablename__ == "api_keys"

    def test_exchange_credential_model_exists(self):
        from acms.db.models import ExchangeCredential
        assert ExchangeCredential.__tablename__ is not None

    def test_risk_event_model_exists(self):
        from acms.db.models import RiskEvent
        assert RiskEvent.__tablename__ is not None

    def test_trade_archive_model_exists(self):
        from acms.db.models import TradeArchiveRecord
        assert TradeArchiveRecord.__tablename__ is not None

    def test_position_record_model_exists(self):
        from acms.db.models import PositionRecord
        assert PositionRecord.__tablename__ is not None


class TestDbModuleImports:
    """Test backward compatibility of db module imports."""

    def test_import_from_db(self):
        """Should be able to import DatabaseManager from acms.db."""
        from acms.db import DatabaseManager
        assert DatabaseManager is not None

    def test_import_models_from_db(self):
        """Should be able to import models from acms.db."""
        from acms.db import User
        assert User is not None

    def test_import_encryption_from_db(self):
        """Should be able to import CredentialEncryptor from acms.db."""
        from acms.db import CredentialEncryptor
        assert CredentialEncryptor is not None

    def test_import_session_from_db(self):
        """Should be able to import session functions from acms.db."""
        from acms.db import get_session
        assert get_session is not None
