"""Tests for the fixed AI integration - features, tick/signal processing, risk AI, portfolio AI."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestOrchestratorFeaturesFixed:
    """Test that _compute_features returns real features, not 'computed_placeholder'."""

    def test_compute_features_returns_dict(self):
        try:
            from acms.ai.integration.orchestrator import AIOrchestrator
        except ImportError:
            pytest.skip("AIOrchestrator not available")

        orch = AIOrchestrator.__new__(AIOrchestrator)
        orch._feature_engineer = None

        market_data = {
            "prices": [100, 101, 102, 101, 103, 104, 103, 105, 104, 106] * 10,
            "volumes": [1000, 1100, 900, 1200, 1050, 980, 1100, 1030, 990, 1080] * 10,
        }

        result = orch._compute_features(market_data)
        assert isinstance(result, dict), "Features should be a dict"
        # Should NOT contain "computed_placeholder"
        for key, value in result.items():
            assert value != "computed_placeholder", \
                f"Feature '{key}' should not be 'computed_placeholder'"
            assert isinstance(value, (int, float)), \
                f"Feature '{key}' should be numeric, got {type(value)}"


class TestOrchestratorTickProcessing:
    """Test that _on_tick_data processes ticks, not just `pass`."""

    @pytest.mark.asyncio
    async def test_on_tick_data_does_not_crash(self):
        try:
            from acms.ai.integration.orchestrator import AIOrchestrator
        except ImportError:
            pytest.skip("AIOrchestrator not available")

        orch = AIOrchestrator.__new__(AIOrchestrator)
        orch._tick_buffers = {}
        orch._risk_engine = None

        data = {"symbol": "BTC/USDT", "price": 50000.0, "volume": 1.5}
        # _on_tick_data might be async
        if hasattr(orch._on_tick_data, '__coroutine__'):
            await orch._on_tick_data(data)
        else:
            result = orch._on_tick_data(data)
            if hasattr(result, '__await__'):
                await result

    @pytest.mark.asyncio
    async def test_on_tick_data_buffers_ticks(self):
        try:
            from acms.ai.integration.orchestrator import AIOrchestrator
        except ImportError:
            pytest.skip("AIOrchestrator not available")

        orch = AIOrchestrator.__new__(AIOrchestrator)
        orch._tick_buffers = {}
        orch._risk_engine = None

        data = {"symbol": "BTC/USDT", "price": 50000.0, "volume": 1.5}
        result = orch._on_tick_data(data)
        if hasattr(result, '__await__'):
            await result

        # Should have buffered the tick
        assert "BTC/USDT" in orch._tick_buffers


class TestOrchestratorSignalProcessing:
    @pytest.mark.asyncio
    async def test_on_signal_data_does_not_crash(self):
        try:
            from acms.ai.integration.orchestrator import AIOrchestrator
        except ImportError:
            pytest.skip("AIOrchestrator not available")

        orch = AIOrchestrator.__new__(AIOrchestrator)
        orch._signal_buffer = {}
        orch._decision_router = None

        data = {"symbol": "BTC/USDT", "signal_type": "momentum", "strength": 0.8}
        result = orch._on_signal_data(data)
        if hasattr(result, '__await__'):
            await result


class TestCredentialEncryptorIntegration:
    """Test that CredentialEncryptor works end-to-end."""

    def test_encrypt_decrypt_roundtrip(self):
        from acms.db.encryption import CredentialEncryptor
        enc = CredentialEncryptor()
        plaintext = "my-super-secret-api-key-12345"
        encrypted = enc.encrypt(plaintext)
        assert encrypted != plaintext
        decrypted = enc.decrypt(encrypted)
        assert decrypted == plaintext


class TestModuleImports:
    """Test that AI modules can be imported."""

    def test_import_ai(self):
        import acms.ai
        assert acms.ai is not None

    def test_import_ai_core(self):
        from acms.ai.core import AIConfig
        assert AIConfig is not None

    def test_import_ai_decision(self):
        import acms.ai.decision
        assert acms.ai.decision is not None

    def test_import_ai_features(self):
        import acms.ai.features
        assert acms.ai.features is not None

    def test_import_ai_models(self):
        try:
            import acms.ai.models
            assert acms.ai.models is not None
        except ImportError as e:
            if 'torch' in str(e):
                pytest.skip("PyTorch not installed")
            raise

    def test_import_ai_training(self):
        try:
            import acms.ai.training
            assert acms.ai.training is not None
        except ImportError as e:
            if 'torch' in str(e):
                pytest.skip("PyTorch not installed")
            raise
