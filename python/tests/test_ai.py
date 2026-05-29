"""Comprehensive tests for ACMS AI Module.

Tests all major components:
- Core: config, GPU manager, base models, types, tensor utils
- Models: TFT, RL agents, GNN, NLP, Meta-learning, Self-supervised, Ensemble
- Training: trainer, online learning, curriculum, hyperopt, walkforward
- Inference: pipeline, server, A/B testing
- Features: engineering, store, drift detection
- Decision: portfolio AI, risk AI, strategy selector, explainer
- Knowledge: memory, graph, adaptation
- Integration: orchestrator, redis cache, kafka consumer, api routes
- Monitoring: GPU monitor, metrics, model monitor
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============================================================================
# Test Configuration
# ============================================================================

# Set environment variables for testing
os.environ.setdefault("ACMS_DATA_DIR", tempfile.gettempdir())
os.environ.setdefault("ACMS_MODEL_DIR", tempfile.gettempdir())
os.environ.setdefault("ACMS_LOG_LEVEL", "DEBUG")

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    n_targets = 3

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)
    timestamps = [
        datetime.utcnow() - timedelta(minutes=i)
        for i in range(n_samples)
    ]

    return {
        "X": X,
        "y": y,
        "timestamps": timestamps,
        "feature_names": [f"feature_{i}" for i in range(n_features)],
        "target_names": [f"target_{i}" for i in range(n_targets)],
    }


@pytest.fixture
def sample_signals():
    """Generate sample trading signals."""
    return [
        {
            "symbol": "BTC/USDT",
            "direction": "long",
            "strength": 0.8,
            "indicators": {"rsi": 30, "macd": 1.5, "bb": 0.2},
            "timestamp": datetime.utcnow(),
            "metadata": {"source": "momentum"},
        },
        {
            "symbol": "ETH/USDT",
            "direction": "short",
            "strength": 0.6,
            "indicators": {"rsi": 70, "macd": -1.0, "bb": 0.8},
            "timestamp": datetime.utcnow(),
            "metadata": {"source": "mean_reversion"},
        },
    ]


@pytest.fixture
def sample_portfolio_state():
    """Generate sample portfolio state."""
    return {
        "total_value": 100000.0,
        "positions": [
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "quantity": 1.5,
                "entry_price": 45000.0,
                "mark_price": 47000.0,
            },
            {
                "symbol": "ETH/USDT",
                "side": "sell",
                "quantity": 10.0,
                "entry_price": 3000.0,
                "mark_price": 2900.0,
            },
        ],
        "cash": 50000.0,
        "margin_used": 10000.0,
    }


# ============================================================================
# Test: Core Types
# ============================================================================


class TestCoreTypes:
    """Test core type definitions."""

    def test_market_regime_enum(self):
        """Test MarketRegime enum values."""
        from acms.ai.core.types import MarketRegime

        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.BEAR.value == "bear"
        assert MarketRegime.HIGH_VOL.value == "high_vol"
        assert MarketRegime.LOW_VOL.value == "low_vol"
        assert MarketRegime.TRENDING.value == "trending"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.UNKNOWN.value == "unknown"

    def test_signal_strength_enum(self):
        """Test SignalStrength enum values."""
        from acms.ai.core.types import SignalStrength

        assert SignalStrength.STRONG_BUY.value == "strong_buy"
        assert SignalStrength.BUY.value == "buy"
        assert SignalStrength.NEUTRAL.value == "neutral"
        assert SignalStrength.SELL.value == "sell"
        assert SignalStrength.STRONG_SELL.value == "strong_sell"

    def test_risk_level_enum(self):
        """Test RiskLevel enum values."""
        from acms.ai.core.types import RiskLevel

        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_model_input_dataclass(self):
        """Test ModelInput dataclass."""
        from acms.ai.core.types import ModelInput, MarketTensor

        # Create sample input
        features = np.random.randn(10, 20).astype(np.float32)
        input_data = ModelInput(
            features=features,
            timestamps=[datetime.utcnow() for _ in range(10)],
            metadata={"source": "test"},
        )

        assert isinstance(input_data.features, np.ndarray)
        assert input_data.features.shape == (10, 20)
        assert len(input_data.timestamps) == 10
        assert input_data.metadata["source"] == "test"

    def test_model_output_dataclass(self):
        """Test ModelOutput dataclass."""
        from acms.ai.core.types import ModelOutput

        predictions = np.random.randn(10, 3).astype(np.float32)
        output = ModelOutput(
            predictions=predictions,
            confidence=np.random.rand(10).astype(np.float32),
            metadata={"model_version": "1.0"},
        )

        assert output.predictions.shape == (10, 3)
        assert output.confidence.shape == (10,)
        assert "model_version" in output.metadata

    def test_regime_prediction_dataclass(self):
        """Test RegimePrediction dataclass."""
        from acms.ai.core.types import RegimePrediction, MarketRegime

        prediction = RegimePrediction(
            regime=MarketRegime.TRENDING,
            confidence=0.85,
            probabilities={
                "bull": 0.1,
                "bear": 0.05,
                "high_vol": 0.05,
                "low_vol": 0.05,
                "trending": 0.7,
                "ranging": 0.05,
            },
        )

        assert prediction.regime == MarketRegime.TRENDING
        assert prediction.confidence == 0.85
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)

    def test_risk_assessment_dataclass(self):
        """Test RiskAssessment dataclass."""
        from acms.ai.core.types import RiskAssessment, RiskLevel

        assessment = RiskAssessment(
            level=RiskLevel.MEDIUM,
            score=0.6,
            var_95=0.02,
            cvar_95=0.035,
            max_drawdown=0.08,
            recommendations=[
                "Reduce position size by 20%",
                "Add stop-loss orders",
            ],
        )

        assert assessment.level == RiskLevel.MEDIUM
        assert assessment.score == 0.6
        assert assessment.var_95 < assessment.cvar_95
        assert len(assessment.recommendations) == 2

    def test_prediction_with_uncertainty(self):
        """Test PredictionWithUncertainty dataclass."""
        from acms.ai.core.types import PredictionWithUncertainty

        pred = PredictionWithUncertainty(
            mean=np.array([1.0, 2.0, 3.0]),
            std=np.array([0.1, 0.15, 0.2]),
            quantiles={
                "q05": np.array([0.9, 1.8, 2.7]),
                "q95": np.array([1.1, 2.2, 3.3]),
            },
        )

        assert len(pred.mean) == 3
        assert len(pred.std) == 3
        assert "q05" in pred.quantiles
        assert "q95" in pred.quantiles
        assert all(pred.quantiles["q05"] < pred.mean)
        assert all(pred.quantiles["q95"] > pred.mean)

    def test_training_state(self):
        """Test TrainingState dataclass."""
        from acms.ai.core.types import TrainingState, TrainingPhase

        state = TrainingState(
            phase=TrainingPhase.TRAINING,
            epoch=10,
            total_epochs=100,
            step=1000,
            total_steps=10000,
            loss=0.15,
            metrics={"accuracy": 0.85, "f1": 0.82},
        )

        assert state.phase == TrainingPhase.TRAINING
        assert state.epoch == 10
        assert state.loss == 0.15
        assert "accuracy" in state.metrics


# ============================================================================
# Test: GPU Manager
# ============================================================================


class TestGPUManager:
    """Test GPU manager functionality."""

    def test_gpu_manager_singleton(self):
        """Test GPU manager is a singleton."""
        from acms.ai.core.gpu_manager import GPUManager, get_gpu_manager

        gpu1 = GPUManager()
        gpu2 = get_gpu_manager()

        assert gpu1 is gpu2

    def test_is_gpu_available(self):
        """Test GPU availability check."""
        from acms.ai.core.gpu_manager import is_gpu_available

        available = is_gpu_available()
        assert isinstance(available, bool)

    def test_gpu_status(self):
        """Test GPU status retrieval."""
        from acms.ai.core.gpu_manager import get_gpu_manager

        gpu = get_gpu_manager()
        status = gpu.get_status()

        assert "device" in status
        assert "is_gpu" in status
        assert "num_gpus" in status
        assert "memory" in status

    def test_memory_info(self):
        """Test GPU memory information."""
        from acms.ai.core.gpu_manager import memory_info

        info = memory_info()

        assert isinstance(info, dict)
        if info:  # If GPU is available
            for device_id, mem in info.items():
                assert "total_mb" in mem
                assert "allocated_mb" in mem
                assert "free_mb" in mem

    def test_to_device(self):
        """Test tensor to device conversion."""
        from acms.ai.core.gpu_manager import to_device, is_gpu_available

        tensor = np.random.randn(10, 20).astype(np.float32)
        device_name = "cuda" if is_gpu_available() else "cpu"

        result = to_device(tensor, device_name)

        # Result should be numpy array or torch tensor depending on backend
        assert result is not None

    def test_empty_cache(self):
        """Test GPU cache clearing."""
        from acms.ai.core.gpu_manager import empty_cache

        # Should not raise
        empty_cache()


# ============================================================================
# Test: Config System
# ============================================================================


class TestConfig:
    """Test configuration system."""

    def test_ai_config_defaults(self):
        """Test AIConfig default values."""
        from acms.ai.core.config import AIConfig

        config = AIConfig()

        assert config.model_dir is not None
        assert config.data_dir is not None
        assert config.log_dir is not None
        assert isinstance(config.gpu, dict)
        assert isinstance(config.training, dict)
        assert isinstance(config.inference, dict)

    def test_gpu_config(self):
        """Test GPUConfig."""
        from acms.ai.core.config import GPUConfig

        config = GPUConfig(
            device="cuda",
            mixed_precision=True,
            memory_fraction=0.8,
        )

        assert config.device == "cuda"
        assert config.mixed_precision is True
        assert config.memory_fraction == 0.8

    def test_training_config(self):
        """Test TrainingConfig."""
        from acms.ai.core.config import TrainingConfig

        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            learning_rate=0.001,
            optimizer="adam",
        )

        assert config.epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"

    def test_inference_config(self):
        """Test InferenceConfig."""
        from acms.ai.core.config import InferenceConfig

        config = InferenceConfig(
            timeout_seconds=5.0,
            max_batch_size=64,
            enable_caching=True,
        )

        assert config.timeout_seconds == 5.0
        assert config.max_batch_size == 64
        assert config.enable_caching is True

    def test_config_validation(self):
        """Test config validation."""
        from acms.ai.core.config import AIConfig

        config = AIConfig()
        # Should not raise
        config.validate()


# ============================================================================
# Test: Tensor Utils
# ============================================================================


class TestTensorUtils:
    """Test tensor utility functions."""

    def test_standard_scaler(self):
        """Test StandardScaler."""
        from acms.ai.core.tensor_utils import StandardScaler

        scaler = StandardScaler()

        # Fit on data
        data = np.random.randn(100, 5).astype(np.float32)
        scaler.fit(data)

        # Transform
        transformed = scaler.transform(data)

        assert transformed.shape == data.shape
        assert np.abs(transformed.mean(axis=0)).max() < 0.1  # Close to 0
        assert np.abs(transformed.std(axis=0) - 1.0).max() < 0.1  # Close to 1

    def test_min_max_scaler(self):
        """Test MinMaxScaler."""
        from acms.ai.core.tensor_utils import MinMaxScaler

        scaler = MinMaxScaler(feature_range=(0, 1))

        data = np.random.randn(100, 5).astype(np.float32) * 10 + 50
        scaler.fit(data)

        transformed = scaler.transform(data)

        assert transformed.min() >= 0
        assert transformed.max() <= 1

    def test_robust_scaler(self):
        """Test RobustScaler."""
        from acms.ai.core.tensor_utils import RobustScaler

        scaler = RobustScaler()

        data = np.random.randn(100, 5).astype(np.float32)
        scaler.fit(data)

        transformed = scaler.transform(data)

        assert transformed.shape == data.shape

    def test_temporal_split(self):
        """Test temporal train/test split."""
        from acms.ai.core.tensor_utils import temporal_split

        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        X_train, X_test, y_train, y_test = temporal_split(X, y, test_size=0.2)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_walk_forward_splits(self):
        """Test walk-forward cross-validation splits."""
        from acms.ai.core.tensor_utils import walk_forward_splits

        n_samples = 100
        window_size = 20
        step_size = 10

        splits = list(
            walk_forward_splits(
                n_samples, window_size=window_size, step_size=step_size
            )
        )

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert test_idx[-1] < n_samples

    def test_pad_sequences(self):
        """Test sequence padding."""
        from acms.ai.core.tensor_utils import pad_sequences

        sequences = [
            np.array([1, 2, 3]),
            np.array([4, 5]),
            np.array([6, 7, 8, 9]),
        ]

        padded = pad_sequences(sequences, max_length=5, padding_value=0)

        assert padded.shape == (3, 5)
        assert np.array_equal(padded[0], np.array([1, 2, 3, 0, 0]))
        assert np.array_equal(padded[1], np.array([4, 5, 0, 0, 0]))
        assert np.array_equal(padded[2], np.array([6, 7, 8, 9, 0]))

    def test_tensor_cache(self):
        """Test LRU tensor cache."""
        from acms.ai.core.tensor_utils import TensorCache

        cache = TensorCache(max_size=3)

        cache.put("key1", np.array([1, 2, 3]))
        cache.put("key2", np.array([4, 5, 6]))
        cache.put("key3", np.array([7, 8, 9]))

        # key1 should be in cache
        assert cache.get("key1") is not None

        # Add new key, should evict key2 (least recently used)
        cache.put("key4", np.array([10, 11, 12]))

        # key2 should be evicted
        assert cache.get("key2") is None

    def test_sliding_window_dataset(self):
        """Test SlidingWindowDataset."""
        from acms.ai.core.tensor_utils import SlidingWindowDataset

        data = np.arange(100, dtype=np.float32)
        dataset = SlidingWindowDataset(
            data=data, window_size=10, step_size=1
        )

        assert len(dataset) == 91  # 100 - 10 + 1

        X, y = dataset[0]
        assert X.shape == (10,)
        assert isinstance(y, np.floating)


# ============================================================================
# Test: Base Models
# ============================================================================


class TestBaseModels:
    """Test base model abstractions."""

    def test_base_model_registry(self, temp_dir):
        """Test model registry."""
        from acms.ai.core.base_models import BaseModelRegistry, ModelMetadata

        registry = BaseModelRegistry(temp_dir)

        # Create mock metadata
        metadata = ModelMetadata(
            model_id="test_model",
            version="1.0.0",
            task="classification",
            created_at=datetime.utcnow(),
        )

        # Register a mock model
        registry.register("test_model", "1.0.0", metadata)

        # Get metadata
        retrieved = registry.get_metadata("test_model", "1.0.0")
        assert retrieved is not None
        assert retrieved.model_id == "test_model"

        # List versions
        versions = registry.list_versions("test_model")
        assert "1.0.0" in versions

    def test_model_metadata(self):
        """Test ModelMetadata dataclass."""
        from acms.ai.core.base_models import ModelMetadata

        metadata = ModelMetadata(
            model_id="test",
            version="1.0.0",
            task="regression",
            created_at=datetime.utcnow(),
            metrics={"loss": 0.1, "accuracy": 0.9},
        )

        assert metadata.model_id == "test"
        assert metadata.version == "1.0.0"
        assert metadata.task == "regression"

    def test_prediction_result(self):
        """Test PredictionResult dataclass."""
        from acms.ai.core.base_models import PredictionResult

        result = PredictionResult(
            prediction=np.array([1.0, 2.0, 3.0]),
            model_id="test_model",
            version="1.0.0",
            inference_time_ms=5.0,
        )

        assert len(result.prediction) == 3
        assert result.model_id == "test_model"
        assert result.inference_time_ms == 5.0


# ============================================================================
# Test: Feature Engineering
# ============================================================================


class TestFeatureEngineering:
    """Test feature engineering module."""

    def test_feature_store_basic_ops(self):
        """Test basic feature store operations."""
        from acms.ai.features.store import FeatureStore

        store = FeatureStore()

        # Write features
        features = np.random.randn(10, 5).astype(np.float32)
        store.write("test_features", features, timestamp=datetime.utcnow())

        # Read features
        retrieved = store.read("test_features")

        assert retrieved is not None
        assert retrieved.shape == features.shape

    def test_feature_engineering(self):
        """Test feature engineering functions."""
        from acms.ai.features.engineering import (
            create_lag_features,
            create_rolling_features,
            create_momentum_features,
        )

        data = np.random.randn(100, 3).astype(np.float32)

        # Lag features
        lagged = create_lag_features(data, lags=[1, 2, 3])
        assert lagged.shape[1] > data.shape[1]

        # Rolling features
        rolled = create_rolling_features(data, windows=[5, 10, 20])
        assert rolled.shape[1] > data.shape[1]

        # Momentum features
        momentum = create_momentum_features(data, periods=[5, 10])
        assert momentum.shape[1] > data.shape[1]

    def test_drift_detection(self):
        """Test drift detection."""
        from acms.ai.features.drift import DriftDetector, DriftType

        detector = DriftDetector(threshold=0.1)

        # Reference data
        ref_data = np.random.randn(1000, 5)

        # Current data (similar)
        current_data = np.random.randn(100, 5)

        detector.set_reference(ref_data)

        # Detect drift
        drift_score = detector.compute_drift_score(current_data)

        assert isinstance(drift_score, float)
        assert 0 <= drift_score <= 1


# ============================================================================
# Test: Decision Components
# ============================================================================


class TestDecisionComponents:
    """Test decision-making components."""

    def test_portfolio_ai_basic(self, sample_portfolio_state, sample_signals):
        """Test basic portfolio AI functionality."""
        from acms.ai.decision.portfolio_ai import PortfolioAI

        ai = PortfolioAI()

        # Get recommendation
        recommendation = ai.get_recommendation(
            signals=sample_signals,
            portfolio_state=sample_portfolio_state,
        )

        assert recommendation is not None
        assert "action" in recommendation
        assert "position_size" in recommendation

    def test_risk_ai_basic(self, sample_portfolio_state, sample_signals):
        """Test basic risk AI functionality."""
        from acms.ai.decision.risk_ai import RiskAI

        ai = RiskAI()

        # Get risk assessment
        assessment = ai.assess_risk(
            signals=sample_signals,
            portfolio_state=sample_portfolio_state,
        )

        assert assessment is not None
        assert "level" in assessment
        assert "score" in assessment

    def test_strategy_selector(self, sample_signals):
        """Test strategy selector."""
        from acms.ai.decision.strategy_selector import StrategySelector

        selector = StrategySelector()

        # Get best strategy
        best = selector.select_strategy(
            signals=sample_signals,
            market_conditions={"volatility": "high", "trend": "up"},
        )

        assert best is not None
        assert "strategy" in best
        assert "confidence" in best

    def test_explainer(self):
        """Test decision explainer."""
        from acms.ai.decision.explainer import DecisionExplainer

        explainer = DecisionExplainer()

        # Generate explanation
        explanation = explainer.explain(
            decision={"action": "buy", "symbol": "BTC/USDT"},
            features={"rsi": 30, "macd": 1.5},
        )

        assert explanation is not None
        assert "factors" in explanation or "reason" in explanation


# ============================================================================
# Test: Knowledge Components
# ============================================================================


class TestKnowledgeComponents:
    """Test knowledge management components."""

    def test_memory_basic(self):
        """Test basic memory functionality."""
        from acms.ai.knowledge.memory import ExperienceMemory

        memory = ExperienceMemory(max_size=100)

        # Add experience
        memory.add(
            state={"price": 45000},
            action="buy",
            reward=100,
            next_state={"price": 46000},
        )

        assert memory.size() == 1

        # Sample batch
        batch = memory.sample(batch_size=1)
        assert len(batch) == 1

    def test_graph_knowledge(self):
        """Test knowledge graph."""
        from acms.ai.knowledge.graph import KnowledgeGraph

        graph = KnowledgeGraph()

        # Add nodes
        graph.add_entity("BTC", "asset", {"price": 45000, "type": "crypto"})
        graph.add_entity("ETH", "asset", {"price": 3000, "type": "crypto"})

        # Add relationship
        graph.add_relation("BTC", "correlated_with", "ETH", {"correlation": 0.8})

        # Query
        entities = graph.get_entities_by_type("asset")
        assert len(entities) >= 2

    def test_adaptation(self):
        """Test adaptation module."""
        from acms.ai.knowledge.adaptation import ModelAdaptor

        adaptor = ModelAdaptor()

        # Simulate adaptation
        success = adaptor.adapt(
            model_id="test_model",
            feedback={"accuracy": 0.85, "latency": 10},
        )

        assert isinstance(success, bool)


# ============================================================================
# Test: Monitoring
# ============================================================================


class TestMonitoring:
    """Test monitoring components."""

    def test_metrics_collector(self):
        """Test metrics collection."""
        from acms.ai.monitoring.metrics import AIMetricsCollector

        collector = AIMetricsCollector()

        # Record metrics
        collector.record_inference(model_id="test", latency_ms=5.0, batch_size=32)
        collector.record_training(epoch=1, loss=0.1, accuracy=0.9)

        # Get metrics
        metrics = collector.get_metrics()

        assert "inference" in metrics or "training" in metrics

    def test_gpu_monitor(self):
        """Test GPU monitoring."""
        from acms.ai.monitoring.gpu_monitor import GPUMonitor

        monitor = GPUMonitor()

        # Get status
        status = monitor.get_status()

        assert status is not None
        assert "device" in status or "utilization" in status

    def test_model_monitor(self):
        """Test model monitoring."""
        from acms.ai.monitoring.model_monitor import AIModelMonitor

        monitor = AIModelMonitor()

        # Record performance
        monitor.record_prediction(
            model_id="test_model",
            prediction=1.0,
            actual=1.1,
            latency_ms=5.0,
        )

        # Check for alerts
        alerts = monitor.get_active_alerts()

        assert isinstance(alerts, list)


# ============================================================================
# Test: Inference
# ============================================================================


class TestInference:
    """Test inference components."""

    def test_ab_testing(self):
        """Test A/B testing framework."""
        from acms.ai.inference.ab_testing import ABTestManager

        manager = ABTestManager()

        # Create experiment
        experiment_id = manager.create_experiment(
            name="test_exp",
            variants=["control", "treatment"],
            metrics=["conversion", "latency"],
        )

        assert experiment_id is not None

        # Record observation
        manager.record(
            experiment_id=experiment_id,
            variant="treatment",
            metrics={"conversion": 0.05, "latency": 10.0},
        )

        # Get results
        results = manager.get_results(experiment_id)

        assert results is not None

    def test_inference_pipeline(self):
        """Test inference pipeline."""
        from acms.ai.inference.pipeline import InferencePipeline

        pipeline = InferencePipeline()

        # Add mock step
        def mock_step(data):
            return {"result": data["input"] * 2}

        pipeline.add_step("double", mock_step)

        # Execute
        result = pipeline.execute({"input": 5})

        assert result["result"] == 10


# ============================================================================
# Test: Training Components
# ============================================================================


class TestTraining:
    """Test training components."""

    def test_curriculum_learning(self):
        """Test curriculum learning."""
        from acms.ai.training.curriculum import CurriculumScheduler

        scheduler = CurriculumScheduler(
            difficulty_levels=["easy", "medium", "hard"],
            transition_criteria={"accuracy": 0.8},
        )

        # Initial level
        assert scheduler.get_current_level() == "easy"

        # Update with good performance
        scheduler.update_progress(accuracy=0.85)

        # Check if advanced
        new_level = scheduler.get_current_level()
        assert new_level in ["easy", "medium", "hard"]

    def test_online_learning(self):
        """Test online learning."""
        from acms.ai.training.online_learning import OnlineLearningManager

        manager = OnlineLearningManager()

        # Record feedback
        manager.record_feedback(
            sample={"features": np.random.randn(10)},
            prediction=0.8,
            actual=0.75,
        )

        # Check if retraining needed
        needs_retrain = manager.check_retraining_criteria()

        assert isinstance(needs_retrain, bool)

    def test_hyperopt(self):
        """Test hyperparameter optimization."""
        from acms.ai.training.hyperopt import HyperparameterOptimizer

        optimizer = HyperparameterOptimizer(
            search_space={
                "lr": [0.001, 0.01, 0.1],
                "batch_size": [16, 32, 64],
            },
            max_trials=5,
        )

        # Define objective
        def objective(params):
            return {"loss": params["lr"] * params["batch_size"] / 100}

        # Run optimization
        best_params = optimizer.optimize(objective)

        assert "lr" in best_params
        assert "batch_size" in best_params

    def test_walkforward_trainer(self):
        """Test walk-forward trainer."""
        from acms.ai.training.walkforward_trainer import WalkForwardTrainer

        trainer = WalkForwardTrainer(
            train_window=100,
            test_window=20,
            step_size=10,
        )

        # Generate synthetic data
        n = 200
        data = np.random.randn(n, 5).astype(np.float32)

        # Get splits
        splits = list(trainer.get_splits(data))

        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ============================================================================
# Test: Models (Basic)
# ============================================================================


class TestModels:
    """Test model implementations."""

    def test_tft_config(self):
        """Test TFT configuration."""
        from acms.ai.models.temporal_fusion_transformer import TFTConfig

        config = TFTConfig(
            input_dim=10,
            output_dim=1,
            num_encoder_steps=100,
            num_heads=4,
            d_model=64,
        )

        assert config.input_dim == 10
        assert config.output_dim == 1
        assert config.num_heads == 4

    def test_replay_buffer(self):
        """Test replay buffer."""
        from acms.ai.models.deep_rl_agents import ReplayBuffer

        buffer = ReplayBuffer(capacity=100)

        # Add experiences
        for i in range(10):
            buffer.add(
                state=np.random.randn(10),
                action=0,
                reward=np.random.randn(),
                next_state=np.random.randn(10),
                done=False,
            )

        assert buffer.size() == 10

        # Sample batch
        batch = buffer.sample(batch_size=5)

        assert len(batch) == 5

    def test_prioritized_replay_buffer(self):
        """Test prioritized replay buffer."""
        from acms.ai.models.deep_rl_agents import PrioritizedReplayBuffer

        buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6)

        # Add with priority
        buffer.add(
            state=np.random.randn(10),
            action=0,
            reward=1.0,
            next_state=np.random.randn(10),
            done=False,
            priority=1.0,
        )

        assert buffer.size() == 1

    def test_ensemble_diversity_tracker(self):
        """Test ensemble diversity tracking."""
        from acms.ai.models.ensemble_orchestrator import EnsembleDiversityTracker

        tracker = EnsembleDiversityTracker()

        # Add predictions
        pred1 = np.random.randn(100)
        pred2 = np.random.randn(100)
        pred3 = np.random.randn(100)

        tracker.add_predictions("model1", pred1)
        tracker.add_predictions("model2", pred2)
        tracker.add_predictions("model3", pred3)

        # Get diversity
        diversity = tracker.get_diversity()

        assert isinstance(diversity, float)
        assert diversity >= 0

    def test_sentiment_data_point(self):
        """Test sentiment data structures."""
        from acms.ai.models.sentiment_nlp import SentimentDataPoint

        dp = SentimentDataPoint(
            text="Bitcoin is going to the moon!",
            sentiment_score=0.8,
            confidence=0.9,
            timestamp=datetime.utcnow(),
        )

        assert dp.sentiment_score == 0.8
        assert dp.confidence == 0.9


# ============================================================================
# Test: Integration
# ============================================================================


class TestIntegration:
    """Test integration components."""

    def test_cache_config(self):
        """Test cache configuration."""
        from acms.ai.integration.redis_cache import CacheConfig

        config = CacheConfig(
            host="localhost",
            port=6379,
            ttl_seconds=300,
        )

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.ttl_seconds == 300

    def test_kafka_consumer_config(self):
        """Test Kafka consumer configuration."""
        from acms.ai.integration.kafka_consumer import AIConsumerConfig

        config = AIConsumerConfig(
            bootstrap_servers="localhost:9092",
            topics=["market_data", "signals"],
            group_id="ai_consumer_group",
        )

        assert config.bootstrap_servers == "localhost:9092"
        assert len(config.topics) == 2

    def test_ai_orchestrator_state(self):
        """Test AI orchestrator state machine."""
        from acms.ai.integration.orchestrator import (
            AIOrchestratorState,
            ModelLifecycleState,
        )

        # Test orchestrator states
        assert AIOrchestratorState.STOPPED.value == "stopped"
        assert AIOrchestratorState.RUNNING.value == "running"
        assert AIOrchestratorState.DEGRADED.value == "degraded"

        # Test model lifecycle states
        assert ModelLifecycleState.READY.value == "ready"
        assert ModelLifecycleState.SERVING.value == "serving"
        assert ModelLifecycleState.RETIRED.value == "retired"


# ============================================================================
# Test: System Integration
# ============================================================================


class TestSystemIntegration:
    """Test overall system integration."""

    def test_ai_module_version(self):
        """Test AI module version."""
        from acms.ai import get_version

        version = get_version()
        assert version == "0.1.0"

    def test_is_available(self):
        """Test AI module availability."""
        from acms.ai import is_available, is_gpu_ready

        assert is_available() is True
        # GPU readiness depends on hardware
        gpu_ready = is_gpu_ready()
        assert isinstance(gpu_ready, bool)

    def test_system_info(self):
        """Test system info retrieval."""
        from acms.ai import get_system_info

        info = get_system_info()

        assert "ai_version" in info
        assert "numpy_available" in info
        assert "torch_available" in info
        assert "cuda_available" in info

    def test_module_imports(self):
        """Test that all major module imports work."""
        # Core
        from acms.ai.core.config import AIConfig
        from acms.ai.core.gpu_manager import GPUManager
        from acms.ai.core.base_models import BaseModel, BaseModelRegistry
        from acms.ai.core.tensor_utils import StandardScaler
        from acms.ai.core.types import MarketRegime, SignalPrediction

        # Models
        from acms.ai.models.temporal_fusion_transformer import TemporalFusionTransformer
        from acms.ai.models.deep_rl_agents import ReplayBuffer, PPOAgent
        from acms.ai.models.ensemble_orchestrator import DynamicWeightedEnsemble

        # Training
        from acms.ai.training.curriculum import CurriculumScheduler
        from acms.ai.training.online_learning import OnlineLearningManager
        from acms.ai.training.hyperopt import HyperparameterOptimizer

        # Features
        from acms.ai.features.store import FeatureStore
        from acms.ai.features.drift import DriftDetector

        # Decision
        from acms.ai.decision.portfolio_ai import PortfolioAI
        from acms.ai.decision.risk_ai import RiskAI

        # Knowledge
        from acms.ai.knowledge.memory import ExperienceMemory
        from acms.ai.knowledge.graph import KnowledgeGraph

        # Monitoring
        from acms.ai.monitoring.metrics import AIMetricsCollector
        from acms.ai.monitoring.gpu_monitor import GPUMonitor

        # Integration
        from acms.ai.integration.orchestrator import AIOrchestratorState

        # All imports successful
        assert True

    def test_config_loading(self):
        """Test configuration loading."""
        from acms.ai.core.config import load_config

        # Should create a valid config
        config = load_config()

        assert config is not None


# ============================================================================
# Test: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        from acms.ai.core.tensor_utils import StandardScaler

        scaler = StandardScaler()

        # Should handle empty gracefully
        data = np.array([]).reshape(0, 5)
        # Fit on empty should not crash
        # (implementation may raise or return gracefully)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        from acms.ai.core.tensor_utils import StandardScaler

        scaler = StandardScaler()

        data = np.random.randn(100, 5).astype(np.float32)
        data[0, 0] = np.nan  # Add NaN

        scaler.fit(data)
        transformed = scaler.transform(data)

        assert not np.isnan(transformed).all()

    def test_invalid_config_handling(self):
        """Test handling of invalid configurations."""
        from acms.ai.core.config import AIConfig

        # Should create with defaults even if invalid
        config = AIConfig()
        config.validate()  # Should not raise

    def test_zero_division_protection(self):
        """Test protection against division by zero."""
        from acms.ai.core.tensor_utils import StandardScaler

        scaler = StandardScaler()

        # Constant data (zero variance)
        data = np.ones((100, 5)).astype(np.float32)

        scaler.fit(data)
        transformed = scaler.transform(data)

        # Should not produce NaN
        assert not np.any(np.isnan(transformed))

    def test_large_batch_handling(self):
        """Test handling of large batches."""
        from acms.ai.core.gpu_manager import to_device

        # Large batch
        data = np.random.randn(10000, 100).astype(np.float32)

        # Should handle without crashing
        result = to_device(data, "cpu")

        assert result is not None

    def test_concurrent_access(self):
        """Test concurrent access to shared resources."""
        from acms.ai.core.gpu_manager import get_gpu_manager

        # Get same instance
        gpu1 = get_gpu_manager()
        gpu2 = get_gpu_manager()

        # Should be same instance
        assert gpu1 is gpu2

        # Both should have valid status
        assert gpu1.get_status() is not None
        assert gpu2.get_status() is not None


# ============================================================================
# Performance Benchmarks (Light)
# ============================================================================


class TestPerformance:
    """Light performance tests."""

    def test_tensor_scaling_performance(self):
        """Test tensor scaling performance."""
        from acms.ai.core.tensor_utils import StandardScaler
        import time

        # Create large dataset
        data = np.random.randn(10000, 100).astype(np.float32)

        scaler = StandardScaler()

        start = time.time()
        scaler.fit(data)
        transformed = scaler.transform(data)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second for 10k samples)
        assert elapsed < 1.0
        assert transformed.shape == data.shape

    def test_gpu_availability_check_performance(self):
        """Test GPU availability check performance."""
        from acms.ai.core.gpu_manager import is_gpu_available
        import time

        start = time.time()
        for _ in range(100):
            is_gpu_available()
        elapsed = time.time() - start

        # 100 checks should be very fast
        assert elapsed < 0.1


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
