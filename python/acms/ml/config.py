"""ML configuration."""

from dataclasses import dataclass


class MLConfig:
    """Configuration for ML model training and evaluation."""
    model_dir: str = "/data/acms/models"
    feature_window: int = 60
    prediction_horizon: int = 5
    train_test_split: float = 0.8
    validation_split: float = 0.1
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    optuna_trials: int = 100
    optuna_timeout: int = 3600

__all__ = ["MLConfig"]
