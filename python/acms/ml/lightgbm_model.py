"""LightGBM gradient boosting model for signal generation."""

import numpy as np
from typing import Optional, Dict
from pathlib import Path

from acms.ml.config import MLConfig


class LightGBMSignalModel:
    """LightGBM gradient boosting model for signal generation.

    Supports multi-class classification (down/neutral/up) with
    early stopping and feature importance tracking.
    """

    def __init__(self, objective: str = "multiclass", num_classes: int = 3):
        self.objective = objective
        self.num_classes = num_classes
        self.model = None
        self.feature_importance = None

    def train(self, X: np.ndarray, y: np.ndarray, config: Optional[MLConfig] = None) -> Dict:
        """Train the LightGBM model.

        Args:
            X: Feature matrix.
            y: Target labels.
            config: Training configuration.

        Returns:
            Dict with best iteration and feature importance.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required")

        if len(X) < 10:
            raise ValueError("Not enough data to train")

        config = config or MLConfig()
        split = int(len(X) * config.train_test_split)
        if split < 5 or (len(X) - split) < 5:
            split = max(5, len(X) - 5)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            "objective": self.objective, "num_class": self.num_classes,
            "metric": "multi_logloss", "boosting_type": "gbdt",
            "num_leaves": 63, "learning_rate": config.learning_rate,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "verbose": -1,
        }
        self.model = lgb.train(
            params, train_data, num_boost_round=config.epochs,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(config.early_stopping_patience)],
        )
        self.feature_importance = self.model.feature_importance(importance_type="gain")
        return {
            "best_iteration": self.model.best_iteration,
            "feature_importance": self.feature_importance.tolist(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class indices.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of class probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required")
        self.model = lgb.Booster(model_file=path)
        self.is_trained = True


# ============================================================================
# Optuna Hyperparameter Optimization
# ============================================================================



__all__ = ["LightGBMSignalModel"]
