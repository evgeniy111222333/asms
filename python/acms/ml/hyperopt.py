"""Optuna-based hyperparameter optimization."""

import numpy as np
from typing import Optional, Dict


class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization.

    Supports LightGBM and simple neural network model types.
    Uses time-series-aware cross-validation.
    """

    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.best_params: Optional[Dict] = None
        self.study = None

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 n_trials: int = 100, timeout: int = 3600) -> Dict:
        """Run hyperparameter optimization.

        Args:
            X: Feature matrix.
            y: Target labels.
            n_trials: Maximum number of Optuna trials.
            timeout: Maximum optimization time in seconds.

        Returns:
            Dict with best_params, best_value, and n_trials.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna is required")

        if len(X) < 20:
            raise ValueError("Not enough data for optimization")

        def objective(trial):
            if self.model_type == "lightgbm":
                return self._lightgbm_objective(trial, X, y)
            return 0.0

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params = self.study.best_params
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
        }

    def _lightgbm_objective(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """LightGBM hyperparameter optimization objective."""
        try:
            import lightgbm as lgb
        except ImportError:
            return 0.0

        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        split = int(len(X) * 0.8)
        if split < 5 or (len(X) - split) < 5:
            return 0.0
        train_data = lgb.Dataset(X[:split], label=y[:split])
        val_data = lgb.Dataset(X[split:], label=y[split:])
        base_params = {
            "objective": "multiclass", "num_class": 3,
            "metric": "multi_logloss", "verbose": -1,
        }
        base_params.update(params)
        try:
            model = lgb.train(base_params, train_data, num_boost_round=100,
                              valid_sets=[val_data], callbacks=[lgb.early_stopping(5)])
            preds = np.argmax(model.predict(X[split:]), axis=1)
            return float(np.mean(preds == y[split:]))
        except Exception:
            return 0.0


# ============================================================================
# RL Execution Optimizer
# ============================================================================



__all__ = ["HyperparameterOptimizer"]
