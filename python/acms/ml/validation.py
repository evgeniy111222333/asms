"""Walk-forward validation for time series ML models."""

import numpy as np
from typing import List, Dict, Tuple


class WalkForwardValidation:
    """Walk-forward validation for time series ML models.

    Properly handles temporal ordering by training on past data
    and testing on future data, avoiding look-ahead bias.
    Supports configurable gap between train and test sets.
    """

    def __init__(self, n_splits: int = 5, train_pct: float = 0.7, gap: int = 0):
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.gap = gap

    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/test splits.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            List of (train_indices, test_indices) tuples preserving temporal order.
        """
        n = len(X)
        splits = []
        test_size = n // (self.n_splits + 1)
        if test_size < 10:
            return [(np.arange(n - 1), np.array([n - 1]))]

        for i in range(self.n_splits):
            test_end = n - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap
            train_start = 0

            if train_end <= 0:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def validate(self, X: np.ndarray, y: np.ndarray, model_factory: callable,
                 metric: callable = None) -> Dict:
        """Run walk-forward validation.

        Args:
            X: Feature matrix.
            y: Target labels.
            model_factory: Callable that returns a new model instance with fit/predict.
            metric: Callable(y_true, y_pred) returning a score.

        Returns:
            Dict with fold scores and aggregate statistics.
        """
        if metric is None:
            metric = lambda yt, yp: np.mean(yt == yp)

        splits = self.split(X, y)
        scores = []

        for train_idx, test_idx in splits:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric(y_test, y_pred)
            scores.append(score)

        return {
            "fold_scores": scores,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "n_splits": len(scores),
        }


__all__ = ["WalkForwardValidation"]
