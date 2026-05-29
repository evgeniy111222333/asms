"""Ensemble model combining multiple ML models."""

import numpy as np
from typing import Optional, List, Any


class EnsembleModel:
    """Ensemble model combining multiple ML models.

    Supports:
    - voting: Majority or weighted voting from base models
    - stacking: Base model predictions fed into a meta-learner
    """

    def __init__(self, models: List = None, method: str = "voting",
                 weights: Optional[List[float]] = None):
        self.models = models or []
        self.method = method
        self.weights = weights
        self.meta_model = None

    def add_model(self, model: Any) -> None:
        """Add a model to the ensemble.

        Args:
            model: Any object with fit/predict methods.
        """
        self.models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Fit all models in the ensemble.

        For stacking, also trains a logistic regression meta-model on
        base model predictions.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for chaining.
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        if self.method == "voting":
            for model in self.models:
                model.fit(X, y)
        elif self.method == "stacking":
            for model in self.models:
                model.fit(X, y)
            meta_features = self._generate_meta_features(X)
            try:
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(max_iter=1000)
                self.meta_model.fit(meta_features, y)
            except ImportError:
                self.meta_model = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the ensemble.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        if self.method == "voting":
            return self._voting_predict(X)
        elif self.method == "stacking":
            return self._stacking_predict(X)
        return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the ensemble (voting only).

        Args:
            X: Feature matrix.

        Returns:
            Array of average probabilities across models.
        """
        probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probas.append(model.predict_proba(X))
        if not probas:
            raise ValueError("No models support predict_proba")
        return np.mean(probas, axis=0)

    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Majority/weighted voting prediction."""
        predictions = np.array([model.predict(X) for model in self.models])
        if self.weights is not None and len(self.weights) == len(self.models):
            weighted = np.zeros(len(X))
            for i, w in enumerate(self.weights):
                weighted += w * (predictions[i] == 1).astype(float)
            return (weighted > sum(self.weights) / 2).astype(int)
        else:
            return (np.mean(predictions, axis=0) > 0.5).astype(int)

    def _stacking_predict(self, X: np.ndarray) -> np.ndarray:
        """Stacking prediction using meta model."""
        if self.meta_model is None:
            return self._voting_predict(X)
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict(meta_features)

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta features from base model predictions."""
        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.T


__all__ = ["EnsembleModel"]
