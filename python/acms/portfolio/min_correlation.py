"""Minimum correlation portfolio optimization."""

import numpy as np


class MinimumCorrelationAlgorithm:
    """Minimum Correlation Algorithm.

    Heuristic approach that finds weights minimizing
    average portfolio correlation. Long-only constraint enforced.
    """

    def optimize(self, corr_matrix: np.ndarray) -> dict:
        """Find minimum correlation portfolio weights."""
        n = corr_matrix.shape[0]
        if n < 2:
            return {"weights": np.array([1.0]), "avg_correlation": 0.0}

        avg_corr = np.mean(np.abs(corr_matrix - np.eye(n)), axis=1)
        ranks = np.argsort(avg_corr)

        weights = np.zeros(n)
        for i, rank in enumerate(ranks):
            weights[rank] = (n - i) / (n * (n + 1) / 2)

        wc = weights @ np.abs(corr_matrix) @ weights
        avg = float(wc)

        return {"weights": weights, "avg_correlation": avg}


__all__ = [
    "MinimumCorrelationAlgorithm",
]
