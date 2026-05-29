"""Bayesian confidence tracking for signal accuracy."""

import numpy as np


class BayesianConfidenceTracker:
    """Bayesian update for signal confidence scoring.

    Maintains a running confidence estimate for each indicator,
    updating based on whether the signal correctly predicted
    subsequent price moves. Uses Beta distribution conjugate prior.
    """

    def __init__(self, num_indicators: int = 13, prior: float = 0.5, decay: float = 0.95):
        """Initialize the tracker.

        Args:
            num_indicators: Number of sub-signals to track.
            prior: Initial confidence prior (0-1).
            decay: Decay factor for old observations.
        """
        self.prior = prior
        self.decay = decay
        self.alpha = np.full(num_indicators, prior * 20 + 1)
        self.beta = np.full(num_indicators, (1 - prior) * 20 + 1)
        self.confidences = np.full(num_indicators, prior)

    def update(self, indicator_idx: int, was_correct: bool) -> float:
        """Update confidence for an indicator based on outcome.

        Args:
            indicator_idx: Index of the indicator.
            was_correct: Whether the signal was correct.

        Returns:
            Updated confidence value.
        """
        if indicator_idx < 0 or indicator_idx >= len(self.confidences):
            return self.prior
        self.alpha[indicator_idx] *= self.decay
        self.beta[indicator_idx] *= self.decay
        if was_correct:
            self.alpha[indicator_idx] += 1
        else:
            self.beta[indicator_idx] += 1
        total = self.alpha[indicator_idx] + self.beta[indicator_idx]
        if total == 0:
            return self.prior
        self.confidences[indicator_idx] = self.alpha[indicator_idx] / total
        return self.confidences[indicator_idx]

    def update_all(self, was_correct: bool) -> None:
        """Update all indicators with the same outcome (simplified batch update)."""
        for i in range(len(self.confidences)):
            self.update(i, was_correct)

    def get_weights(self) -> np.ndarray:
        """Get normalized confidence-based weights for signal combination."""
        total = self.confidences.sum()
        if total == 0:
            return np.ones_like(self.confidences) / len(self.confidences)
        return self.confidences / total

    def get_confidence(self) -> float:
        """Get overall mean confidence across all indicators."""
        return float(np.mean(self.confidences))


__all__ = [
    "BayesianConfidenceTracker",
]
