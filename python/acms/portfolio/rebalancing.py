"""Dynamic portfolio rebalancing."""

from typing import Dict, Optional
from datetime import datetime

import numpy as np


class DynamicRebalancing:
    """Dynamic portfolio rebalancing triggers.

    Supports three rebalancing approaches:
    - Threshold-based: rebalance when any weight drifts beyond threshold
    - Time-based: rebalance at fixed intervals
    - Drift-based: rebalance based on cumulative portfolio drift
    """

    def __init__(self, threshold: float = 0.05, time_interval_days: int = 30,
                 max_drift: float = 0.10, transaction_cost_bps: float = 10.0):
        self.threshold = threshold
        self.time_interval_days = time_interval_days
        self.max_drift = max_drift
        self.transaction_cost_bps = transaction_cost_bps
        self._last_rebalance: Optional[datetime] = None

    def check_threshold_rebalance(self, current_weights: np.ndarray,
                                  target_weights: np.ndarray) -> bool:
        """Check if any weight has drifted beyond threshold."""
        drift = np.abs(current_weights - target_weights)
        return bool(np.any(drift > self.threshold))

    def check_time_rebalance(self, current_time: datetime) -> bool:
        """Check if time-based rebalance is due."""
        if self._last_rebalance is None:
            return True
        elapsed_days = (current_time - self._last_rebalance).days
        return elapsed_days >= self.time_interval_days

    def check_drift_rebalance(self, current_weights: np.ndarray,
                              target_weights: np.ndarray) -> bool:
        """Check if cumulative drift exceeds maximum."""
        total_drift = np.sum(np.abs(current_weights - target_weights))
        return total_drift > self.max_drift

    def should_rebalance(self, current_weights: np.ndarray, target_weights: np.ndarray,
                         current_time: datetime) -> Dict:
        """Check all rebalancing triggers."""
        threshold_trigger = self.check_threshold_rebalance(current_weights, target_weights)
        time_trigger = self.check_time_rebalance(current_time)
        drift_trigger = self.check_drift_rebalance(current_weights, target_weights)

        should = threshold_trigger or time_trigger or drift_trigger
        reasons = []
        if threshold_trigger:
            reasons.append("threshold_breach")
        if time_trigger:
            reasons.append("time_interval")
        if drift_trigger:
            reasons.append("drift_exceeded")

        return {
            "should_rebalance": should,
            "reasons": reasons,
            "max_weight_drift": float(np.max(np.abs(current_weights - target_weights))),
            "total_drift": float(np.sum(np.abs(current_weights - target_weights))),
        }

    def compute_rebalance_cost(self, current_weights: np.ndarray,
                               target_weights: np.ndarray,
                               portfolio_value: float) -> Dict:
        """Compute transaction costs for rebalancing."""
        trades = np.abs(target_weights - current_weights)
        total_turnover = np.sum(trades) * portfolio_value
        cost = total_turnover * self.transaction_cost_bps / 10000
        return {
            "total_turnover": float(total_turnover),
            "transaction_cost": float(cost),
            "cost_bps": float(self.transaction_cost_bps),
        }


__all__ = [
    "DynamicRebalancing",
]
