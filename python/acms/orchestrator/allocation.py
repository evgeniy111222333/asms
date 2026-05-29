"""Strategy allocation management."""

from typing import Optional, Dict, List
from collections import defaultdict

import numpy as np


class StrategyAllocationManager:
    """Manages strategy capital allocation.

    Supports:
    - Equal weight allocation
    - Risk parity allocation
    - Custom allocation with weights
    """

    def __init__(self, method: str = "equal_weight",
                 custom_weights: Optional[Dict[str, float]] = None):
        self.method = method
        self.custom_weights = custom_weights or {}
        self._strategy_returns: Dict[str, List[float]] = defaultdict(list)
        self._strategy_volatilities: Dict[str, float] = {}

    def set_allocation(self, weights: Dict[str, float]) -> None:
        """Set custom allocation weights."""
        total = sum(weights.values())
        if total > 0:
            self.custom_weights = {k: v / total for k, v in weights.items()}
        self.method = "custom"

    def get_allocation(self, strategy_ids: List[str], total_capital: float) -> Dict[str, float]:
        """Get capital allocation for each strategy."""
        if not strategy_ids:
            return {}

        if self.method == "equal_weight":
            weight_per_strategy = 1.0 / len(strategy_ids)
            return {sid: total_capital * weight_per_strategy for sid in strategy_ids}
        elif self.method == "risk_parity":
            return self._risk_parity_allocation(strategy_ids, total_capital)
        elif self.method == "custom":
            allocation = {}
            for sid in strategy_ids:
                weight = self.custom_weights.get(sid, 1.0 / len(strategy_ids))
                allocation[sid] = total_capital * weight
            return allocation

        return {sid: total_capital / len(strategy_ids) for sid in strategy_ids}

    def _risk_parity_allocation(self, strategy_ids: List[str],
                                total_capital: float) -> Dict[str, float]:
        """Risk parity allocation based on inverse volatility."""
        volatilities = {}
        for sid in strategy_ids:
            volatilities[sid] = self._strategy_volatilities.get(sid, 0.20)

        inv_vols = {sid: 1.0 / max(v, 1e-6) for sid, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        allocation = {}
        for sid in strategy_ids:
            weight = inv_vols[sid] / total_inv_vol if total_inv_vol > 0 else 1.0 / len(strategy_ids)
            allocation[sid] = total_capital * weight

        return allocation

    def update_performance(self, strategy_id: str, return_pct: float) -> None:
        """Update strategy performance for risk parity calculations."""
        self._strategy_returns[strategy_id].append(return_pct)
        if len(self._strategy_returns[strategy_id]) > 252:
            self._strategy_returns[strategy_id] = self._strategy_returns[strategy_id][-252:]
        returns = self._strategy_returns[strategy_id]
        if len(returns) > 10:
            self._strategy_volatilities[strategy_id] = float(np.std(returns)) * np.sqrt(252)


__all__ = [
    "StrategyAllocationManager",
]
