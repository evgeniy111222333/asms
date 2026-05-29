"""Risk Budgeting for ACMS."""

import numpy as np
from typing import Dict, List, Optional


class RiskBudgeting:
    """Risk budgeting - allocate risk budget across strategies.

    Ensures each strategy receives a fair allocation of total
    portfolio risk, and enforces per-strategy risk limits.
    """

    def __init__(self, total_risk_budget: float = 1.0,
                 max_strategy_risk_pct: float = 0.40):
        """Initialize risk budgeting.

        Args:
            total_risk_budget: Total risk budget (1.0 = 100%).
            max_strategy_risk_pct: Maximum risk allocation per strategy.
        """
        self.total_risk_budget = total_risk_budget
        self.max_strategy_risk_pct = max_strategy_risk_pct
        self._strategy_risk_usage: Dict[str, float] = {}
        self._strategy_risk_budgets: Dict[str, float] = {}

    def allocate_budget(self, strategies: List[str],
                        target_contributions: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Allocate risk budget across strategies.

        Args:
            strategies: List of strategy identifiers.
            target_contributions: Target risk contribution per strategy.
                If None, equal risk budget allocation.

        Returns:
            Dict mapping strategy to risk budget allocation.
        """
        n = len(strategies)
        if n == 0:
            return {}

        if target_contributions is None:
            per_strategy = self.total_risk_budget / n
            # Cap each strategy at max
            per_strategy = min(per_strategy, self.max_strategy_risk_pct)
        else:
            per_strategy_arr = target_contributions.copy()
            per_strategy_arr = np.minimum(per_strategy_arr, self.max_strategy_risk_pct)
            if per_strategy_arr.sum() > self.total_risk_budget:
                per_strategy_arr *= self.total_risk_budget / per_strategy_arr.sum()
            per_strategy = per_strategy_arr  # type: ignore

        budgets = {}
        for i, strategy in enumerate(strategies):
            if isinstance(per_strategy, np.ndarray):
                budgets[strategy] = float(per_strategy[i])
            else:
                budgets[strategy] = float(per_strategy)
            self._strategy_risk_budgets[strategy] = budgets[strategy]

        return budgets

    def check_budget_utilization(self, strategy: str,
                                  current_risk_usage: float) -> Dict:
        """Check if a strategy is within its risk budget.

        Args:
            strategy: Strategy identifier.
            current_risk_usage: Current risk usage for the strategy.

        Returns:
            Dict with budget utilization details.
        """
        self._strategy_risk_usage[strategy] = current_risk_usage
        budget = self._strategy_risk_budgets.get(strategy, 0)
        utilization = current_risk_usage / budget if budget > 0 else float('inf')
        over_budget = utilization > 1.0

        return {
            "strategy": strategy,
            "budget": budget,
            "usage": current_risk_usage,
            "utilization_pct": float(utilization * 100),
            "over_budget": over_budget,
            "remaining_budget": max(0, budget - current_risk_usage),
        }

    def compute_risk_contribution_targets(self, strategy_returns: Dict[str, np.ndarray],
                                           cov_matrix: np.ndarray,
                                           strategy_indices: Dict[str, List[int]]) -> Dict:
        """Compute target risk contributions based on strategy characteristics.

        Args:
            strategy_returns: Dict mapping strategy to its returns array.
            cov_matrix: Full portfolio covariance matrix.
            strategy_indices: Dict mapping strategy to asset indices.

        Returns:
            Dict with target risk contributions per strategy.
        """
        strategy_vols = {}
        for name, rets in strategy_returns.items():
            if len(rets) > 0:
                strategy_vols[name] = np.std(rets)
            else:
                strategy_vols[name] = 0.0

        total_vol = sum(strategy_vols.values())
        if total_vol == 0:
            return {name: 1.0 / len(strategy_vols) for name in strategy_vols}

        # Allocate inversely proportional to volatility (risk parity style)
        inverse_vols = {name: 1.0 / v if v > 0 else 0 for name, v in strategy_vols.items()}
        inv_total = sum(inverse_vols.values())
        if inv_total == 0:
            return {name: 1.0 / len(strategy_vols) for name in strategy_vols}

        targets = {name: inv / inv_total for name, inv in inverse_vols.items()}
        return targets

__all__ = ['RiskBudgeting']
