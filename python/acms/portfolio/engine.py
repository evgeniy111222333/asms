"""Portfolio engine - unified interface for all portfolio optimization methods."""

from typing import Optional, List, Dict
from datetime import datetime

import numpy as np

from acms.core import PortfolioSnapshot
from acms.portfolio.config import PortfolioConfig
from acms.portfolio.mean_variance import MeanVarianceOptimizer
from acms.portfolio.risk_parity import RiskParityOptimizer
from acms.portfolio.hrp import HierarchicalRiskParity
from acms.portfolio.max_diversification import MaximumDiversificationPortfolio
from acms.portfolio.min_correlation import MinimumCorrelationAlgorithm
from acms.portfolio.cvar import CVaRPortfolioOptimization, CVaRRiskBudgeting
from acms.portfolio.kelly import KellyAllocator
from acms.portfolio.black_litterman import BlackLitterman
from acms.portfolio.leverage import LeverageOptimizer
from acms.portfolio.rebalancing import DynamicRebalancing
from acms.portfolio.transaction_costs import TransactionCostModel


class PortfolioEngine:
    """Main portfolio management engine.

    Integrates all optimization methods and provides a unified
    interface for portfolio construction, rebalancing, and monitoring.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.mv_optimizer = MeanVarianceOptimizer(self.config)
        self.rp_optimizer = RiskParityOptimizer()
        self.hrp_optimizer = HierarchicalRiskParity()
        self.max_div_optimizer = MaximumDiversificationPortfolio()
        self.min_corr_optimizer = MinimumCorrelationAlgorithm()
        self.cvar_optimizer = CVaRPortfolioOptimization()
        self.cvar_budget = CVaRRiskBudgeting()
        self.kelly_allocator = KellyAllocator()
        self.bl_model = BlackLitterman()
        self.rebalancing = DynamicRebalancing(
            self.config.rebalance_threshold,
            transaction_cost_bps=self.config.transaction_cost_bps,
        )
        self.leverage_optimizer = LeverageOptimizer(max_leverage=self.config.max_leverage)
        self.transaction_cost_model = TransactionCostModel(
            fixed_cost_usd=self.config.fixed_cost_usd,
            proportional_cost_bps=self.config.proportional_cost_bps,
            market_impact_alpha=self.config.market_impact_alpha,
        )

    def optimize_portfolio(self, method: str, expected_returns: np.ndarray,
                           cov_matrix: np.ndarray, **kwargs) -> dict:
        """Optimize portfolio allocation using specified method."""
        if method == "mean_variance":
            return self.mv_optimizer.optimize(expected_returns, cov_matrix, kwargs.get("target_return"))
        elif method == "risk_parity":
            return self.rp_optimizer.optimize(cov_matrix)
        elif method == "hrp":
            return self.hrp_optimizer.optimize(kwargs.get("returns_matrix", np.eye(len(expected_returns))))
        elif method == "max_diversification":
            return self.max_div_optimizer.optimize(cov_matrix)
        elif method == "min_correlation":
            return self.min_corr_optimizer.optimize(kwargs.get("corr_matrix", np.eye(len(expected_returns))))
        elif method == "cvar":
            return self.cvar_optimizer.optimize(kwargs.get("returns_matrix", np.eye(len(expected_returns))))
        elif method == "cvar_budget":
            return self.cvar_budget.optimize(kwargs.get("returns_matrix", np.eye(len(expected_returns))),
                                             kwargs.get("risk_budget"))
        elif method == "black_litterman":
            return self.bl_model.compute(
                kwargs.get("market_weights", np.ones(len(expected_returns)) / len(expected_returns)),
                cov_matrix, views=kwargs.get("views"), view_confidence=kwargs.get("view_confidence"),
                view_returns=kwargs.get("view_returns"),
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def compute_rebalance_trades(self, current_weights: np.ndarray,
                                 target_weights: np.ndarray, total_value: float,
                                 threshold: Optional[float] = None) -> List[dict]:
        """Compute trades needed to rebalance portfolio."""
        threshold = threshold or self.config.rebalance_threshold
        trades = []
        for i, (curr, target) in enumerate(zip(current_weights, target_weights)):
            diff = target - curr
            if abs(diff) > threshold:
                trades.append({
                    "asset_index": i, "weight_change": diff,
                    "notional_change": diff * total_value,
                    "action": "buy" if diff > 0 else "sell",
                })
        return trades

    def reconcile(self, expected: PortfolioSnapshot, actual: PortfolioSnapshot) -> dict:
        """Reconcile expected vs actual portfolio state."""
        discrepancies = []
        if abs(expected.total_value - actual.total_value) > 1.0:
            discrepancies.append({
                "field": "total_value",
                "expected": expected.total_value,
                "actual": actual.total_value,
                "diff": actual.total_value - expected.total_value,
            })
        for exp_pos in expected.positions:
            act_pos = next((p for p in actual.positions if p.symbol == exp_pos.symbol), None)
            if act_pos is None:
                discrepancies.append({"field": "missing_position", "symbol": exp_pos.symbol})
            elif abs(exp_pos.quantity - act_pos.quantity) > 1e-8:
                discrepancies.append({
                    "field": "quantity", "symbol": exp_pos.symbol,
                    "expected": exp_pos.quantity, "actual": act_pos.quantity,
                })
        return {
            "is_reconciled": len(discrepancies) == 0,
            "discrepancies": discrepancies,
            "timestamp": datetime.utcnow().isoformat(),
        }


__all__ = [
    "PortfolioEngine",
]
