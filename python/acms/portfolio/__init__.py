"""Portfolio Engine - Portfolio optimization and management.

Implements:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Kelly Criterion allocation
- Black-Litterman
- Hierarchical Risk Parity (HRP)
- Maximum Diversification Portfolio
- Minimum Correlation Algorithm
- CVaR Portfolio Optimization with linear programming
- Risk budgeting with CVaR constraints
- Dynamic rebalancing triggers
- Transaction cost modeling
- Leverage optimization
- Hedging strategies
- Portfolio reconciliation
"""

from acms.portfolio.config import PortfolioConfig
from acms.portfolio.mean_variance import MeanVarianceOptimizer
from acms.portfolio.risk_parity import RiskParityOptimizer
from acms.portfolio.hrp import HierarchicalRiskParity
from acms.portfolio.max_diversification import MaximumDiversificationPortfolio
from acms.portfolio.min_correlation import MinimumCorrelationAlgorithm
from acms.portfolio.cvar import CVaRPortfolioOptimization, CVaRRiskBudgeting
from acms.portfolio.black_litterman import BlackLitterman
from acms.portfolio.kelly import KellyAllocator
from acms.portfolio.leverage import LeverageOptimizer
from acms.portfolio.rebalancing import DynamicRebalancing
from acms.portfolio.transaction_costs import TransactionCostModel
from acms.portfolio.engine import PortfolioEngine

__all__ = [
    # Config
    "PortfolioConfig",
    # Optimizers
    "MeanVarianceOptimizer", "RiskParityOptimizer", "HierarchicalRiskParity",
    "MaximumDiversificationPortfolio", "MinimumCorrelationAlgorithm",
    "CVaRPortfolioOptimization", "CVaRRiskBudgeting",
    "BlackLitterman", "KellyAllocator", "LeverageOptimizer",
    # Rebalancing & Costs
    "DynamicRebalancing", "TransactionCostModel",
    # Engine
    "PortfolioEngine",
]
