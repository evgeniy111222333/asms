"""Portfolio configuration."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PortfolioConfig:
    """Portfolio management configuration."""
    target_return: Optional[float] = None
    risk_free_rate: float = 0.0
    max_weight: float = 0.40
    min_weight: float = 0.0
    rebalance_threshold: float = 0.05
    transaction_cost_bps: float = 10.0
    max_leverage: float = 3.0
    # Transaction cost model parameters
    fixed_cost_usd: float = 1.0
    proportional_cost_bps: float = 5.0
    market_impact_alpha: float = 0.1
    # Rebalancing
    rebalance_interval_days: int = 30
    max_drift: float = 0.10


__all__ = [
    "PortfolioConfig",
]
