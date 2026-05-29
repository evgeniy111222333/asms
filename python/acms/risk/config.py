"""Risk Configuration for ACMS."""

from dataclasses import dataclass


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_per_symbol: float = 100000.0
    max_total_position: float = 1000000.0
    max_order_notional: float = 50000.0
    max_order_quantity: float = 10.0
    max_daily_drawdown: float = 0.05
    max_weekly_drawdown: float = 0.10
    max_drawdown: float = 0.20
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 100
    max_net_exposure: float = 500000.0
    max_gross_exposure: float = 1000000.0
    max_concentration_pct: float = 0.25
    var_confidence: float = 0.99
    cvar_confidence: float = 0.99
    max_correlation: float = 0.85
    initial_margin_ratio: float = 0.10
    maintenance_margin_ratio: float = 0.05
    circuit_breaker_loss_pct: float = 0.03
    circuit_breaker_cooldown_minutes: int = 30
    # Risk budgeting
    risk_budget_per_strategy: float = 0.25
    max_strategy_risk_pct: float = 0.40


__all__ = ['RiskConfig']
