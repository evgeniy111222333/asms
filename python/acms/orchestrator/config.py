"""Orchestrator configuration."""

from dataclasses import dataclass, field

from acms.risk import RiskConfig
from acms.signals import SignalConfig


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    symbol: str = "BTC/USDT"
    timeframe: str = "1m"
    strategy_type: str = "momentum_trend"
    exchange: str = "paper"
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    signal_config: SignalConfig = field(default_factory=SignalConfig)
    check_interval_seconds: float = 1.0
    max_concurrent_strategies: int = 5
    sizing_method: str = "risk_based"
    max_position_pct: float = 0.02
    allocation_method: str = "equal_weight"
    auto_disable_underperformers: bool = True
    min_sharpe_threshold: float = -1.0
    degradation_enabled: bool = True


__all__ = [
    "OrchestratorConfig",
]
