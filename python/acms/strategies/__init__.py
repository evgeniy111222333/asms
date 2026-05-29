"""Strategy Engine - Trading strategy implementations.

Implements 12 major strategy categories with dynamic parameter
adaptation based on regime detection.
"""

from acms.strategies.base import Strategy
from acms.strategies.momentum import (
    TrendFollowingMomentum,
    BreakoutMomentum,
    RSIMomentum,
    MACDMomentum,
    SupertrendMomentum,
)
from acms.strategies.mean_reversion import MeanReversionStrategy
from acms.strategies.stat_arbitrage import StatisticalArbitrageStrategy
from acms.strategies.grid import GridTradingStrategy
from acms.strategies.turtle import TurtleTradingStrategy
from acms.strategies.wyckoff import WyckoffStrategy
from acms.strategies.carry import CarryStrategy
from acms.strategies.volatility import VolatilityStrategy
from acms.strategies.market_making import MarketMakingStrategy
from acms.strategies.cross_exchange_arb import CrossExchangeArbitrageStrategy

# Strategy Registry & Factory
STRATEGY_REGISTRY: dict[str, type] = {
    "trend_following": TrendFollowingMomentum,
    "breakout": BreakoutMomentum,
    "rsi_momentum": RSIMomentum,
    "macd_momentum": MACDMomentum,
    "supertrend": SupertrendMomentum,
    "mean_reversion": MeanReversionStrategy,
    "statistical_arbitrage": StatisticalArbitrageStrategy,
    "grid_trading": GridTradingStrategy,
    "turtle": TurtleTradingStrategy,
    "wyckoff": WyckoffStrategy,
    "carry": CarryStrategy,
    "volatility": VolatilityStrategy,
    "market_making": MarketMakingStrategy,
    "cross_exchange_arbitrage": CrossExchangeArbitrageStrategy,
}


def create_strategy(name: str, **kwargs) -> Strategy:
    """Create a strategy instance by name.

    Args:
        name: Strategy name from STRATEGY_REGISTRY.
        **kwargs: Arguments passed to the strategy constructor.

    Returns:
        Strategy instance.

    Raises:
        ValueError: If strategy name is not found in registry.
    """
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return cls(**kwargs)


__all__ = [
    # Base
    "Strategy",
    # Momentum
    "TrendFollowingMomentum", "BreakoutMomentum", "RSIMomentum",
    "MACDMomentum", "SupertrendMomentum",
    # Mean Reversion
    "MeanReversionStrategy",
    # Statistical Arbitrage
    "StatisticalArbitrageStrategy",
    # Grid
    "GridTradingStrategy",
    # Turtle
    "TurtleTradingStrategy",
    # Wyckoff
    "WyckoffStrategy",
    # Carry
    "CarryStrategy",
    # Volatility
    "VolatilityStrategy",
    # Market Making
    "MarketMakingStrategy",
    # Cross-Exchange Arbitrage
    "CrossExchangeArbitrageStrategy",
    # Registry & Factory
    "STRATEGY_REGISTRY", "create_strategy",
]
