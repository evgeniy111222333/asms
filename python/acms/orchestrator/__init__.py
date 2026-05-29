"""Orchestrator - Central coordination of all ACMS components.

Manages the lifecycle of:
- Signal generators
- Strategy evaluators with allocation management
- Risk checks before every order submission
- Position sizing (Kelly, risk-based, fixed-fractional)
- Execution routing
- Portfolio reconciliation
- Kill switch propagation to all components
- Circuit breaker integration
- Graceful degradation modes
- Performance monitoring and auto-disable
- Real-time P&L tracking with equity curve
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
from collections import defaultdict

import numpy as np

from acms.core import Signal, Order, Position, Side, SignalDirection, ACMSConfig
from acms.signals import SignalEngine, SignalConfig
from acms.strategies import Strategy, create_strategy, STRATEGY_REGISTRY
from acms.risk import RiskEngine, RiskConfig
from acms.portfolio import PortfolioEngine
from acms.exchanges import ExchangeAdapter, create_exchange_adapter, PaperTradingAdapter
from acms.db import init_db

logger = logging.getLogger(__name__)


class OrchestratorState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    DEGRADED = "degraded"
    CIRCUIT_BREAKER = "circuit_breaker"


class DegradationLevel(str, Enum):
    NONE = "none"
    REDUCE_POSITIONS = "reduce_positions"
    WIDEN_STOPS = "widen_stops"
    HALT_NEW_ORDERS = "halt_new_orders"
    FULL_HALT = "full_halt"


# ============================================================================
# Position Sizing
# ============================================================================

class PositionSizer:
    """Position sizing with multiple methodologies.

    Supports:
    - Kelly Criterion (fractional)
    - Risk-based (volatility targeting)
    - Fixed-fractional
    - Fixed-quantity
    """

    def __init__(self, method: str = "risk_based", max_position_pct: float = 0.02,
                 kelly_fraction: float = 0.5, target_volatility: float = 0.15,
                 risk_per_trade_pct: float = 0.01):
        self.method = method
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        self.target_volatility = target_volatility
        self.risk_per_trade_pct = risk_per_trade_pct

    def compute_size(self, equity: float, price: float, volatility: float = 0.0,
                     win_rate: float = 0.5, avg_win_loss_ratio: float = 1.0,
                     stop_distance_pct: float = 0.02) -> float:
        """Compute position size based on the configured method.

        Args:
            equity: Current account equity.
            price: Current asset price.
            volatility: Annualized volatility estimate.
            win_rate: Historical win rate (for Kelly).
            avg_win_loss_ratio: Average win / average loss ratio.
            stop_distance_pct: Stop-loss distance as percentage.

        Returns:
            Position size in base currency units.
        """
        if equity <= 0 or price <= 0:
            return 0.0

        if self.method == "kelly":
            return self._kelly_size(equity, price, win_rate, avg_win_loss_ratio)
        elif self.method == "risk_based":
            return self._risk_based_size(equity, price, volatility, stop_distance_pct)
        elif self.method == "fixed_fractional":
            return self._fixed_fractional_size(equity, price)
        elif self.method == "volatility_target":
            return self._volatility_target_size(equity, price, volatility)
        else:
            return self._risk_based_size(equity, price, volatility, stop_distance_pct)

    def _kelly_size(self, equity: float, price: float,
                    win_rate: float, avg_win_loss_ratio: float) -> float:
        """Kelly Criterion position sizing (fractional).

        f* = (p * b - q) / b where p = win_rate, q = 1-p, b = avg_win/avg_loss.
        Uses fractional Kelly for safety.
        """
        if win_rate <= 0 or win_rate >= 1 or avg_win_loss_ratio <= 0:
            return 0.0

        kelly_pct = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio
        if kelly_pct <= 0:
            return 0.0

        # Apply fractional Kelly
        position_pct = kelly_pct * self.kelly_fraction
        position_pct = min(position_pct, self.max_position_pct)

        notional = equity * position_pct
        return notional / price

    def _risk_based_size(self, equity: float, price: float,
                         volatility: float, stop_distance_pct: float) -> float:
        """Risk-based position sizing.

        Sizes positions so that the maximum loss per trade equals
        a fixed percentage of equity.
        """
        if stop_distance_pct <= 0:
            stop_distance_pct = max(volatility * 0.25, 0.01) if volatility > 0 else 0.02

        risk_amount = equity * self.risk_per_trade_pct
        size = risk_amount / (price * stop_distance_pct)

        # Cap at max position percentage
        max_notional = equity * self.max_position_pct
        max_size = max_notional / price
        return min(size, max_size)

    def _fixed_fractional_size(self, equity: float, price: float) -> float:
        """Fixed-fractional position sizing."""
        notional = equity * self.max_position_pct
        return notional / price

    def _volatility_target_size(self, equity: float, price: float,
                                volatility: float) -> float:
        """Volatility-targeting position sizing.

        Sizes positions so that the portfolio's volatility contribution
        equals the target volatility.
        """
        if volatility <= 0:
            volatility = 0.20  # Default 20% annualized
        notional = equity * self.target_volatility / volatility
        max_notional = equity * self.max_position_pct
        notional = min(notional, max_notional)
        return notional / price


# ============================================================================
# Strategy Allocation Manager
# ============================================================================

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
        """Set custom allocation weights.

        Args:
            weights: Dict mapping strategy_id to weight (0.0-1.0).
        """
        total = sum(weights.values())
        if total > 0:
            self.custom_weights = {k: v / total for k, v in weights.items()}
        self.method = "custom"

    def get_allocation(self, strategy_ids: List[str], total_capital: float) -> Dict[str, float]:
        """Get capital allocation for each strategy.

        Args:
            strategy_ids: Active strategy identifiers.
            total_capital: Total capital to allocate.

        Returns:
            Dict mapping strategy_id to allocated capital.
        """
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

        # Inverse volatility weighting
        inv_vols = {sid: 1.0 / max(v, 1e-6) for sid, v in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        allocation = {}
        for sid in strategy_ids:
            weight = inv_vols[sid] / total_inv_vol if total_inv_vol > 0 else 1.0 / len(strategy_ids)
            allocation[sid] = total_capital * weight

        return allocation

    def update_performance(self, strategy_id: str, return_pct: float) -> None:
        """Update strategy performance for risk parity calculations.

        Args:
            strategy_id: Strategy identifier.
            return_pct: Period return as decimal.
        """
        self._strategy_returns[strategy_id].append(return_pct)
        # Keep last 252 returns (approximately 1 trading year)
        if len(self._strategy_returns[strategy_id]) > 252:
            self._strategy_returns[strategy_id] = self._strategy_returns[strategy_id][-252:]
        # Update volatility estimate
        returns = self._strategy_returns[strategy_id]
        if len(returns) > 10:
            self._strategy_volatilities[strategy_id] = float(np.std(returns)) * np.sqrt(252)


# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """Monitors strategy performance and auto-disables underperformers.

    Tracks rolling Sharpe ratio and disables strategies that
    consistently underperform.
    """

    def __init__(self, min_sharpe: float = -1.0, lookback_trades: int = 20,
                 auto_disable: bool = True):
        self.min_sharpe = min_sharpe
        self.lookback_trades = lookback_trades
        self.auto_disable = auto_disable
        self._strategy_pnls: Dict[str, List[float]] = defaultdict(list)
        self._disabled_strategies: set = set()

    def record_pnl(self, strategy_id: str, pnl: float) -> None:
        """Record P&L for a strategy.

        Args:
            strategy_id: Strategy identifier.
            pnl: Trade P&L amount.
        """
        self._strategy_pnls[strategy_id].append(pnl)
        # Trim to lookback window
        if len(self._strategy_pnls[strategy_id]) > self.lookback_trades * 2:
            self._strategy_pnls[strategy_id] = self._strategy_pnls[strategy_id][-self.lookback_trades:]

    def check_strategy(self, strategy_id: str) -> Dict:
        """Check if a strategy should be auto-disabled.

        Args:
            strategy_id: Strategy identifier.

        Returns:
            Dict with performance metrics and disable recommendation.
        """
        pnls = self._strategy_pnls.get(strategy_id, [])
        if len(pnls) < self.lookback_trades:
            return {"strategy_id": strategy_id, "should_disable": False,
                    "reason": "insufficient_data", "sharpe": 0.0}

        recent = pnls[-self.lookback_trades:]
        mean_pnl = np.mean(recent)
        std_pnl = np.std(recent)
        sharpe = mean_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0

        should_disable = self.auto_disable and sharpe < self.min_sharpe
        if should_disable:
            self._disabled_strategies.add(strategy_id)

        return {
            "strategy_id": strategy_id,
            "should_disable": should_disable,
            "sharpe": float(sharpe),
            "mean_pnl": float(mean_pnl),
            "std_pnl": float(std_pnl),
            "total_pnl": float(sum(recent)),
            "win_rate": float(sum(1 for p in recent if p > 0) / len(recent)),
        }

    def is_disabled(self, strategy_id: str) -> bool:
        """Check if a strategy has been auto-disabled."""
        return strategy_id in self._disabled_strategies

    def reenable(self, strategy_id: str) -> None:
        """Re-enable a previously disabled strategy."""
        self._disabled_strategies.discard(strategy_id)


# ============================================================================
# Equity Curve Tracker
# ============================================================================

class EquityCurveTracker:
    """Real-time P&L tracking with equity curve."""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.equity_history: List[Dict] = []
        self._current_equity = initial_capital

    def update(self, equity: float, timestamp: Optional[datetime] = None) -> None:
        """Record equity snapshot.

        Args:
            equity: Current portfolio equity.
            timestamp: Snapshot timestamp.
        """
        self._current_equity = equity
        self.equity_history.append({
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "equity": equity,
            "pnl": equity - self.initial_capital,
            "pnl_pct": (equity - self.initial_capital) / self.initial_capital,
        })

    @property
    def current_equity(self) -> float:
        return self._current_equity

    @property
    def current_pnl(self) -> float:
        return self._current_equity - self.initial_capital

    @property
    def current_pnl_pct(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return (self._current_equity - self.initial_capital) / self.initial_capital

    def get_equity_array(self) -> np.ndarray:
        """Get equity values as numpy array."""
        return np.array([e["equity"] for e in self.equity_history])

    def get_max_drawdown(self) -> float:
        """Compute maximum drawdown from equity curve."""
        if len(self.equity_history) < 2:
            return 0.0
        equity = self.get_equity_array()
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        return float(np.max(drawdown))


# ============================================================================
# Orchestrator Configuration and Implementation
# ============================================================================

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


class Orchestrator:
    """Central orchestrator for the ACMS trading system.

    Coordinates all components: signal generation, risk management,
    position sizing, order execution, and performance monitoring.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None,
                 acms_config: Optional[ACMSConfig] = None):
        self.config = config or OrchestratorConfig()
        self.acms_config = acms_config or ACMSConfig()
        self.state = OrchestratorState.STOPPED
        self.degradation_level = DegradationLevel.NONE

        # Components
        self.signal_engine = SignalEngine(self.config.signal_config)
        self.risk_engine = RiskEngine(self.config.risk_config)
        self.portfolio_engine = PortfolioEngine()
        self.exchange: Optional[ExchangeAdapter] = None
        self.position_sizer = PositionSizer(
            method=self.config.sizing_method,
            max_position_pct=self.config.max_position_pct,
        )
        self.allocation_manager = StrategyAllocationManager(
            method=self.config.allocation_method,
        )
        self.performance_monitor = PerformanceMonitor(
            auto_disable=self.config.auto_disable_underperformers,
            min_sharpe=self.config.min_sharpe_threshold,
        )
        self.equity_tracker = EquityCurveTracker()

        # Active strategies
        self.strategies: Dict[str, Strategy] = {}

        # State
        self._task: Optional[asyncio.Task] = None
        self._candles: list = []
        self._positions: Dict[str, Position] = {}
        self._orders: List[Order] = []
        self._signals: List[Signal] = []
        self._last_equity: float = 100000.0

    async def start(self) -> None:
        """Start the orchestrator and all components."""
        if self.state == OrchestratorState.RUNNING:
            return

        self.state = OrchestratorState.STARTING

        try:
            # Initialize exchange
            self.exchange = create_exchange_adapter(self.config.exchange)

            # Initialize strategies
            if self.config.strategy_type not in self.strategies:
                strategy = create_strategy(
                    self.config.strategy_type, self.config.symbol
                )
                self.strategies[self.config.strategy_type] = strategy

            # Start main loop
            self._task = asyncio.create_task(self._main_loop())
            self.state = OrchestratorState.RUNNING
            logger.info("Orchestrator started with strategy '%s'", self.config.strategy_type)

        except Exception as e:
            self.state = OrchestratorState.ERROR
            logger.error("Orchestrator start failed: %s", e)
            raise

    async def stop(self) -> None:
        """Stop the orchestrator gracefully."""
        self.state = OrchestratorState.STOPPING
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Close exchange connections
        if self.exchange:
            await self.exchange.close()
        self.state = OrchestratorState.STOPPED
        logger.info("Orchestrator stopped")

    async def pause(self) -> None:
        """Pause trading (keep data flowing)."""
        self.state = OrchestratorState.PAUSED
        logger.info("Orchestrator paused")

    async def resume(self) -> None:
        """Resume trading from paused state."""
        if self.state == OrchestratorState.PAUSED:
            self.state = OrchestratorState.RUNNING
            logger.info("Orchestrator resumed")

    async def _main_loop(self) -> None:
        """Main trading loop."""
        while self.state in (OrchestratorState.RUNNING, OrchestratorState.PAUSED,
                             OrchestratorState.DEGRADED, OrchestratorState.CIRCUIT_BREAKER):
            try:
                if self.state == OrchestratorState.RUNNING or self.state == OrchestratorState.DEGRADED:
                    await self._trading_cycle()
                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Trading cycle error: %s", e)
                await asyncio.sleep(5)

    async def _trading_cycle(self) -> None:
        """Execute one trading cycle."""
        # 1. Fetch market data
        await self._fetch_market_data()

        # 2. Check circuit breakers
        if self._check_circuit_breakers():
            return

        # 3. Update equity
        await self._update_equity()

        # 4. Evaluate strategies
        for strategy_id, strategy in list(self.strategies.items()):
            if not strategy.is_active:
                continue

            # Check if auto-disabled
            if self.performance_monitor.is_disabled(strategy_id):
                logger.warning("Strategy '%s' auto-disabled due to poor performance", strategy_id)
                continue

            # Check degradation level
            if self.degradation_level == DegradationLevel.HALT_NEW_ORDERS:
                continue

            signal = strategy.evaluate(self._candles)
            if signal and signal.direction != SignalDirection.NEUTRAL:
                self._signals.append(signal)

                # 3. Risk check BEFORE order creation
                risk_result = self._check_risk(signal)
                if not risk_result:
                    continue

                # 4. Create order with proper sizing
                order = self._signal_to_order(signal)
                if order is None:
                    continue

                # 5. Execute
                await self._execute_order(order)

        # 5. Check performance
        self._check_performance()

    async def _fetch_market_data(self) -> None:
        """Fetch latest market data from exchange."""
        if not self.exchange:
            return
        try:
            candles = await self.exchange.get_candles(
                self.config.symbol, self.config.timeframe, limit=200
            )
            self._candles = candles
        except Exception as e:
            logger.warning("Failed to fetch market data: %s", e)

    async def _update_equity(self) -> None:
        """Update equity curve tracker."""
        try:
            if self.exchange:
                balance = await self.exchange.get_balance()
                total = sum(v.get("free", 0) + v.get("locked", 0) for v in balance.values())
                if total > 0:
                    self._last_equity = total
        except Exception:
            pass
        self.equity_tracker.update(self._last_equity)

    def _check_risk(self, signal: Signal) -> bool:
        """Perform risk checks before order submission.

        Args:
            signal: Trading signal to validate.

        Returns:
            True if signal passes all risk checks.
        """
        if self.risk_engine.kill_switch_active:
            logger.warning("Kill switch active - rejecting signal")
            return False

        # Check drawdown
        max_dd = self.equity_tracker.get_max_drawdown()
        if max_dd > self.config.risk_config.max_drawdown:
            logger.warning("Max drawdown exceeded: %.2f%% > %.2f%%",
                           max_dd * 100, self.config.risk_config.max_drawdown * 100)
            self._activate_circuit_breaker("max_drawdown_exceeded")
            return False

        # Check daily loss
        current_pnl_pct = self.equity_tracker.current_pnl_pct
        if current_pnl_pct < -self.config.risk_config.max_daily_drawdown:
            logger.warning("Daily loss limit exceeded: %.2f%%", current_pnl_pct * 100)
            self._activate_circuit_breaker("daily_loss_exceeded")
            return False

        return True

    def _check_circuit_breakers(self) -> bool:
        """Check if circuit breaker is active.

        Returns:
            True if trading should be halted.
        """
        if self.state == OrchestratorState.CIRCUIT_BREAKER:
            return True
        return False

    def _activate_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker, pausing all trading.

        Args:
            reason: Reason for circuit breaker activation.
        """
        self.state = OrchestratorState.CIRCUIT_BREAKER
        logger.critical("Circuit breaker activated: %s", reason)
        # Apply degradation
        if self.config.degradation_enabled:
            self._apply_degradation(DegradationLevel.HALT_NEW_ORDERS)

    def _apply_degradation(self, level: DegradationLevel) -> None:
        """Apply graceful degradation mode.

        Args:
            level: Degradation level to apply.
        """
        self.degradation_level = level
        if level == DegradationLevel.REDUCE_POSITIONS:
            logger.warning("Degradation: REDUCE_POSITIONS - reducing position sizes by 50%")
        elif level == DegradationLevel.WIDEN_STOPS:
            logger.warning("Degradation: WIDEN_STOPS - widening stop-loss distances")
        elif level == DegradationLevel.HALT_NEW_ORDERS:
            logger.warning("Degradation: HALT_NEW_ORDERS - no new orders allowed")
        elif level == DegradationLevel.FULL_HALT:
            logger.warning("Degradation: FULL_HALT - all trading halted")
            self.state = OrchestratorState.PAUSED

    def _signal_to_order(self, signal: Signal) -> Optional[Order]:
        """Convert a signal to an order with proper position sizing.

        Args:
            signal: Trading signal to convert.

        Returns:
            Order instance or None if sizing fails.
        """
        side = Side.BUY if signal.direction == SignalDirection.LONG else Side.SELL

        # Get current price
        current_price = 0.0
        if self._candles:
            current_price = self._candles[-1].close if hasattr(self._candles[-1], 'close') else 0.0

        if current_price <= 0:
            logger.warning("Cannot size position: no valid price")
            return None

        # Compute volatility from recent candles
        volatility = 0.0
        if len(self._candles) > 20:
            closes = [c.close for c in self._candles[-20:] if hasattr(c, 'close')]
            if len(closes) > 5:
                returns = np.diff(closes) / closes[:-1]
                volatility = float(np.std(returns) * np.sqrt(365 * 24 * 60))

        # Get allocated capital for this strategy
        allocations = self.allocation_manager.get_allocation(
            list(self.strategies.keys()), self.equity_tracker.current_equity
        )
        strategy_capital = allocations.get(signal.strategy_id, self.equity_tracker.current_equity)

        # Compute position size
        quantity = self.position_sizer.compute_size(
            equity=strategy_capital,
            price=current_price,
            volatility=volatility,
            stop_distance_pct=0.02,
        )

        # Apply degradation adjustments
        if self.degradation_level == DegradationLevel.REDUCE_POSITIONS:
            quantity *= 0.5

        if quantity <= 0:
            return None

        # Apply signal strength scaling
        quantity *= signal.strength

        # Round to reasonable precision
        quantity = round(quantity, 6)

        order = Order(
            id=f"ord_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            symbol=signal.symbol, side=side, order_type=OrderType.MARKET,
            status=OrderStatus.CREATED, quantity=quantity,
            exchange=self.config.exchange, strategy_id=signal.strategy_id,
        )
        return order

    async def _execute_order(self, order: Order) -> None:
        """Execute an order through the exchange adapter.

        Args:
            order: Order to execute.
        """
        if not self.exchange:
            return

        try:
            result = await self.exchange.place_order(order)
            self._orders.append(result)
            logger.info("Order executed: %s %s %.6f %s",
                        result.side.value, result.symbol, result.quantity, result.id)
        except Exception as e:
            logger.error("Order execution failed: %s", e)

    def _check_performance(self) -> None:
        """Check strategy performance and auto-disable underperformers."""
        for strategy_id in list(self.strategies.keys()):
            result = self.performance_monitor.check_strategy(strategy_id)
            if result.get("should_disable") and self.strategies.get(strategy_id):
                logger.warning("Auto-disabling strategy '%s': Sharpe=%.2f",
                               strategy_id, result.get("sharpe", 0))
                self.strategies[strategy_id].is_active = False

    def add_strategy(self, strategy_type: str, symbol: str, **kwargs) -> str:
        """Add a new strategy to the orchestrator.

        Args:
            strategy_type: Type of strategy from STRATEGY_REGISTRY.
            symbol: Trading pair symbol.
            **kwargs: Additional strategy parameters.

        Returns:
            Strategy identifier.
        """
        if len(self.strategies) >= self.config.max_concurrent_strategies:
            raise ValueError(f"Maximum concurrent strategies ({self.config.max_concurrent_strategies}) reached")

        strategy = create_strategy(strategy_type, symbol, **kwargs)
        self.strategies[strategy_type] = strategy
        logger.info("Strategy '%s' added for %s", strategy_type, symbol)
        return strategy_type

    def remove_strategy(self, strategy_type: str) -> None:
        """Remove a strategy from the orchestrator.

        Args:
            strategy_type: Strategy identifier to remove.
        """
        if strategy_type in self.strategies:
            self.strategies[strategy_type].is_active = False
            del self.strategies[strategy_type]
            logger.info("Strategy '%s' removed", strategy_type)

    def trigger_kill_switch(self, reason: str = "Manual") -> None:
        """Trigger emergency kill switch.

        Propagates to all components and halts all trading.

        Args:
            reason: Reason for kill switch activation.
        """
        self.risk_engine.trigger_kill_switch(reason)
        self.state = OrchestratorState.PAUSED
        self._apply_degradation(DegradationLevel.FULL_HALT)
        logger.critical("Kill switch triggered: %s", reason)

    def reset_kill_switch(self) -> None:
        """Reset kill switch and resume operations."""
        self.risk_engine.reset_kill_switch()
        self.degradation_level = DegradationLevel.NONE
        if self.state in (OrchestratorState.PAUSED, OrchestratorState.CIRCUIT_BREAKER):
            self.state = OrchestratorState.RUNNING
        logger.info("Kill switch reset, resuming operations")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status.

        Returns:
            Dict with current state, strategy info, and performance metrics.
        """
        return {
            "state": self.state.value,
            "degradation_level": self.degradation_level.value,
            "active_strategies": list(self.strategies.keys()),
            "total_signals": len(self._signals),
            "total_orders": len(self._orders),
            "kill_switch": self.risk_engine.kill_switch_active,
            "exchange": self.config.exchange,
            "symbol": self.config.symbol,
            "current_equity": self.equity_tracker.current_equity,
            "current_pnl": self.equity_tracker.current_pnl,
            "current_pnl_pct": self.equity_tracker.current_pnl_pct,
            "max_drawdown": self.equity_tracker.get_max_drawdown(),
            "position_sizing_method": self.position_sizer.method,
            "allocation_method": self.allocation_manager.method,
            "disabled_strategies": list(self.performance_monitor._disabled_strategies),
        }
