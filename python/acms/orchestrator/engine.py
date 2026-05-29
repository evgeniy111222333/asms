"""Main orchestrator engine."""

import asyncio
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

import numpy as np

from acms.core import Signal, Order, Position, Side, SignalDirection, OrderType, OrderStatus, ACMSConfig
from acms.signals import SignalEngine, SignalConfig
from acms.strategies import Strategy, create_strategy, STRATEGY_REGISTRY
from acms.risk import RiskEngine, RiskConfig
from acms.portfolio import PortfolioEngine
from acms.exchanges import ExchangeAdapter, create_exchange_adapter, PaperTradingAdapter
from acms.db import init_db
from acms.orchestrator.state import OrchestratorState, DegradationLevel
from acms.orchestrator.position_sizer import PositionSizer
from acms.orchestrator.allocation import StrategyAllocationManager
from acms.orchestrator.performance import PerformanceMonitor, EquityCurveTracker
from acms.orchestrator.config import OrchestratorConfig

logger = logging.getLogger(__name__)


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
            self.exchange = create_exchange_adapter(self.config.exchange)

            if self.config.strategy_type not in self.strategies:
                strategy = create_strategy(
                    self.config.strategy_type, self.config.symbol
                )
                self.strategies[self.config.strategy_type] = strategy

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
                logger.debug("Orchestrator task cancelled during stop")
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
        await self._fetch_market_data()

        if self._check_circuit_breakers():
            return

        await self._update_equity()

        for strategy_id, strategy in list(self.strategies.items()):
            if not strategy.is_active:
                continue

            if self.performance_monitor.is_disabled(strategy_id):
                logger.warning("Strategy '%s' auto-disabled due to poor performance", strategy_id)
                continue

            if self.degradation_level == DegradationLevel.HALT_NEW_ORDERS:
                continue

            signal = strategy.evaluate(self._candles)
            if signal and signal.direction != SignalDirection.NEUTRAL:
                self._signals.append(signal)

                risk_result = self._check_risk(signal)
                if not risk_result:
                    continue

                order = self._signal_to_order(signal)
                if order is None:
                    continue

                await self._execute_order(order)

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
        except Exception as e:
            logger.warning("Failed to update equity from exchange: %s", e)
        self.equity_tracker.update(self._last_equity)

    def _check_risk(self, signal: Signal) -> bool:
        """Perform risk checks before order submission."""
        if self.risk_engine.kill_switch_active:
            logger.warning("Kill switch active - rejecting signal")
            return False

        max_dd = self.equity_tracker.get_max_drawdown()
        if max_dd > self.config.risk_config.max_drawdown:
            logger.warning("Max drawdown exceeded: %.2f%% > %.2f%%",
                           max_dd * 100, self.config.risk_config.max_drawdown * 100)
            self._activate_circuit_breaker("max_drawdown_exceeded")
            return False

        current_pnl_pct = self.equity_tracker.current_pnl_pct
        if current_pnl_pct < -self.config.risk_config.max_daily_drawdown:
            logger.warning("Daily loss limit exceeded: %.2f%%", current_pnl_pct * 100)
            self._activate_circuit_breaker("daily_loss_exceeded")
            return False

        return True

    def _check_circuit_breakers(self) -> bool:
        """Check if circuit breaker is active."""
        if self.state == OrchestratorState.CIRCUIT_BREAKER:
            return True
        return False

    def _activate_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker, pausing all trading."""
        self.state = OrchestratorState.CIRCUIT_BREAKER
        logger.critical("Circuit breaker activated: %s", reason)
        if self.config.degradation_enabled:
            self._apply_degradation(DegradationLevel.HALT_NEW_ORDERS)

    def _apply_degradation(self, level: DegradationLevel) -> None:
        """Apply graceful degradation mode."""
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
        """Convert a signal to an order with proper position sizing."""
        side = Side.BUY if signal.direction == SignalDirection.LONG else Side.SELL

        current_price = 0.0
        if self._candles:
            current_price = self._candles[-1].close if hasattr(self._candles[-1], 'close') else 0.0

        if current_price <= 0:
            logger.warning("Cannot size position: no valid price")
            return None

        volatility = 0.0
        if len(self._candles) > 20:
            closes = [c.close for c in self._candles[-20:] if hasattr(c, 'close')]
            if len(closes) > 5:
                returns = np.diff(closes) / closes[:-1]
                volatility = float(np.std(returns) * np.sqrt(365 * 24 * 60))

        allocations = self.allocation_manager.get_allocation(
            list(self.strategies.keys()), self.equity_tracker.current_equity
        )
        strategy_capital = allocations.get(signal.strategy_id, self.equity_tracker.current_equity)

        quantity = self.position_sizer.compute_size(
            equity=strategy_capital,
            price=current_price,
            volatility=volatility,
            stop_distance_pct=0.02,
        )

        if self.degradation_level == DegradationLevel.REDUCE_POSITIONS:
            quantity *= 0.5

        if quantity <= 0:
            return None

        quantity *= signal.strength
        quantity = round(quantity, 6)

        order = Order(
            id=f"ord_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            symbol=signal.symbol, side=side, order_type=OrderType.MARKET,
            status=OrderStatus.CREATED, quantity=quantity,
            exchange=self.config.exchange, strategy_id=signal.strategy_id,
        )
        return order

    async def _execute_order(self, order: Order) -> None:
        """Execute an order through the exchange adapter."""
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
        """Add a new strategy to the orchestrator."""
        if len(self.strategies) >= self.config.max_concurrent_strategies:
            raise ValueError(f"Maximum concurrent strategies ({self.config.max_concurrent_strategies}) reached")

        strategy = create_strategy(strategy_type, symbol, **kwargs)
        self.strategies[strategy_type] = strategy
        logger.info("Strategy '%s' added for %s", strategy_type, symbol)
        return strategy_type

    def remove_strategy(self, strategy_type: str) -> None:
        """Remove a strategy from the orchestrator."""
        if strategy_type in self.strategies:
            self.strategies[strategy_type].is_active = False
            del self.strategies[strategy_type]
            logger.info("Strategy '%s' removed", strategy_type)

    def trigger_kill_switch(self, reason: str = "Manual") -> None:
        """Trigger emergency kill switch."""
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
        """Get comprehensive orchestrator status."""
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


__all__ = [
    "Orchestrator",
]
