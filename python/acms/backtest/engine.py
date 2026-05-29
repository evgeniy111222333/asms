"""Backtest engine - single, walk-forward, and Monte Carlo."""

import logging

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from acms.core import Candle, Signal, SignalDirection, Position, Side, Trade
from acms.strategies import Strategy
from acms.risk import RiskEngine, RiskConfig
from acms.indicators import ATR
from acms.backtest.config import BacktestConfig, BacktestMode, BacktestResult, BacktestTrade, MCStatistics
from acms.backtest.slippage import SlippageModel
from acms.backtest.fill_model import FillModel
from acms.backtest.analytics import TradeAnalytics, RollingMetrics
from acms.backtest.benchmark import BenchmarkComparison, RegimeDetector, SensitivityAnalysis

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Full-featured backtesting engine.

    Supports single-pass, walk-forward, and Monte Carlo backtesting
    with realistic execution simulation, multiple slippage models,
    and comprehensive performance analytics.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.risk_engine = RiskEngine(RiskConfig())
        self.slippage = SlippageModel()
        self.trade_analytics = TradeAnalytics()
        self.rolling_metrics = RollingMetrics()
        self.benchmark = BenchmarkComparison()
        self.regime_detector = RegimeDetector(
            lookback=self.config.regime_lookback,
        )
        self.sensitivity = SensitivityAnalysis()
        self.fill_model = FillModel()

    def run(self, candles: List[Candle], strategy: Strategy,
            mode: BacktestMode = BacktestMode.SINGLE,
            multi_asset_candles: Optional[Dict[str, List[Candle]]] = None) -> BacktestResult:
        """Run a backtest.

        Args:
            candles: Price candles for primary symbol.
            strategy: Strategy to backtest.
            mode: Backtest execution mode.
            multi_asset_candles: Optional dict of symbol -> candles for multi-asset.

        Returns:
            BacktestResult with complete performance metrics.

        Raises:
            ValueError: If mode is unknown.
        """
        if mode == BacktestMode.SINGLE:
            return self._run_single(candles, strategy, multi_asset_candles)
        elif mode == BacktestMode.WALK_FORWARD:
            return self._run_walk_forward(candles, strategy)
        elif mode == BacktestMode.MONTE_CARLO:
            return self._run_monte_carlo(candles, strategy)
        else:
            raise ValueError(f"Unknown backtest mode: {mode}")

    def run_sensitivity(self, candles: List[Candle], strategy: Strategy,
                        params: Optional[Dict[str, List[float]]] = None) -> Dict:
        """Run parametric sensitivity analysis.

        Args:
            candles: Price candles.
            strategy: Strategy to test.
            params: Parameters to vary. Defaults to position_size_pct and slippage_bps.

        Returns:
            Dict with sensitivity analysis results.
        """
        if params is None:
            params = {
                "position_size_pct": [0.01, 0.02, 0.03, 0.05],
                "slippage_bps": [0, 5, 10, 20, 50],
            }
        return self.sensitivity.run(self, candles, strategy, params)

    def _apply_slippage(self, price: float, quantity: float, side: Side) -> float:
        """Apply configured slippage model to fill price.

        Args:
            price: Market price.
            quantity: Order quantity.
            side: Order side.

        Returns:
            Fill price after slippage.
        """
        if self.config.slippage_model == "sqrt":
            return self.slippage.square_root(price, quantity, 10000.0,
                                             self.config.slippage_bps, side)
        elif self.config.slippage_model == "almgren_chriss":
            return self.slippage.almgren_chriss(price, quantity, 10000.0,
                                                sigma=0.02, side=side)
        else:
            return self.slippage.percentage(price, quantity, self.config.slippage_bps, side)

    def _simulate_fill(self, quantity: float, price: float, side: Side) -> Dict:
        """Simulate order fill using configured fill model.

        Args:
            quantity: Order quantity.
            price: Fill price.
            side: Order side.

        Returns:
            Dict with fill details.
        """
        if self.config.fill_model == "partial":
            return self.fill_model.partial_fill(
                quantity, price, fill_pct=self.config.partial_fill_pct,
            )
        elif self.config.fill_model == "fok":
            return self.fill_model.fill_or_kill(quantity, price)
        else:
            return self.fill_model.immediate_fill(quantity, price)

    def _run_single(self, candles: List[Candle], strategy: Strategy,
                    multi_asset_candles: Optional[Dict[str, List[Candle]]] = None,
                    start_idx: int = 50, end_idx: Optional[int] = None,
                    initial_cap: Optional[float] = None) -> BacktestResult:
        """Run a single-pass backtest.

        Args:
            candles: Price candles.
            strategy: Strategy to backtest.
            multi_asset_candles: Optional multi-asset data for benchmarks.
            start_idx: Index of candle to start evaluating from.
            end_idx: Index of candle to end evaluation (exclusive).
            initial_cap: Custom initial capital to start with.

        Returns:
            BacktestResult with performance metrics.
        """
        capital = initial_cap if initial_cap is not None else self.config.initial_capital
        equity = [capital]
        closed_trades: List[BacktestTrade] = []
        active_positions: Dict[str, dict] = {}

        # Detect regimes if configured
        closes = np.array([c.close for c in candles])
        regimes = np.zeros(len(candles), dtype=int)
        if self.config.detect_regimes and len(candles) > self.config.regime_lookback:
            regimes = self.regime_detector.detect_regimes(closes)

        actual_end_idx = end_idx if end_idx is not None else len(candles)

        for i in range(start_idx, actual_end_idx):
            candle = candles[i]

            # Update existing positions
            for sym, pos_info in list(active_positions.items()):
                pos = pos_info["position"]
                pos.mark_price = candle.close
                pos.unrealized_pnl = (candle.close - pos.entry_price) * pos.quantity * (
                    1 if pos.side == Side.BUY else -1
                )

                pos_info["highs"].append(candle.high)
                pos_info["lows"].append(candle.low)

                if strategy.should_exit(candles[:i + 1], pos):
                    if pos.side == Side.BUY:
                        exit_fill = self._apply_slippage(candle.close, pos.quantity, Side.SELL)
                    else:
                        exit_fill = self._apply_slippage(candle.close, pos.quantity, Side.BUY)

                    fill_result = self._simulate_fill(pos.quantity, exit_fill, pos.side)
                    filled_qty = fill_result["filled_quantity"]

                    if filled_qty > 0:
                        commission = filled_qty * exit_fill * self.config.commission_bps / 10000
                        slippage_cost = abs(exit_fill - candle.close) * filled_qty

                        if pos.side == Side.BUY:
                            pnl = (exit_fill - pos.entry_price) * filled_qty - commission - slippage_cost
                        else:
                            pnl = (pos.entry_price - exit_fill) * filled_qty - commission - slippage_cost

                        analytics = self.trade_analytics.compute_mae_mfe(
                            pos.entry_price, exit_fill, pos.side,
                            np.array(pos_info["highs"]), np.array(pos_info["lows"]),
                            filled_qty,
                        )

                        regime_label = "low_vol" if regimes[i] == 0 else "normal" if regimes[i] == 1 else "crisis"

                        closed_trades.append(BacktestTrade(
                            entry_time=pos_info.get("entry_time", candle.open_time),
                            exit_time=candle.open_time,
                            symbol=sym, side=pos.side,
                            entry_price=pos.entry_price, exit_price=exit_fill,
                            quantity=filled_qty, pnl=pnl,
                            pnl_pct=pnl / (pos.entry_price * filled_qty) if pos.entry_price * filled_qty > 0 else 0,
                            commission=commission, slippage=slippage_cost,
                            holding_period_bars=i - pos_info.get("entry_bar", i),
                            strategy_id=strategy.strategy_id,
                            regime=regime_label,
                            mae=analytics["mae"], mfe=analytics["mfe"], etd=analytics["etd"],
                        ))

                        capital += pnl

                        # Handle partial fills - keep unfilled portion
                        if fill_result.get("partial", False) and fill_result.get("unfilled_quantity", 0) > 0:
                            pos.quantity = fill_result["unfilled_quantity"]
                        else:
                            del active_positions[sym]
                    elif fill_result.get("unfilled_quantity", 0) == pos.quantity:
                        # FOK rejected - keep position open
                        pass

            # Check for new signals
            if len(active_positions) < self.config.max_positions:
                signal = strategy.evaluate(candles[:i + 1])
                if signal and signal.direction != SignalDirection.NEUTRAL:
                    side = Side.BUY if signal.direction == SignalDirection.LONG else Side.SELL
                    risk_amount = capital * self.config.position_size_pct
                    atr_val = ATR(14).compute(
                        np.array([c.high for c in candles[:i + 1]]),
                        np.array([c.low for c in candles[:i + 1]]),
                        np.array([c.close for c in candles[:i + 1]]),
                    )
                    stop_distance = atr_val * 2 if not np.isnan(atr_val) else candle.close * 0.02
                    quantity = risk_amount / stop_distance if stop_distance > 0 else 0

                    if quantity > 0 and capital > 0:
                        entry_fill = self._apply_slippage(candle.close, quantity, side)
                        fill_result = self._simulate_fill(quantity, entry_fill, side)
                        filled_qty = fill_result["filled_quantity"]

                        if filled_qty > 0:
                            pos = Position(
                                symbol=signal.symbol, side=side, quantity=filled_qty,
                                entry_price=entry_fill, mark_price=candle.close,
                                unrealized_pnl=0.0, realized_pnl=0.0,
                                exchange="backtest",
                            )
                            active_positions[signal.symbol] = {
                                "position": pos,
                                "entry_time": candle.open_time,
                                "entry_bar": i,
                                "highs": [candle.high],
                                "lows": [candle.low],
                            }

            unrealized = sum(p["position"].unrealized_pnl for p in active_positions.values())
            equity.append(capital + unrealized)

        equity = np.array(equity)

        # Benchmark comparison based on evaluated candles only
        evaluated_candles = candles[start_idx:actual_end_idx]
        benchmark_data = self.benchmark.compute_benchmarks(evaluated_candles, multi_asset_candles)
        benchmark_return = benchmark_data["buy_and_hold_return"]

        # Compute dynamic annualization factor
        annualization = 525600.0
        if len(candles) >= 2:
            try:
                dt_seconds = (candles[1].open_time - candles[0].open_time).total_seconds()
                if dt_seconds > 0.0:
                    annualization = (365.0 * 24.0 * 3600.0) / dt_seconds
            except Exception as e:
                logger.debug("Could not compute dynamic annualization factor: %s", e)

        # Compute rolling metrics with dynamic annualization
        rolling_sharpe = self.rolling_metrics.rolling_sharpe(equity, annualization_factor=annualization)
        rolling_sortino = self.rolling_metrics.rolling_sortino(equity, annualization_factor=annualization)
        rolling_dd = self.rolling_metrics.rolling_max_drawdown(equity)

        result = self._compute_results(closed_trades, equity, benchmark_return=benchmark_return, annualization=annualization)
        result.buy_hold_return = benchmark_data["buy_and_hold_return"]
        result.equal_weight_return = benchmark_data["equal_weight_return"]
        result.rolling_sharpe = rolling_sharpe
        result.rolling_sortino = rolling_sortino
        result.rolling_max_dd = rolling_dd

        if self.config.detect_regimes:
            result.regime_labels = regimes[start_idx:actual_end_idx]

        return result

    def _run_walk_forward(self, candles: List[Candle], strategy: Strategy) -> BacktestResult:
        """Walk-forward analysis.

        Splits data into train/test windows and runs sequential
        backtests to test out-of-sample performance.

        Args:
            candles: Price candles.
            strategy: Strategy to backtest.

        Returns:
            BacktestResult with walk-forward performance.
        """
        n = len(candles)
        train_size = int(n * self.config.wf_train_pct)
        test_size = int(n * self.config.wf_test_pct)

        all_trades: List[BacktestTrade] = []
        current_capital = self.config.initial_capital
        all_equity = [current_capital]

        # Compute dynamic annualization factor
        annualization = 525600.0
        if len(candles) >= 2:
            try:
                dt_seconds = (candles[1].open_time - candles[0].open_time).total_seconds()
                if dt_seconds > 0.0:
                    annualization = (365.0 * 24.0 * 3600.0) / dt_seconds
            except Exception as e:
                logger.debug("Could not compute dynamic annualization factor: %s", e)

        start = 50
        while start + train_size + test_size <= n:
            start_idx = start + train_size
            end_idx = start_idx + test_size
            result = self._run_single(
                candles,
                strategy,
                start_idx=start_idx,
                end_idx=end_idx,
                initial_cap=current_capital
            )
            all_trades.extend(result.trades)
            if len(result.equity_curve) > 1:
                all_equity.extend(result.equity_curve[1:].tolist())
                current_capital = float(result.equity_curve[-1])
            start += test_size

        equity = np.array(all_equity)
        return self._compute_results(all_trades, equity, annualization=annualization)

    def _run_monte_carlo(self, candles: List[Candle], strategy: Strategy) -> BacktestResult:
        """Monte Carlo resampling backtest with proper statistics.

        FIXED: Monte Carlo now computes and stores all simulation
        results (returns, drawdowns, Sharpes) instead of discarding them.

        Args:
            candles: Price candles.
            strategy: Strategy to backtest.

        Returns:
            BacktestResult with MC statistics properly computed and attached.
        """
        base_result = self._run_single(candles, strategy)

        if not base_result.trades:
            return base_result

        trade_pnls = np.array([t.pnl_pct for t in base_result.trades])
        simulated_returns = np.zeros(self.config.mc_simulations)
        simulated_drawdowns = np.zeros(self.config.mc_simulations)
        simulated_sharpes = np.zeros(self.config.mc_simulations)

        for sim_idx in range(self.config.mc_simulations):
            if self.config.mc_method == "bootstrap":
                resampled = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
            else:
                mu, sigma = np.mean(trade_pnls), np.std(trade_pnls)
                resampled = np.random.normal(mu, sigma, len(trade_pnls))

            equity_path = self.config.initial_capital * np.cumprod(1 + resampled)
            simulated_returns[sim_idx] = equity_path[-1] / equity_path[0] - 1

            # Max drawdown for this simulation
            peak = np.maximum.accumulate(equity_path)
            dd = (peak - equity_path) / peak
            simulated_drawdowns[sim_idx] = np.max(dd)

            # Sharpe for this simulation
            rets = np.diff(equity_path) / equity_path[:-1]
            if np.std(rets) > 0:
                simulated_sharpes[sim_idx] = np.mean(rets) / np.std(rets) * np.sqrt(252)
            else:
                simulated_sharpes[sim_idx] = 0.0

        # FIX: Now properly compute and store MC statistics
        tail_mask = simulated_returns <= np.percentile(simulated_returns, 5)
        cvar_95 = float(-np.mean(simulated_returns[tail_mask])) if np.any(tail_mask) else 0.0

        mc_stats = MCStatistics(
            mean_return=float(np.mean(simulated_returns)),
            median_return=float(np.median(simulated_returns)),
            p5_return=float(np.percentile(simulated_returns, 5)),
            p95_return=float(np.percentile(simulated_returns, 95)),
            var_95=float(-np.percentile(simulated_returns, 5)),
            cvar_95=cvar_95,
            max_drawdown_p5=float(np.percentile(simulated_drawdowns, 95)),
            max_drawdown_median=float(np.median(simulated_drawdowns)),
            sharpe_p5=float(np.percentile(simulated_sharpes, 5)),
            sharpe_median=float(np.median(simulated_sharpes)),
            prob_positive=float(np.mean(simulated_returns > 0)),
            num_simulations=self.config.mc_simulations,
            simulated_returns=simulated_returns,
            simulated_drawdowns=simulated_drawdowns,
            simulated_sharpes=simulated_sharpes,
        )

        base_result.mc_statistics = mc_stats
        return base_result

    def _compute_results(self, trades: List[BacktestTrade], equity: np.ndarray,
                         benchmark_return: float = 0.0,
                         annualization: Optional[float] = None) -> BacktestResult:
        """Compute backtest performance metrics from trade list and equity curve.

        Args:
            trades: List of closed trades.
            equity: Equity curve array.
            benchmark_return: Benchmark return for alpha/IR computation.
            annualization: Optional dynamic annualization factor.

        Returns:
            BacktestResult with all performance metrics.
        """
        if len(equity) < 2:
            return BacktestResult(
                total_return=0, annualized_return=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown=0, max_drawdown_duration_bars=0,
                calmar_ratio=0, win_rate=0, profit_factor=0,
                total_trades=0, avg_trade_pnl=0, avg_winning_trade=0,
                avg_losing_trade=0, avg_holding_period=0,
                trades=trades, equity_curve=equity,
                benchmark_return=benchmark_return,
            )

        # Guard against zero/negative equity to prevent division by zero
        if equity[0] <= 0:
            return BacktestResult(
                total_return=0, annualized_return=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown=0, max_drawdown_duration_bars=0,
                calmar_ratio=0, win_rate=0, profit_factor=0,
                total_trades=0, avg_trade_pnl=0, avg_winning_trade=0,
                avg_losing_trade=0, avg_holding_period=0,
                trades=trades, equity_curve=equity,
                benchmark_return=benchmark_return,
            )

        # Safe returns calculation - replace zeros to prevent warnings
        equity_safe = np.where(equity[:-1] == 0, np.nan, equity[:-1])
        returns = np.diff(equity) / equity_safe
        returns = np.nan_to_num(returns, nan=0.0)

        total_return = (equity[-1] / equity[0]) - 1
        bars = len(equity)
        if annualization is None:
            annualization = 365 * 24 * 60  # minute bars

        # Guard against extreme values causing overflow
        # Cap total_return at reasonable bounds for power calculation
        total_return_clamped = max(min(total_return, 100), -0.9999)
        exponent = annualization / bars if bars > 0 else 0
        
        # For extreme cases, cap the annualized return directly
        if abs(total_return) > 100 or exponent > 5000:
            # Extreme case - just return a capped value based on total return sign
            annualized_return = 1000.0 if total_return > 0 else -0.99
        else:
            try:
                annualized_return = (1 + total_return_clamped) ** exponent - 1
                if not np.isfinite(annualized_return):
                    annualized_return = 1000.0 if total_return > 0 else -0.99
            except (OverflowError, FloatingPointError):
                annualized_return = 1000.0 if total_return > 0 else -0.99

        rf_per_bar = 0.0
        excess_returns = returns - rf_per_bar
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization) if np.std(excess_returns) > 0 else 0

        downside = excess_returns[excess_returns < 0]
        sortino = np.mean(excess_returns) / np.std(downside) * np.sqrt(annualization) if len(downside) > 0 and np.std(downside) > 0 else 0

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = float(np.max(drawdown))

        dd_duration = 0
        max_dd_duration = 0
        for d in drawdown:
            if d > 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        calmar = annualized_return / max_dd if max_dd > 0 else 0

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]
        win_rate = len(winning) / len(trades) if trades else 0
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        alpha = total_return - benchmark_return
        if benchmark_return != 0 and len(returns) > 0:
            benchmark_returns_arr = np.full_like(returns, benchmark_return / len(returns))
            active_returns = returns - benchmark_returns_arr
            tracking_error = np.std(active_returns) * np.sqrt(annualization) if np.std(active_returns) > 0 else 1e-10
            information_ratio = (alpha / len(returns) * annualization) / tracking_error if tracking_error > 0 else 0
        else:
            information_ratio = 0.0

        return BacktestResult(
            total_return=float(total_return),
            annualized_return=float(annualized_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_dd),
            max_drawdown_duration_bars=max_dd_duration,
            calmar_ratio=float(calmar),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor),
            total_trades=len(trades),
            avg_trade_pnl=float(np.mean([t.pnl for t in trades])) if trades else 0,
            avg_winning_trade=float(np.mean([t.pnl for t in winning])) if winning else 0,
            avg_losing_trade=float(np.mean([t.pnl for t in losing])) if losing else 0,
            avg_holding_period=float(np.mean([t.holding_period_bars for t in trades])) if trades else 0,
            trades=trades, equity_curve=equity, drawdown_curve=drawdown,
            benchmark_return=benchmark_return, alpha=float(alpha),
            information_ratio=float(information_ratio),
        )


__all__ = ["BacktestEngine"]
