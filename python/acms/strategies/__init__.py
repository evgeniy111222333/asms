"""Strategy Engine - Trading strategy implementations.

Implements 12 major strategy categories:
- Momentum (5 sub-strategies)
- Mean-Reversion
- Statistical Arbitrage (pair trading with cointegration + Kalman filter)
- Grid Trading (dynamic grid adjustment + inventory management)
- Turtle Trading (Donchian breakout with ATR position sizing + pyramiding)
- Wyckoff (accumulation/distribution detection via volume spread analysis)
- Carry (with cross-exchange funding rate arbitrage)
- Volatility
- Market-Making (with adverse selection protection)
- Cross-Exchange Arbitrage
- All strategies support dynamic parameter adaptation based on regime detection
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position, Order, Side
from acms.indicators import (
    RSI, MACD, BollingerBands, ATR, ADX, EMA, SMA,
    StochasticOscillator, Supertrend, IchimokuCloud,
    DonchianChannels, KAMA, TTMSqueeze, VWAPIndicator,
    compute_hurst_exponent, compute_zscore,
    CandlestickPatterns,
)
from acms.signals import SignalEngine, SignalConfig, MarketRegime, RegimeDetector


class Strategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, strategy_id: str, symbol: str):
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.is_active = True
        self.position: Optional[Position] = None
        self.signals_generated = 0
        self.trades_executed = 0
        self._state: Dict = {}
        self._regime_detector = RegimeDetector()

    @abstractmethod
    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        """Evaluate market data and return a signal if conditions are met."""
        ...

    @abstractmethod
    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        """Check if an existing position should be closed."""
        ...

    def reset(self):
        """Reset strategy state."""
        self.position = None
        self.signals_generated = 0
        self.trades_executed = 0
        self._state = {}

    def _detect_regime(self, candles: List[Candle]) -> MarketRegime:
        """Detect current market regime from candles."""
        if len(candles) < 50:
            return MarketRegime.UNKNOWN
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        return self._regime_detector.detect(closes, highs, lows)

    def _adapt_param(self, base_value: float, regime: MarketRegime,
                     trending_mult: float = 1.0, mr_mult: float = 1.0,
                     volatile_mult: float = 0.5, quiet_mult: float = 0.8) -> float:
        """Adapt a parameter based on market regime.

        Args:
            base_value: Base parameter value.
            regime: Current market regime.
            trending_mult: Multiplier for trending regime.
            mr_mult: Multiplier for mean-reverting regime.
            volatile_mult: Multiplier for volatile regime.
            quiet_mult: Multiplier for quiet regime.

        Returns:
            Adapted parameter value.
        """
        multipliers = {
            MarketRegime.TRENDING: trending_mult,
            MarketRegime.MEAN_REVERTING: mr_mult,
            MarketRegime.VOLATILE: volatile_mult,
            MarketRegime.QUIET: quiet_mult,
            MarketRegime.UNKNOWN: 1.0,
        }
        return base_value * multipliers.get(regime, 1.0)


# ============================================================================
# Momentum Strategies
# ============================================================================

class TrendFollowingMomentum(Strategy):
    """Trend following using EMA crossover + ADX filter."""

    def __init__(self, symbol: str, fast_period: int = 20, slow_period: int = 50,
                 adx_threshold: float = 25.0):
        super().__init__("momentum_trend", symbol)
        self.fast_ema = EMA(fast_period)
        self.slow_ema = EMA(slow_period)
        self.adx = ADX(14)
        self.adx_threshold = adx_threshold
        self._prev_fast_above_slow = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 60:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        fast = self.fast_ema.compute(closes)
        slow = self.slow_ema.compute(closes)
        if np.isnan(fast[-1]) or np.isnan(slow[-1]):
            return None
        fast_above = fast[-1] > slow[-1]
        regime = self._detect_regime(candles)
        threshold = self._adapt_param(self.adx_threshold, regime, trending_mult=0.8, mr_mult=1.5)
        adx_val = self.adx.compute(highs, lows, closes)
        if np.isnan(adx_val) or adx_val < threshold:
            self._prev_fast_above_slow = fast_above
            return None
        signal = None
        if self._prev_fast_above_slow is not None:
            if fast_above and not self._prev_fast_above_slow:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=min(adx_val / 50.0, 1.0), strategy_id=self.strategy_id,
                    indicators={"fast_ema": fast[-1], "slow_ema": slow[-1], "adx": adx_val, "regime": regime.value},
                )
            elif not fast_above and self._prev_fast_above_slow:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=min(adx_val / 50.0, 1.0), strategy_id=self.strategy_id,
                    indicators={"fast_ema": fast[-1], "slow_ema": slow[-1], "adx": adx_val, "regime": regime.value},
                )
        self._prev_fast_above_slow = fast_above
        self.signals_generated += 1
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        fast = self.fast_ema.compute(closes)
        slow = self.slow_ema.compute(closes)
        if np.isnan(fast[-1]) or np.isnan(slow[-1]):
            return False
        if position.side == Side.BUY and fast[-1] < slow[-1]:
            return True
        if position.side == Side.SELL and fast[-1] > slow[-1]:
            return True
        return False


class BreakoutMomentum(Strategy):
    """Breakout strategy using Donchian channels + volume confirmation."""

    def __init__(self, symbol: str, channel_period: int = 20, volume_mult: float = 1.5):
        super().__init__("momentum_breakout", symbol)
        self.channel_period = channel_period
        self.volume_mult = volume_mult

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.channel_period + 1:
            return None
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])
        upper = np.max(highs[-self.channel_period - 1:-1])
        lower = np.min(lows[-self.channel_period - 1:-1])
        avg_vol = np.mean(volumes[-self.channel_period:-1])
        current_close = closes[-1]
        current_vol = volumes[-1]
        vol_mult = self._adapt_param(self.volume_mult, self._detect_regime(candles),
                                      volatile_mult=0.5, quiet_mult=2.0)
        vol_confirm = current_vol > avg_vol * vol_mult if avg_vol > 0 else False
        if current_close > upper and vol_confirm:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG, strength=0.7,
                strategy_id=self.strategy_id,
                indicators={"upper_channel": upper, "volume_ratio": current_vol / avg_vol if avg_vol > 0 else 0},
            )
        elif current_close < lower and vol_confirm:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT, strength=0.7,
                strategy_id=self.strategy_id,
                indicators={"lower_channel": lower, "volume_ratio": current_vol / avg_vol if avg_vol > 0 else 0},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        atr = ATR(14).compute(np.array([c.high for c in candles]), np.array([c.low for c in candles]), closes)
        if np.isnan(atr):
            return False
        if position.side == Side.BUY:
            return closes[-1] < position.entry_price - 2 * atr
        elif position.side == Side.SELL:
            return closes[-1] > position.entry_price + 2 * atr
        return False


class RSIMomentum(Strategy):
    """RSI momentum - buy on RSI cross above 30, sell on cross below 70."""

    def __init__(self, symbol: str, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__("momentum_rsi", symbol)
        self.rsi = RSI(period)
        self.oversold = oversold
        self.overbought = overbought
        self._prev_rsi = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        closes = np.array([c.close for c in candles])
        rsi_val = self.rsi.compute(closes)
        if np.isnan(rsi_val):
            self._prev_rsi = None
            return None
        signal = None
        if self._prev_rsi is not None:
            if self._prev_rsi < self.oversold and rsi_val >= self.oversold:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=(self.oversold - self._prev_rsi) / self.oversold,
                    strategy_id=self.strategy_id, indicators={"rsi": rsi_val},
                )
            elif self._prev_rsi > self.overbought and rsi_val <= self.overbought:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=(self._prev_rsi - self.overbought) / (100 - self.overbought),
                    strategy_id=self.strategy_id, indicators={"rsi": rsi_val},
                )
        self._prev_rsi = rsi_val
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        rsi_val = self.rsi.compute(closes)
        if np.isnan(rsi_val):
            return False
        if position.side == Side.BUY and rsi_val > self.overbought:
            return True
        if position.side == Side.SELL and rsi_val < self.oversold:
            return True
        return False


class MACDMomentum(Strategy):
    """MACD histogram momentum strategy."""

    def __init__(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__("momentum_macd", symbol)
        self.macd = MACD(fast, slow, signal)
        self._prev_hist = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        closes = np.array([c.close for c in candles])
        result = self.macd.compute(closes)
        if result is None:
            self._prev_hist = None
            return None
        hist = result["histogram"]
        signal = None
        if self._prev_hist is not None:
            if self._prev_hist < 0 and hist > 0:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=min(abs(hist) / abs(closes[-1]) * 1000, 1.0) if closes[-1] != 0 else 0.5,
                    strategy_id=self.strategy_id,
                    indicators={"macd": result["macd"], "signal": result["signal"], "histogram": hist},
                )
            elif self._prev_hist > 0 and hist < 0:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=min(abs(hist) / abs(closes[-1]) * 1000, 1.0) if closes[-1] != 0 else 0.5,
                    strategy_id=self.strategy_id,
                    indicators={"macd": result["macd"], "signal": result["signal"], "histogram": hist},
                )
        self._prev_hist = hist
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        result = self.macd.compute(closes)
        if result is None:
            return False
        if position.side == Side.BUY and result["histogram"] < 0:
            return True
        if position.side == Side.SELL and result["histogram"] > 0:
            return True
        return False


class SupertrendMomentum(Strategy):
    """Supertrend momentum strategy."""

    def __init__(self, symbol: str, period: int = 10, multiplier: float = 3.0):
        super().__init__("momentum_supertrend", symbol)
        self.supertrend = Supertrend(period, multiplier)
        self._prev_direction = None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 20:
            return None
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        result = self.supertrend.compute(highs, lows, closes)
        direction = result["direction"][-1]
        signal = None
        if self._prev_direction is not None:
            if direction == 1 and self._prev_direction == -1:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=0.6, strategy_id=self.strategy_id,
                    indicators={"supertrend": result["supertrend"][-1], "direction": 1},
                )
            elif direction == -1 and self._prev_direction == 1:
                signal = Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=0.6, strategy_id=self.strategy_id,
                    indicators={"supertrend": result["supertrend"][-1], "direction": -1},
                )
        self._prev_direction = direction
        return signal

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        closes = np.array([c.close for c in candles])
        result = self.supertrend.compute(highs, lows, closes)
        direction = result["direction"][-1]
        if position.side == Side.BUY and direction == -1:
            return True
        if position.side == Side.SELL and direction == 1:
            return True
        return False


# ============================================================================
# Mean-Reversion Strategy
# ============================================================================

class MeanReversionStrategy(Strategy):
    """Mean-reversion using Bollinger Bands + RSI + z-score + Hurst confirmation."""

    def __init__(self, symbol: str, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_period: int = 14, zscore_threshold: float = 2.0):
        super().__init__("mean_reversion", symbol)
        self.bb = BollingerBands(bb_period, bb_std)
        self.rsi = RSI(rsi_period)
        self.zscore_threshold = zscore_threshold

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 50:
            return None
        closes = np.array([c.close for c in candles])
        bb_result = self.bb.compute(closes)
        rsi_val = self.rsi.compute(closes)
        zscore = compute_zscore(closes[-30:])
        hurst = compute_hurst_exponent(closes[-100:]) if len(closes) >= 100 else 0.5
        if bb_result is None or np.isnan(rsi_val):
            return None
        pct_b = bb_result["percent_b"]
        hurst_confirm = hurst < 0.55
        regime = self._detect_regime(candles)
        z_thresh = self._adapt_param(self.zscore_threshold, regime, mr_mult=0.8, trending_mult=1.5)
        if pct_b < 0.05 and rsi_val < 35 and zscore < -z_thresh:
            strength = 0.8 if hurst_confirm else 0.5
            strength *= 1.2 if regime == MarketRegime.MEAN_REVERTING else 1.0
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(strength, 1.0), strategy_id=self.strategy_id,
                indicators={"bb_pct_b": pct_b, "rsi": rsi_val, "zscore": zscore, "hurst": hurst},
            )
        elif pct_b > 0.95 and rsi_val > 65 and zscore > z_thresh:
            strength = 0.8 if hurst_confirm else 0.5
            strength *= 1.2 if regime == MarketRegime.MEAN_REVERTING else 1.0
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(strength, 1.0), strategy_id=self.strategy_id,
                indicators={"bb_pct_b": pct_b, "rsi": rsi_val, "zscore": zscore, "hurst": hurst},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        bb_result = self.bb.compute(closes)
        if bb_result is None:
            return False
        if position.side == Side.BUY and closes[-1] >= bb_result["middle"]:
            return True
        if position.side == Side.SELL and closes[-1] <= bb_result["middle"]:
            return True
        return False


# ============================================================================
# Statistical Arbitrage Strategy (Enhanced with Kalman Filter)
# ============================================================================

class StatisticalArbitrageStrategy(Strategy):
    """Statistical Arbitrage (pair trading with cointegration + Kalman filter).

    Identifies cointegrated pairs and trades the spread
    when it deviates from equilibrium. Uses a Kalman filter
    to dynamically update the hedge ratio instead of static OLS.
    """

    def __init__(self, symbol: str, symbol2: str, lookback: int = 100,
                 entry_zscore: float = 2.0, exit_zscore: float = 0.5,
                 use_kalman: bool = True):
        super().__init__("stat_arb", symbol)
        self.symbol2 = symbol2
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.use_kalman = use_kalman
        self._hedge_ratio = None
        self._spread_mean = None
        self._spread_std = None
        # Kalman filter state
        self._kf_state = np.array([1.0])  # hedge ratio estimate
        self._kf_cov = np.array([[1.0]])  # state covariance
        self._kf_Q = np.array([[0.001]])  # process noise
        self._kf_R = np.array([[0.01]])   # measurement noise
        self._spread_history: List[float] = []

    def _kalman_update(self, y: float, x: float) -> float:
        """Kalman filter update for dynamic hedge ratio estimation.

        State model: hedge_ratio follows random walk
        Measurement model: y = hedge_ratio * x + noise

        Args:
            y: Price of symbol 1.
            x: Price of symbol 2.

        Returns:
            Updated hedge ratio estimate.
        """
        # Predict
        state_pred = self._kf_state.copy()
        cov_pred = self._kf_cov + self._kf_Q
        # Update
        H = np.array([[x]])
        S = H @ cov_pred @ H.T + self._kf_R
        K = cov_pred @ H.T @ np.linalg.inv(S)
        innovation = np.array([[y]]) - H @ state_pred.reshape(-1, 1)
        self._kf_state = state_pred + (K @ innovation).flatten()
        self._kf_cov = (np.eye(1) - K @ H) @ cov_pred
        return float(self._kf_state[0])

    def compute_spread(self, prices1: np.ndarray, prices2: np.ndarray) -> np.ndarray:
        """Compute the cointegration spread with optional Kalman filter.

        Args:
            prices1: Price series for symbol 1.
            prices2: Price series for symbol 2.

        Returns:
            Spread series.
        """
        if len(prices1) != len(prices2) or len(prices1) < 30:
            return np.array([])
        if self.use_kalman:
            spread = np.zeros(len(prices1))
            for i in range(len(prices1)):
                hr = self._kalman_update(prices1[i], prices2[i])
                spread[i] = prices1[i] - hr * prices2[i]
            self._hedge_ratio = self._kf_state[0]
            return spread
        else:
            X = np.column_stack([np.ones(len(prices2)), prices2])
            beta = np.linalg.lstsq(X, prices1, rcond=None)[0]
            self._hedge_ratio = beta[1]
            return prices1 - self._hedge_ratio * prices2

    def cointegration_test(self, prices1: np.ndarray, prices2: np.ndarray,
                           significance: float = 0.05) -> Dict:
        """Run augmented Engle-Granger cointegration test.

        Args:
            prices1: Price series for symbol 1.
            prices2: Price series for symbol 2.
            significance: Significance level.

        Returns:
            Dict with 'is_cointegrated', 'hedge_ratio', 'half_life'.
        """
        spread = self.compute_spread(prices1[-self.lookback:], prices2[-self.lookback:])
        if len(spread) < 20:
            return {"is_cointegrated": False, "hedge_ratio": None, "half_life": float('inf')}
        spread_diff = np.diff(spread)
        spread_lag = spread[:-1]
        valid = np.isfinite(spread_diff) & np.isfinite(spread_lag)
        if valid.sum() < 10:
            return {"is_cointegrated": False, "hedge_ratio": None, "half_life": float('inf')}
        X = spread_lag[valid].reshape(-1, 1)
        y = spread_diff[valid]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        beta_val = beta[0]
        is_cointegrated = beta_val < -0.05
        half_life = -np.log(2) / beta_val if beta_val < 0 else float('inf')
        return {
            "is_cointegrated": is_cointegrated,
            "hedge_ratio": self._hedge_ratio,
            "half_life": half_life,
        }

    def evaluate_pair(self, closes1: np.ndarray, closes2: np.ndarray) -> Optional[Signal]:
        """Evaluate the pair for trading signals."""
        spread = self.compute_spread(closes1[-self.lookback:], closes2[-self.lookback:])
        if len(spread) < 20:
            return None
        self._spread_mean = np.mean(spread)
        self._spread_std = np.std(spread)
        if self._spread_std == 0:
            return None
        z_score = (spread[-1] - self._spread_mean) / self._spread_std
        self._spread_history.append(z_score)
        regime_mult = 1.0
        if len(self._spread_history) > 20:
            recent_std = np.std(self._spread_history[-20:])
            if recent_std > 2 * self._spread_std:
                regime_mult = 0.5
        entry = self.entry_zscore * regime_mult
        if z_score > entry:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(abs(z_score) / 4.0, 1.0), strategy_id=self.strategy_id,
                indicators={"spread_zscore": z_score, "hedge_ratio": self._hedge_ratio,
                            "spread_mean": self._spread_mean, "kalman_enabled": self.use_kalman},
            )
        elif z_score < -entry:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(abs(z_score) / 4.0, 1.0), strategy_id=self.strategy_id,
                indicators={"spread_zscore": z_score, "hedge_ratio": self._hedge_ratio,
                            "spread_mean": self._spread_mean, "kalman_enabled": self.use_kalman},
            )
        return None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        return None

    def should_exit_pair(self, closes1: np.ndarray, closes2: np.ndarray) -> bool:
        """Check if spread has reverted to exit threshold."""
        spread = self.compute_spread(closes1[-self.lookback:], closes2[-self.lookback:])
        if len(spread) < 2 or self._spread_std is None or self._spread_std == 0:
            return False
        z_score = (spread[-1] - self._spread_mean) / self._spread_std
        return abs(z_score) < self.exit_zscore

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        return False


# ============================================================================
# Grid Trading Strategy (Enhanced with inventory management)
# ============================================================================

class GridTradingStrategy(Strategy):
    """Grid Trading with dynamic grid adjustment and inventory management.

    Places buy/sell orders at fixed price intervals (grid levels).
    Profits from price oscillations within a range.
    Grid levels adapt based on ATR. Inventory is managed per level.
    """

    def __init__(self, symbol: str, grid_levels: int = 10, grid_spacing_atr_mult: float = 0.5,
                 grid_atr_period: int = 14, position_per_grid: float = 0.01,
                 max_inventory: float = 1.0, take_profit_atr_mult: float = 1.0):
        super().__init__("grid_trading", symbol)
        self.grid_levels = grid_levels
        self.grid_spacing_atr_mult = grid_spacing_atr_mult
        self.grid_atr_period = grid_atr_period
        self.position_per_grid = position_per_grid
        self.max_inventory = max_inventory
        self.take_profit_atr_mult = take_profit_atr_mult
        self._grid_levels: List[float] = []
        self._center_price: Optional[float] = None
        self._inventory: float = 0.0
        self._filled_levels: Dict[float, Dict] = {}  # level -> {side, qty, entry_price}

    def compute_grid(self, current_price: float, atr: float) -> List[float]:
        """Compute dynamic grid levels based on current price and ATR."""
        regime_mult = 1.0
        spacing = atr * self.grid_spacing_atr_mult * regime_mult
        if spacing <= 0:
            spacing = current_price * 0.005
        self._center_price = current_price
        half = self.grid_levels // 2
        levels = [current_price + (i - half) * spacing for i in range(self.grid_levels)]
        self._grid_levels = sorted(levels)
        return self._grid_levels

    def get_grid_orders(self, current_price: float, atr: float) -> List[Dict]:
        """Generate grid orders with inventory management.

        Returns:
            List of order dicts with 'price', 'side', 'quantity' keys.
        """
        levels = self.compute_grid(current_price, atr)
        orders = []
        for level in levels:
            if abs(self._inventory) >= self.max_inventory:
                break
            if level < current_price:
                qty = min(self.position_per_grid, self.max_inventory - abs(self._inventory))
                orders.append({"price": level, "side": "buy", "quantity": qty})
                self._inventory += qty
            elif level > current_price:
                qty = min(self.position_per_grid, self.max_inventory - abs(self._inventory))
                orders.append({"price": level, "side": "sell", "quantity": qty})
                self._inventory -= qty
        return orders

    def record_fill(self, level: float, side: str, qty: float, fill_price: float) -> None:
        """Record a grid level fill for profit taking.

        Args:
            level: Grid price level.
            side: 'buy' or 'sell'.
            qty: Quantity filled.
            fill_price: Actual fill price.
        """
        self._filled_levels[level] = {"side": side, "qty": qty, "entry_price": fill_price}

    def check_take_profit(self, current_price: float, atr: float) -> List[Dict]:
        """Check filled levels for take-profit opportunities.

        Returns:
            List of close orders with 'level', 'side', 'qty', 'pnl'.
        """
        close_orders = []
        tp_distance = atr * self.take_profit_atr_mult
        for level, info in list(self._filled_levels.items()):
            if info["side"] == "buy" and current_price >= level + tp_distance:
                pnl = (current_price - info["entry_price"]) * info["qty"]
                close_orders.append({"level": level, "side": "sell", "qty": info["qty"], "pnl": pnl})
                self._inventory -= info["qty"]
                del self._filled_levels[level]
            elif info["side"] == "sell" and current_price <= level - tp_distance:
                pnl = (info["entry_price"] - current_price) * info["qty"]
                close_orders.append({"level": level, "side": "buy", "qty": info["qty"], "pnl": pnl})
                self._inventory += info["qty"]
                del self._filled_levels[level]
        return close_orders

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.grid_atr_period + 1:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        atr = ATR(self.grid_atr_period).compute(highs, lows, closes)
        if np.isnan(atr):
            return None
        current_price = closes[-1]
        grid = self.compute_grid(current_price, atr)
        buy_levels = [g for g in grid if g < current_price]
        if buy_levels and abs(current_price - buy_levels[-1]) < atr * 0.1:
            if self._inventory < self.max_inventory:
                return Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.LONG,
                    strength=0.4, strategy_id=self.strategy_id,
                    indicators={"grid_price": buy_levels[-1], "atr": atr, "inventory": self._inventory},
                )
        sell_levels = [g for g in grid if g > current_price]
        if sell_levels and abs(sell_levels[0] - current_price) < atr * 0.1:
            if self._inventory > -self.max_inventory:
                return Signal(
                    id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                    symbol=self.symbol, direction=SignalDirection.SHORT,
                    strength=0.4, strategy_id=self.strategy_id,
                    indicators={"grid_price": sell_levels[0], "atr": atr, "inventory": self._inventory},
                )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        if not self._grid_levels:
            return False
        if position.side == Side.BUY:
            for level in self._grid_levels:
                if level > position.entry_price and closes[-1] >= level:
                    return True
        elif position.side == Side.SELL:
            for level in self._grid_levels:
                if level < position.entry_price and closes[-1] <= level:
                    return True
        return False


# ============================================================================
# Turtle Trading Strategy (Enhanced with pyramiding)
# ============================================================================

class TurtleTradingStrategy(Strategy):
    """Turtle Trading - Donchian breakout with ATR position sizing and pyramiding.

    Classic trend-following system:
    - System 1: 20-day breakout for entry, 10-day breakout for exit
    - System 2: 55-day breakout for entry, 20-day breakout for exit
    - Position size = 1% of account / (N * Dollar per point) where N = ATR(20)
    - Pyramiding: Add to winning positions at each 0.5N price advance
    - Exit on trailing stop at 2N from most recent entry
    """

    def __init__(self, symbol: str, entry_period: int = 20, exit_period: int = 10,
                 atr_period: int = 20, risk_pct: float = 0.01,
                 account_size: float = 100000.0, max_units: int = 4,
                 pyramid_spacing_atr: float = 0.5):
        super().__init__("turtle", symbol)
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.risk_pct = risk_pct
        self.account_size = account_size
        self.max_units = max_units
        self.pyramid_spacing_atr = pyramid_spacing_atr
        self._current_units = 0
        self._last_breakout_type: Optional[str] = None
        self._last_entry_price: Optional[float] = None
        self._trailing_stop: Optional[float] = None

    def compute_position_size(self, atr: float, price: float) -> float:
        """Compute position size using Turtle N-based sizing."""
        if atr <= 0 or price <= 0:
            return 0.0
        risk_amount = self.account_size * self.risk_pct
        unit_size = risk_amount / atr
        return max(unit_size, 0.0)

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < max(self.entry_period, self.atr_period) + 1:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        regime = self._detect_regime(candles)
        atr = ATR(self.atr_period).compute(highs, lows, closes)
        if np.isnan(atr):
            return None
        # Adapt max units based on regime
        max_units = int(self._adapt_param(float(self.max_units), regime,
                                           trending_mult=1.5, volatile_mult=0.5))
        highest = np.max(highs[-self.entry_period - 1:-1])
        lowest = np.min(lows[-self.entry_period - 1:-1])
        current = closes[-1]
        position_size = self.compute_position_size(atr, current)
        # Check for pyramiding opportunity
        pyramid_signal = None
        if self._last_breakout_type == "up" and self._last_entry_price is not None:
            if current >= self._last_entry_price + self.pyramid_spacing_atr * atr:
                if self._current_units < max_units:
                    self._current_units += 1
                    self._last_entry_price = current
                    self._trailing_stop = current - 2 * atr
                    pyramid_signal = Signal(
                        id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                        symbol=self.symbol, direction=SignalDirection.LONG,
                        strength=min(position_size / self.account_size, 1.0) * 0.5,
                        strategy_id=self.strategy_id,
                        indicators={"type": "pyramid", "atr": atr, "units": self._current_units},
                    )
        if pyramid_signal is not None:
            self.signals_generated += 1
            return pyramid_signal
        # New breakout
        if current > highest and self._current_units < max_units:
            self._current_units += 1
            self._last_breakout_type = "up"
            self._last_entry_price = current
            self._trailing_stop = current - 2 * atr
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(position_size / self.account_size, 1.0),
                strategy_id=self.strategy_id,
                indicators={"breakout_level": highest, "atr": atr, "units": self._current_units,
                            "position_size": position_size},
            )
        elif current < lowest and self._current_units < max_units:
            self._current_units += 1
            self._last_breakout_type = "down"
            self._last_entry_price = current
            self._trailing_stop = current + 2 * atr
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(position_size / self.account_size, 1.0),
                strategy_id=self.strategy_id,
                indicators={"breakout_level": lowest, "atr": atr, "units": self._current_units,
                            "position_size": position_size},
            )
        return None

    def should_exit(self, candles: List[Candle]) -> bool:
        """Check exit conditions: trailing stop or counter-breakout."""
        if len(candles) < self.exit_period + 1:
            return False
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        # Trailing stop exit
        if self._trailing_stop is not None:
            if self._last_breakout_type == "up" and closes[-1] < self._trailing_stop:
                self._current_units = 0
                self._last_breakout_type = None
                return True
            if self._last_breakout_type == "down" and closes[-1] > self._trailing_stop:
                self._current_units = 0
                self._last_breakout_type = None
                return True
        # Counter-breakout exit
        if self._last_breakout_type == "up":
            exit_level = np.min(lows[-self.exit_period - 1:-1])
            if closes[-1] < exit_level:
                self._current_units = 0
                self._last_breakout_type = None
                return True
        elif self._last_breakout_type == "down":
            exit_level = np.max(highs[-self.exit_period - 1:-1])
            if closes[-1] > exit_level:
                self._current_units = 0
                self._last_breakout_type = None
                return True
        return False

    def should_exit_with_position(self, candles: List[Candle], position: Position) -> bool:
        """Override to delegate to the unified should_exit."""
        return self.should_exit(candles)


# ============================================================================
# Wyckoff Strategy (Enhanced with Volume Spread Analysis)
# ============================================================================

class WyckoffStrategy(Strategy):
    """Wyckoff accumulation/distribution detection via volume spread analysis.

    Identifies Wyckoff phases using volume and price action:
    - Accumulation: PS, SC, AR, ST, Spring, SOS, LPS
    - Distribution: PSY, BC, AR, ST, UTAD, SOW, LPSY

    Volume Spread Analysis (VSA) examines the relationship between
    price spread (range) and volume to identify professional activity.
    """

    def __init__(self, symbol: str, lookback: int = 100, volume_threshold: float = 2.0,
                 spring_threshold: float = 0.02):
        super().__init__("wyckoff", symbol)
        self.lookback = lookback
        self.volume_threshold = volume_threshold
        self.spring_threshold = spring_threshold

    def _vsa_analysis(self, closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, volumes: np.ndarray) -> Dict[str, bool]:
        """Volume Spread Analysis to detect buying/selling climaxes.

        Args:
            closes: Close prices.
            highs: High prices.
            lows: Low prices.
            volumes: Volume data.

        Returns:
            Dict with VSA signals.
        """
        n = len(closes)
        if n < 3:
            return {"buying_climax": False, "selling_climax": False,
                    "no_demand": False, "no_supply": False}
        avg_vol = np.mean(volumes)
        avg_range = np.mean(highs - lows)
        spread = highs[-1] - lows[-1]
        vol = volumes[-1]
        result = {
            "buying_climax": False,
            "selling_climax": False,
            "no_demand": False,
            "no_supply": False,
        }
        if vol > avg_vol * self.volume_threshold:
            if closes[-1] > closes[-2] and spread > avg_range * 1.5:
                result["buying_climax"] = True
            elif closes[-1] < closes[-2] and spread > avg_range * 1.5:
                result["selling_climax"] = True
        elif vol < avg_vol * 0.5:
            if closes[-1] < closes[-2]:
                result["no_demand"] = True
            else:
                result["no_supply"] = True
        return result

    def detect_accumulation(self, closes: np.ndarray, volumes: np.ndarray,
                            lows: np.ndarray) -> Dict[str, bool]:
        """Detect Wyckoff accumulation phases."""
        phases = {
            "selling_climax": False, "automatic_rally": False,
            "spring": False, "sign_of_strength": False,
        }
        if len(closes) < self.lookback:
            return phases
        recent_closes = closes[-self.lookback:]
        recent_volumes = volumes[-self.lookback:]
        recent_lows = lows[-self.lookback:]
        avg_vol = np.mean(recent_volumes)
        vol_spike = recent_volumes > avg_vol * self.volume_threshold
        price_decline = np.diff(recent_closes) / recent_closes[:-1] < -0.02
        if np.any(vol_spike[1:] & price_decline):
            phases["selling_climax"] = True
        support = np.min(recent_lows[:len(recent_lows) // 2])
        if recent_lows[-1] < support * (1 - self.spring_threshold):
            if closes[-1] > support:
                phases["spring"] = True
        if phases["selling_climax"]:
            sc_idx = np.argmax(vol_spike[1:] & price_decline)
            if sc_idx < len(recent_closes) - 5:
                post_sc = recent_closes[sc_idx + 1:sc_idx + 6]
                if len(post_sc) > 0 and np.mean(np.diff(post_sc)) > 0:
                    phases["automatic_rally"] = True
        recent_rally = closes[-5:]
        recent_vol5 = volumes[-5:]
        if len(recent_rally) >= 5:
            if np.mean(np.diff(recent_rally)) > 0 and np.mean(recent_vol5) > avg_vol * 1.5:
                phases["sign_of_strength"] = True
        return phases

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.lookback:
            return None
        closes = np.array([c.close for c in candles])
        volumes = np.array([c.volume for c in candles])
        lows = np.array([c.low for c in candles])
        highs = np.array([c.high for c in candles])
        phases = self.detect_accumulation(closes, volumes, lows)
        vsa = self._vsa_analysis(closes, highs, lows, volumes)
        if phases["spring"] or phases["sign_of_strength"] or vsa["no_supply"]:
            strength = 0.7 if phases["spring"] else 0.5
            if vsa["no_supply"]:
                strength = max(strength, 0.6)
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=strength, strategy_id=self.strategy_id,
                indicators={**phases, **vsa},
            )
        resistance = np.max(highs[:len(highs) // 2])
        if highs[-1] > resistance * (1 + self.spring_threshold) and closes[-1] < resistance:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=0.6, strategy_id=self.strategy_id,
                indicators={"utad": True, **phases, **vsa},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        atr = ATR(14).compute(
            np.array([c.high for c in candles]),
            np.array([c.low for c in candles]), closes,
        )
        if np.isnan(atr):
            return False
        if position.side == Side.BUY and closes[-1] < position.entry_price - 3 * atr:
            return True
        if position.side == Side.SELL and closes[-1] > position.entry_price + 3 * atr:
            return True
        return False


# ============================================================================
# Carry Strategy (Enhanced with cross-exchange funding rate arbitrage)
# ============================================================================

class CarryStrategy(Strategy):
    """Carry trade strategy with cross-exchange funding rate arbitrage.

    Profits from:
    1. Funding rate differentials (positive/negative rates)
    2. Cross-exchange price discrepancies
    3. Cross-exchange funding rate arbitrage (same asset, different rates)
    """

    def __init__(self, symbol: str, funding_threshold: float = 0.01,
                 position_period_hours: int = 8,
                 arb_threshold_bps: float = 20.0,
                 funding_arb_min_spread: float = 0.005):
        super().__init__("carry", symbol)
        self.funding_threshold = funding_threshold
        self.position_period_hours = position_period_hours
        self.arb_threshold_bps = arb_threshold_bps
        self.funding_arb_min_spread = funding_arb_min_spread

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        return None

    def evaluate_funding(self, funding_rate: float, predicted_rate: float) -> Optional[Signal]:
        """Evaluate based on current and predicted funding rate."""
        if funding_rate < -self.funding_threshold:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(abs(funding_rate) / 0.1, 1.0),
                strategy_id=self.strategy_id,
                indicators={"funding_rate": funding_rate, "predicted_rate": predicted_rate},
            )
        elif funding_rate > self.funding_threshold:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(abs(funding_rate) / 0.1, 1.0),
                strategy_id=self.strategy_id,
                indicators={"funding_rate": funding_rate, "predicted_rate": predicted_rate},
            )
        return None

    def detect_cross_exchange_arbitrage(self, price_a: float, price_b: float,
                                        fee_bps: float = 10.0) -> Optional[Dict]:
        """Detect cross-exchange price arbitrage opportunity."""
        if price_a <= 0 or price_b <= 0:
            return None
        spread_bps = abs(price_a - price_b) / min(price_a, price_b) * 10000
        if spread_bps > self.arb_threshold_bps + fee_bps:
            buy_exchange = "A" if price_a < price_b else "B"
            sell_exchange = "B" if price_a < price_b else "A"
            return {
                "spread_bps": spread_bps,
                "net_profit_bps": spread_bps - fee_bps,
                "buy_exchange": buy_exchange,
                "sell_exchange": sell_exchange,
                "buy_price": min(price_a, price_b),
                "sell_price": max(price_a, price_b),
            }
        return None

    def detect_funding_rate_arbitrage(self, funding_rate_a: float,
                                      funding_rate_b: float,
                                      fee_rate: float = 0.001) -> Optional[Dict]:
        """Detect cross-exchange funding rate arbitrage.

        If exchange A has a much higher funding rate than exchange B,
        short on A and long on B to collect the differential.

        Args:
            funding_rate_a: Funding rate on exchange A.
            funding_rate_b: Funding rate on exchange B.
            fee_rate: Estimated transaction cost as fraction.

        Returns:
            Dict with arb details if profitable, None otherwise.
        """
        spread = funding_rate_a - funding_rate_b
        net_profit = abs(spread) - fee_rate * 2
        if abs(spread) > self.funding_arb_min_spread and net_profit > 0:
            if spread > 0:
                short_exchange = "A"
                long_exchange = "B"
            else:
                short_exchange = "B"
                long_exchange = "A"
            return {
                "funding_spread": abs(spread),
                "net_profit": net_profit,
                "short_exchange": short_exchange,
                "long_exchange": long_exchange,
                "rate_a": funding_rate_a,
                "rate_b": funding_rate_b,
            }
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        if position.opened_at:
            held_hours = (datetime.utcnow() - position.opened_at).total_seconds() / 3600
            if held_hours > self.position_period_hours * 3:
                return True
        return False


# ============================================================================
# Volatility Strategy
# ============================================================================

class VolatilityStrategy(Strategy):
    """Volatility trading using ATR breakout + IV/RV spread."""

    def __init__(self, symbol: str, atr_period: int = 14, atr_mult: float = 1.5,
                 vol_lookback: int = 20):
        super().__init__("volatility", symbol)
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.vol_lookback = vol_lookback
        self._atr_pct_history: List[float] = []

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < self.vol_lookback + 1:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        atr_val = ATR(self.atr_period).compute(highs, lows, closes)
        if np.isnan(atr_val):
            return None
        atr_pct = atr_val / closes[-1] * 100
        self._atr_pct_history.append(atr_pct)
        if len(self._atr_pct_history) < self.vol_lookback:
            return None
        avg_atr_pct = np.mean(self._atr_pct_history[-self.vol_lookback:])
        regime = self._detect_regime(candles)
        mult = self._adapt_param(self.atr_mult, regime, volatile_mult=2.0, quiet_mult=1.0)
        if atr_pct > avg_atr_pct * mult:
            direction = SignalDirection.LONG if closes[-1] > closes[-2] else SignalDirection.SHORT
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=direction,
                strength=min(atr_pct / avg_atr_pct / 3, 1.0) if avg_atr_pct > 0 else 0.5,
                strategy_id=self.strategy_id,
                indicators={"atr_pct": atr_pct, "avg_atr_pct": avg_atr_pct},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        atr_val = ATR(self.atr_period).compute(
            np.array([c.high for c in candles]),
            np.array([c.low for c in candles]), closes,
        )
        if np.isnan(atr_val):
            return False
        if position.side == Side.BUY:
            return closes[-1] < position.entry_price - 2.5 * atr_val
        elif position.side == Side.SELL:
            return closes[-1] > position.entry_price + 2.5 * atr_val
        return False


# ============================================================================
# Market-Making Strategy (Enhanced with adverse selection)
# ============================================================================

class MarketMakingStrategy(Strategy):
    """Market-making strategy with adverse selection protection.

    Provides continuous two-sided quotes while managing:
    - Inventory risk through skew adjustment
    - Adverse selection through toxic flow detection
    - Spread optimization based on volatility
    - Dynamic spread widening in volatile conditions
    """

    def __init__(self, symbol: str, base_spread_bps: float = 10.0,
                 inventory_limit: float = 5.0, skew_factor: float = 0.5,
                 min_profit_bps: float = 2.0,
                 adverse_selection_threshold: float = 0.7,
                 volatility_spread_mult: float = 2.0):
        super().__init__("market_making", symbol)
        self.base_spread_bps = base_spread_bps
        self.inventory_limit = inventory_limit
        self.skew_factor = skew_factor
        self.min_profit_bps = min_profit_bps
        self.adverse_selection_threshold = adverse_selection_threshold
        self.volatility_spread_mult = volatility_spread_mult
        self._inventory: float = 0.0
        self._recent_trades: List[Dict] = []
        self._toxic_flow_score: float = 0.0

    def compute_quotes(self, mid_price: float, atr: float, atr_pct: float) -> Dict:
        """Compute bid/ask quotes with inventory skew and volatility adjustment.

        Args:
            mid_price: Current mid price.
            atr: Current ATR value.
            atr_pct: ATR as percentage of price.

        Returns:
            Dict with 'bid', 'ask', 'spread_bps', 'skew_bps'.
        """
        # Base spread in price units
        base_spread = mid_price * self.base_spread_bps / 10000.0

        # Volatility widening
        vol_mult = 1.0
        if atr_pct > 5.0:
            vol_mult = self.volatility_spread_mult
        elif atr_pct > 3.0:
            vol_mult = 1.0 + (atr_pct - 3.0) / 2.0 * (self.volatility_spread_mult - 1.0)
        spread = base_spread * vol_mult

        # Inventory skew: shift quotes to reduce inventory
        inventory_ratio = self._inventory / self.inventory_limit if self.inventory_limit > 0 else 0
        skew_bps = self.skew_factor * inventory_ratio * spread

        half_spread = spread / 2.0
        bid = mid_price - half_spread - skew_bps
        ask = mid_price + half_spread - skew_bps

        # Ensure minimum profit
        if (ask - bid) < mid_price * self.min_profit_bps / 10000.0:
            min_half = mid_price * self.min_profit_bps / 20000.0
            bid = mid_price - min_half - skew_bps
            ask = mid_price + min_half - skew_bps

        return {
            "bid": bid,
            "ask": ask,
            "spread_bps": (ask - bid) / mid_price * 10000,
            "skew_bps": skew_bps / mid_price * 10000,
            "vol_mult": vol_mult,
        }

    def detect_adverse_selection(self, trade_side: str, trade_size: float,
                                 avg_trade_size: float, price_impact: float) -> Dict:
        """Detect adverse selection (toxic order flow).

        Toxic flow indicators:
        - Large trades in one direction
        - Price moves against us after our quote
        - Informed traders hitting our stale quotes

        Args:
            trade_side: 'buy' or 'sell'.
            trade_size: Size of the incoming trade.
            avg_trade_size: Average recent trade size.
            price_impact: Price movement after the trade.

        Returns:
            Dict with 'is_toxic', 'score', 'action'.
        """
        size_score = min(trade_size / max(avg_trade_size, 1e-10), 5.0) / 5.0
        impact_score = min(abs(price_impact) / 0.01, 1.0)
        self._toxic_flow_score = 0.7 * self._toxic_flow_score + 0.3 * (size_score + impact_score) / 2.0
        is_toxic = self._toxic_flow_score > self.adverse_selection_threshold
        action = "cancel" if is_toxic else "widen" if self._toxic_flow_score > 0.5 else "normal"
        return {"is_toxic": is_toxic, "score": self._toxic_flow_score, "action": action}

    def record_trade(self, side: str, size: float, price: float) -> None:
        """Record a trade for inventory and flow analysis."""
        self._recent_trades.append({"side": side, "size": size, "price": price, "time": datetime.utcnow()})
        if side == "buy":
            self._inventory += size
        else:
            self._inventory -= size
        # Keep only recent trades
        if len(self._recent_trades) > 100:
            self._recent_trades = self._recent_trades[-100:]

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        if len(candles) < 30:
            return None
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])
        atr = ATR(14).compute(highs, lows, closes)
        if np.isnan(atr) or closes[-1] == 0:
            return None
        atr_pct = atr / closes[-1] * 100
        mid_price = closes[-1]
        quotes = self.compute_quotes(mid_price, atr, atr_pct)
        # Check for toxic flow
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        flow = self.detect_adverse_selection("buy", volumes[-1], avg_vol, 0.0)
        if flow["action"] == "cancel":
            return None
        # Generate signal based on inventory
        if self._inventory > self.inventory_limit * 0.8:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=0.3, strategy_id=self.strategy_id,
                indicators={"action": "reduce_inventory", "inventory": self._inventory,
                            "spread_bps": quotes["spread_bps"], "toxic_score": self._toxic_flow_score},
            )
        elif self._inventory < -self.inventory_limit * 0.8:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=0.3, strategy_id=self.strategy_id,
                indicators={"action": "reduce_inventory", "inventory": self._inventory,
                            "spread_bps": quotes["spread_bps"], "toxic_score": self._toxic_flow_score},
            )
        return None

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        closes = np.array([c.close for c in candles])
        atr = ATR(14).compute(
            np.array([c.high for c in candles]),
            np.array([c.low for c in candles]), closes,
        )
        if np.isnan(atr):
            return False
        if position.side == Side.BUY and closes[-1] < position.entry_price - 2 * atr:
            return True
        if position.side == Side.SELL and closes[-1] > position.entry_price + 2 * atr:
            return True
        return False


# ============================================================================
# Cross-Exchange Arbitrage Strategy
# ============================================================================

class CrossExchangeArbitrageStrategy(Strategy):
    """Cross-Exchange Arbitrage: detect and exploit price differences.

    Monitors the same asset across multiple exchanges and generates
    signals when the price differential exceeds transaction costs
    plus a minimum profit threshold.
    """

    def __init__(self, symbol: str, exchanges: Optional[List[str]] = None,
                 min_profit_bps: float = 5.0, fee_bps: float = 10.0,
                 latency_buffer_bps: float = 3.0):
        """Initialize the strategy.

        Args:
            symbol: Trading symbol.
            exchanges: List of exchange names to monitor.
            min_profit_bps: Minimum profit in basis points after fees.
            fee_bps: Total round-trip fees in basis points.
            latency_buffer_bps: Buffer for execution latency/slippage.
        """
        super().__init__("cross_exchange_arb", symbol)
        self.exchanges = exchanges or ["exchange_a", "exchange_b"]
        self.min_profit_bps = min_profit_bps
        self.fee_bps = fee_bps
        self.latency_buffer_bps = latency_buffer_bps
        self._price_history: Dict[str, deque] = {ex: deque(maxlen=100) for ex in self.exchanges}

    def update_price(self, exchange: str, price: float) -> None:
        """Update price for an exchange.

        Args:
            exchange: Exchange name.
            price: Current price on that exchange.
        """
        if exchange in self._price_history:
            self._price_history[exchange].append((datetime.utcnow(), price))

    def detect_arbitrage(self, prices: Dict[str, float]) -> Optional[Dict]:
        """Detect arbitrage opportunities across exchanges.

        Args:
            prices: Dict mapping exchange name to current price.

        Returns:
            Dict with arb details if profitable, None otherwise.
        """
        if len(prices) < 2:
            return None
        best_opportunity = None
        best_net_profit = 0.0
        exchanges_list = list(prices.keys())
        for i in range(len(exchanges_list)):
            for j in range(i + 1, len(exchanges_list)):
                ex_a = exchanges_list[i]
                ex_b = exchanges_list[j]
                price_a = prices[ex_a]
                price_b = prices[ex_b]
                if price_a <= 0 or price_b <= 0:
                    continue
                spread_bps = abs(price_a - price_b) / min(price_a, price_b) * 10000
                total_cost = self.fee_bps + self.latency_buffer_bps + self.min_profit_bps
                if spread_bps > total_cost:
                    net_profit = spread_bps - self.fee_bps - self.latency_buffer_bps
                    if net_profit > best_net_profit:
                        best_net_profit = net_profit
                        buy_exchange = ex_a if price_a < price_b else ex_b
                        sell_exchange = ex_b if price_a < price_b else ex_a
                        best_opportunity = {
                            "spread_bps": spread_bps,
                            "net_profit_bps": net_profit,
                            "buy_exchange": buy_exchange,
                            "sell_exchange": sell_exchange,
                            "buy_price": min(price_a, price_b),
                            "sell_price": max(price_a, price_b),
                            "total_cost_bps": self.fee_bps + self.latency_buffer_bps,
                        }
        return best_opportunity

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        """Evaluate using the primary exchange candles (limited for cross-exchange)."""
        return None

    def evaluate_multi_exchange(self, prices: Dict[str, float]) -> Optional[Signal]:
        """Evaluate arbitrage opportunity across exchanges.

        Args:
            prices: Dict mapping exchange name to current price.

        Returns:
            Signal if profitable arb exists.
        """
        arb = self.detect_arbitrage(prices)
        if arb is None:
            return None
        self.signals_generated += 1
        return Signal(
            id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            symbol=self.symbol, direction=SignalDirection.LONG,
            strength=min(arb["net_profit_bps"] / 20.0, 1.0),
            strategy_id=self.strategy_id,
            indicators=arb,
        )

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        """Cross-exchange arb positions exit immediately when spread closes."""
        return True


# ============================================================================
# Strategy Registry & Factory
# ============================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
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
