"""Signal Engine - Composite signal generation from multiple indicators.

Generates trading signals by combining multiple indicator readings
with configurable weights, confirmation logic, and advanced features:
- Multi-timeframe signal aggregation (combine signals from 1m, 5m, 1h, 4h, 1d)
- Bayesian confidence scoring (update confidence based on historical signal accuracy)
- Regime-aware signal weighting (trending market -> weight trend signals higher,
  ranging -> weight oscillator signals higher)
- Signal persistence filter (require N consecutive signals in same direction)
- Adaptive weight adjustment (track recent signal accuracy per sub-signal and adjust weights)
- Signal-to-noise ratio computation
- Composite signal with dynamic threshold (instead of fixed threshold)
- Signal divergence detection (RSI, MACD, volume)
"""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque

from acms.core import Signal, SignalDirection, Candle
from acms.indicators import (
    RSI, MACD, BollingerBands, ATR, ADX, StochasticOscillator,
    IchimokuCloud, VWAPIndicator, OBVIndicator, CMFIndicator,
    ConnorsRSI, TTMSqueeze, VolumeWeightedMACD,
    compute_hurst_exponent, compute_zscore,
    detect_bearish_divergence, detect_bullish_divergence,
)


class SignalStrength(str, Enum):
    """Signal strength classification."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    VOLATILE = "volatile"
    QUIET = "quiet"
    UNKNOWN = "unknown"


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_weight: float = 0.12

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_weight: float = 0.12

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    bb_weight: float = 0.08

    # ATR
    atr_period: int = 14
    atr_weight: float = 0.05

    # ADX
    adx_period: int = 14
    adx_threshold: float = 25.0
    adx_weight: float = 0.08

    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    stoch_weight: float = 0.08

    # Ichimoku
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    ichimoku_weight: float = 0.08

    # Volume
    volume_weight: float = 0.08

    # Divergence
    divergence_lookback: int = 50
    divergence_weight: float = 0.12

    # Connors RSI
    connors_rsi_weight: float = 0.05

    # TTM Squeeze
    ttm_squeeze_weight: float = 0.05

    # Hurst / z-score
    hurst_weight: float = 0.04
    zscore_weight: float = 0.05

    # Thresholds
    min_signal_strength: float = 0.3
    confirmation_threshold: float = 0.6

    # Persistence filter
    persistence_bars: int = 2

    # Bayesian confidence
    prior_confidence: float = 0.5
    confidence_decay: float = 0.95

    # Dynamic threshold
    dynamic_threshold_enabled: bool = True
    dynamic_threshold_lookback: int = 100
    dynamic_threshold_percentile: float = 60.0

    # SNR
    snr_lookback: int = 50


@dataclass
class MultiTimeframeSignal:
    """Signal aggregated across multiple timeframes."""
    symbol: str
    timeframes: Dict[str, float] = field(default_factory=dict)
    aggregated_signal: float = 0.0
    direction: SignalDirection = SignalDirection.NEUTRAL
    confidence: float = 0.0
    dominant_timeframe: str = ""
    snr: float = 0.0


class BayesianConfidenceTracker:
    """Bayesian update for signal confidence scoring.

    Maintains a running confidence estimate for each indicator,
    updating based on whether the signal correctly predicted
    subsequent price moves. Uses Beta distribution conjugate prior.
    """

    def __init__(self, num_indicators: int = 13, prior: float = 0.5, decay: float = 0.95):
        """Initialize the tracker.

        Args:
            num_indicators: Number of sub-signals to track.
            prior: Initial confidence prior (0-1).
            decay: Decay factor for old observations.
        """
        self.prior = prior
        self.decay = decay
        self.alpha = np.full(num_indicators, prior * 20 + 1)
        self.beta = np.full(num_indicators, (1 - prior) * 20 + 1)
        self.confidences = np.full(num_indicators, prior)

    def update(self, indicator_idx: int, was_correct: bool) -> float:
        """Update confidence for an indicator based on outcome.

        Args:
            indicator_idx: Index of the indicator.
            was_correct: Whether the signal was correct.

        Returns:
            Updated confidence value.
        """
        if indicator_idx < 0 or indicator_idx >= len(self.confidences):
            return self.prior
        self.alpha[indicator_idx] *= self.decay
        self.beta[indicator_idx] *= self.decay
        if was_correct:
            self.alpha[indicator_idx] += 1
        else:
            self.beta[indicator_idx] += 1
        total = self.alpha[indicator_idx] + self.beta[indicator_idx]
        if total == 0:
            return self.prior
        self.confidences[indicator_idx] = self.alpha[indicator_idx] / total
        return self.confidences[indicator_idx]

    def update_all(self, was_correct: bool) -> None:
        """Update all indicators with the same outcome (simplified batch update)."""
        for i in range(len(self.confidences)):
            self.update(i, was_correct)

    def get_weights(self) -> np.ndarray:
        """Get normalized confidence-based weights for signal combination."""
        total = self.confidences.sum()
        if total == 0:
            return np.ones_like(self.confidences) / len(self.confidences)
        return self.confidences / total

    def get_confidence(self) -> float:
        """Get overall mean confidence across all indicators."""
        return float(np.mean(self.confidences))


class SignalPersistenceFilter:
    """Filters whipsaw signals by requiring persistence.

    A signal must persist for N consecutive bars before
    being considered valid. This reduces false signals in
    choppy markets.
    """

    def __init__(self, persistence_bars: int = 2):
        self.persistence_bars = max(1, persistence_bars)
        self._signal_history: deque = deque(maxlen=persistence_bars + 1)
        self._last_direction = SignalDirection.NEUTRAL
        self._consecutive_count = 0

    def filter(self, direction: SignalDirection, strength: float) -> tuple:
        """Apply persistence filter to a signal.

        Args:
            direction: Current signal direction.
            strength: Current signal strength.

        Returns:
            Tuple of (filtered_direction, filtered_strength).
        """
        self._signal_history.append((direction, strength))
        if direction == self._last_direction:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_direction = direction
        if self._consecutive_count >= self.persistence_bars:
            return direction, strength
        elif self._consecutive_count == 1:
            return direction, strength * 0.3
        else:
            ratio = self._consecutive_count / self.persistence_bars
            return direction, strength * ratio

    def reset(self):
        """Reset filter state."""
        self._signal_history.clear()
        self._last_direction = SignalDirection.NEUTRAL
        self._consecutive_count = 0


class DivergenceDetector:
    """Detects multiple types of price-indicator divergences.

    Supports:
    - RSI divergence (regular and hidden)
    - MACD divergence
    - Volume divergence
    """

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def detect_rsi_divergence(self, closes: np.ndarray, rsi_series: np.ndarray) -> Dict[str, bool]:
        """Detect RSI divergences (regular and hidden)."""
        result = {
            "bullish_regular": False, "bearish_regular": False,
            "bullish_hidden": False, "bearish_hidden": False,
        }
        if len(closes) < self.lookback or len(rsi_series) < self.lookback:
            return result
        prices = closes[-self.lookback:]
        rsi = rsi_series[-self.lookback:]
        valid = np.isfinite(rsi)
        if valid.sum() < 20:
            return result
        mid = len(prices) // 2
        p_first_low = np.min(prices[:mid])
        p_second_low = np.min(prices[mid:])
        p_first_low_idx = np.argmin(prices[:mid])
        p_second_low_idx = mid + np.argmin(prices[mid:])
        if p_second_low < p_first_low:
            rsi_first = rsi[p_first_low_idx]
            rsi_second = rsi[p_second_low_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second):
                if rsi_second > rsi_first:
                    result["bullish_regular"] = True
        p_first_high = np.max(prices[:mid])
        p_second_high = np.max(prices[mid:])
        p_first_high_idx = np.argmax(prices[:mid])
        p_second_high_idx = mid + np.argmax(prices[mid:])
        if p_second_high > p_first_high:
            rsi_first = rsi[p_first_high_idx]
            rsi_second = rsi[p_second_high_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second):
                if rsi_second < rsi_first:
                    result["bearish_regular"] = True
        if p_second_low > p_first_low:
            rsi_first = rsi[p_first_low_idx]
            rsi_second = rsi[p_second_low_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second) and rsi_second < rsi_first:
                result["bullish_hidden"] = True
        if p_second_high < p_first_high:
            rsi_first = rsi[p_first_high_idx]
            rsi_second = rsi[p_second_high_idx]
            if np.isfinite(rsi_first) and np.isfinite(rsi_second) and rsi_second > rsi_first:
                result["bearish_hidden"] = True
        return result

    def detect_macd_divergence(self, closes: np.ndarray, macd_histogram: np.ndarray) -> Dict[str, bool]:
        """Detect MACD histogram divergences."""
        result = {"bullish": False, "bearish": False}
        if len(closes) < self.lookback or len(macd_histogram) < self.lookback:
            return result
        prices = closes[-self.lookback:]
        hist = macd_histogram[-self.lookback:]
        valid = np.isfinite(hist)
        if valid.sum() < 20:
            return result
        mid = len(prices) // 2
        p_first_low = np.min(prices[:mid])
        p_second_low = np.min(prices[mid:])
        if p_second_low < p_first_low:
            h_first = np.min(hist[:mid][np.isfinite(hist[:mid])]) if np.any(np.isfinite(hist[:mid])) else 0
            h_second = np.min(hist[mid:][np.isfinite(hist[mid:])]) if np.any(np.isfinite(hist[mid:])) else 0
            if h_second > h_first:
                result["bullish"] = True
        p_first_high = np.max(prices[:mid])
        p_second_high = np.max(prices[mid:])
        if p_second_high > p_first_high:
            h_first = np.max(hist[:mid][np.isfinite(hist[:mid])]) if np.any(np.isfinite(hist[:mid])) else 0
            h_second = np.max(hist[mid:][np.isfinite(hist[mid:])]) if np.any(np.isfinite(hist[mid:])) else 0
            if h_second < h_first:
                result["bearish"] = True
        return result

    def detect_volume_divergence(self, closes: np.ndarray, volumes: np.ndarray) -> Dict[str, bool]:
        """Detect volume divergences."""
        result = {"bullish": False, "bearish": False}
        if len(closes) < self.lookback:
            return result
        prices = closes[-self.lookback:]
        vols = volumes[-self.lookback:]
        mid = len(prices) // 2
        p_first_high = np.max(prices[:mid])
        p_second_high = np.max(prices[mid:])
        v_first = np.mean(vols[:mid])
        v_second = np.mean(vols[mid:])
        if p_second_high > p_first_high and v_second < v_first * 0.8:
            result["bearish"] = True
        p_first_low = np.min(prices[:mid])
        p_second_low = np.min(prices[mid:])
        if p_second_low < p_first_low and v_second < v_first * 0.8:
            result["bullish"] = True
        return result


class RegimeDetector:
    """Market regime detection.

    Classifies the current market into regimes:
    - TRENDING: Strong directional movement
    - MEAN_REVERTING: Range-bound, oscillating
    - VOLATILE: High volatility, unpredictable
    - QUIET: Low volatility, consolidation
    """

    def __init__(self, lookback: int = 100, adx_trend_threshold: float = 25.0,
                 vol_high_threshold: float = 0.05, vol_low_threshold: float = 0.01):
        self.lookback = lookback
        self.adx_trend_threshold = adx_trend_threshold
        self.vol_high_threshold = vol_high_threshold
        self.vol_low_threshold = vol_low_threshold

    def detect(self, closes: np.ndarray, highs: Optional[np.ndarray] = None,
               lows: Optional[np.ndarray] = None) -> MarketRegime:
        """Detect current market regime."""
        if len(closes) < 50:
            return MarketRegime.UNKNOWN
        returns = np.diff(np.log(closes[-min(self.lookback, len(closes)):]))
        returns = returns[np.isfinite(returns)]
        if len(returns) < 10:
            return MarketRegime.UNKNOWN
        vol = np.std(returns)
        hurst = compute_hurst_exponent(closes[-min(self.lookback, len(closes)):])
        adx = float('nan')
        if highs is not None and lows is not None and len(highs) >= 30:
            adx = ADX(14).compute(highs, lows, closes)
        if vol > self.vol_high_threshold:
            return MarketRegime.VOLATILE
        elif vol < self.vol_low_threshold:
            return MarketRegime.QUIET
        elif not np.isnan(adx) and adx > self.adx_trend_threshold:
            return MarketRegime.TRENDING
        elif not np.isnan(hurst) and hurst < 0.45:
            return MarketRegime.MEAN_REVERTING
        elif not np.isnan(adx) and adx > 20:
            return MarketRegime.TRENDING
        else:
            return MarketRegime.QUIET


class SignalEngine:
    """Composite signal generator combining multiple indicators.

    The engine computes signals from 13+ sub-signals:
    1. RSI - overbought/oversold + divergence
    2. MACD - trend direction + crossover
    3. Bollinger Bands - mean reversion signals
    4. ATR - volatility-adjusted confidence
    5. ADX - trend strength filter
    6. Stochastic - momentum confirmation
    7. Ichimoku - multi-timeframe trend
    8. Volume - confirmation via CMF
    9. Divergence - price-indicator divergence
    10. Hurst Exponent - mean-reversion vs trend regime
    11. Z-Score - statistical deviation
    12. Connors RSI - short-term momentum
    13. TTM Squeeze - volatility compression/expansion

    Advanced features:
    - Multi-timeframe signal aggregation
    - Bayesian confidence scoring
    - Regime-aware signal weighting
    - Signal persistence filter
    - Adaptive weight adjustment
    - Signal-to-noise ratio computation
    - Dynamic threshold (instead of fixed)
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self._indicator_values: Dict[str, float] = {}
        self._rsi_series: Optional[np.ndarray] = None
        self._macd_histogram: Optional[np.ndarray] = None
        # Advanced components
        self.bayesian = BayesianConfidenceTracker(num_indicators=13)
        self.persistence_filter = SignalPersistenceFilter(self.config.persistence_bars)
        self.divergence_detector = DivergenceDetector(self.config.divergence_lookback)
        self.regime_detector = RegimeDetector()
        # Accuracy tracking for adaptive weights
        self._signal_accuracy: deque = deque(maxlen=100)
        self._adaptive_weights: Optional[np.ndarray] = None
        # History for dynamic threshold and SNR
        self._signal_history: deque = deque(maxlen=self.config.dynamic_threshold_lookback)
        self._price_changes: deque = deque(maxlen=self.config.dynamic_threshold_lookback)

    def generate_signal(
        self,
        candles: List[Candle],
        symbol: str,
        strategy_id: str = "composite",
    ) -> Signal:
        """Generate a composite signal from candle data.

        Args:
            candles: List of candles (most recent last), minimum 100 recommended.
            symbol: Trading symbol.
            strategy_id: Strategy identifier.

        Returns:
            Signal with direction, strength, and indicator breakdown.
        """
        if len(candles) < 50:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=symbol, direction=SignalDirection.NEUTRAL,
                strength=0.0, strategy_id=strategy_id,
                indicators={}, timestamp=datetime.utcnow(),
            )
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Detect regime for weight adjustment
        regime = self.regime_detector.detect(closes, highs, lows)

        # Compute individual sub-signals
        rsi_signal = self._rsi_signal(closes)
        macd_signal = self._macd_signal(closes)
        bb_signal = self._bollinger_signal(closes)
        atr_signal = self._atr_signal(highs, lows, closes)
        adx_signal = self._adx_signal(highs, lows, closes)
        stoch_signal = self._stochastic_signal(highs, lows, closes)
        ichimoku_signal = self._ichimoku_signal(highs, lows, closes)
        volume_signal = self._volume_signal(highs, lows, closes, volumes)
        divergence_signal = self._divergence_signal(closes, volumes)
        hurst_signal = self._hurst_signal(closes)
        zscore_signal = self._zscore_signal(closes)
        connors_signal = self._connors_rsi_signal(closes)
        squeeze_signal = self._ttm_squeeze_signal(highs, lows, closes)

        signals = {
            "rsi": (rsi_signal, self.config.rsi_weight),
            "macd": (macd_signal, self.config.macd_weight),
            "bb": (bb_signal, self.config.bb_weight),
            "atr": (atr_signal, self.config.atr_weight),
            "adx": (adx_signal, self.config.adx_weight),
            "stoch": (stoch_signal, self.config.stoch_weight),
            "ichimoku": (ichimoku_signal, self.config.ichimoku_weight),
            "volume": (volume_signal, self.config.volume_weight),
            "divergence": (divergence_signal, self.config.divergence_weight),
            "hurst": (hurst_signal, self.config.hurst_weight),
            "zscore": (zscore_signal, self.config.zscore_weight),
            "connors_rsi": (connors_signal, self.config.connors_rsi_weight),
            "ttm_squeeze": (squeeze_signal, self.config.ttm_squeeze_weight),
        }

        # Apply adaptive weights from Bayesian tracker
        if self._adaptive_weights is not None and len(self._adaptive_weights) == len(signals):
            signal_items = list(signals.items())
            for i, (name, (sig, _)) in enumerate(signal_items):
                signals[name] = (sig, self._adaptive_weights[i])

        # Apply regime-based weight adjustment
        signals = self._adjust_for_regime(signals, regime)

        # Weighted combination
        total_weight = sum(w for _, w in signals.values())
        if total_weight == 0:
            weighted_sum = 0.0
        else:
            weighted_sum = sum(s * w for (s, _), w in signals.values()) / total_weight

        # Compute signal-to-noise ratio
        snr = self._compute_snr(signals)

        # Dynamic threshold
        threshold = self._compute_dynamic_threshold()

        # Direction
        if weighted_sum > threshold:
            direction = SignalDirection.LONG
        elif weighted_sum < -threshold:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        # Strength (0-1)
        strength = min(abs(weighted_sum), 1.0)

        # Confirmation check
        agreeing = sum(1 for (s, _), _ in signals.items() if s * weighted_sum > 0)
        total = len(signals)
        if total > 0 and agreeing / total < self.config.confirmation_threshold:
            strength *= 0.5

        # Apply persistence filter
        direction, strength = self.persistence_filter.filter(direction, strength)

        # Bayesian confidence
        confidence = self.bayesian.get_confidence()

        # Store signal for SNR and dynamic threshold
        self._signal_history.append(weighted_sum)

        indicator_values = {name: val for name, (val, _) in signals.items()}

        return Signal(
            id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            symbol=symbol, direction=direction, strength=strength,
            strategy_id=strategy_id, indicators=indicator_values,
            timestamp=datetime.utcnow(),
            metadata={
                "weighted_sum": weighted_sum,
                "confirmation_ratio": agreeing / total if total > 0 else 0,
                "agreeing_indicators": agreeing,
                "total_indicators": total,
                "regime": regime.value,
                "confidence": confidence,
                "snr": snr,
                "dynamic_threshold": threshold,
            },
        )

    def generate_multi_timeframe_signal(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        symbol: str,
    ) -> MultiTimeframeSignal:
        """Generate signals across multiple timeframes and aggregate.

        Args:
            candles_by_tf: Dict mapping timeframe name to candle list.
            symbol: Trading symbol.

        Returns:
            MultiTimeframeSignal with aggregated results.
        """
        tf_signals: Dict[str, float] = {}
        for tf, candles in candles_by_tf.items():
            sig = self.generate_signal(candles, symbol)
            val = sig.strength * (1 if sig.direction == SignalDirection.LONG else -1 if sig.direction == SignalDirection.SHORT else 0)
            tf_signals[tf] = val

        # Weight higher timeframes more (trend from higher TF, entry from lower)
        tf_weights = {"1d": 3.0, "4h": 2.0, "1h": 1.5, "15m": 1.0, "5m": 0.5, "1m": 0.3}
        total_w = 0.0
        weighted_sum = 0.0
        for tf, val in tf_signals.items():
            w = tf_weights.get(tf, 1.0)
            weighted_sum += val * w
            total_w += w

        aggregated = weighted_sum / total_w if total_w > 0 else 0.0

        # Dynamic threshold for MTF
        threshold = self._compute_dynamic_threshold()
        if aggregated > threshold:
            direction = SignalDirection.LONG
        elif aggregated < -threshold:
            direction = SignalDirection.SHORT
        else:
            direction = SignalDirection.NEUTRAL

        dominant = max(tf_signals, key=lambda k: abs(tf_signals[k])) if tf_signals else ""

        # Compute SNR for the aggregated signal
        snr = 0.0
        if len(tf_signals) > 1:
            vals = np.array(list(tf_signals.values()))
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            snr = abs(mean_val) / std_val if std_val > 0 else 0.0

        return MultiTimeframeSignal(
            symbol=symbol, timeframes=tf_signals,
            aggregated_signal=aggregated, direction=direction,
            confidence=self.bayesian.get_confidence(),
            dominant_timeframe=dominant, snr=snr,
        )

    def update_accuracy(self, signal_direction: SignalDirection, price_change: float):
        """Update signal accuracy tracking for adaptive weights.

        Args:
            signal_direction: Direction of the original signal.
            price_change: Subsequent price change (positive = up).
        """
        correct = (
            (signal_direction == SignalDirection.LONG and price_change > 0) or
            (signal_direction == SignalDirection.SHORT and price_change < 0) or
            (signal_direction == SignalDirection.NEUTRAL and abs(price_change) < 0.001)
        )
        self._signal_accuracy.append(1.0 if correct else 0.0)
        self._price_changes.append(price_change)
        # Update Bayesian tracker for all indicators
        self.bayesian.update_all(correct)
        # Update adaptive weights from Bayesian confidence
        self._adaptive_weights = self.bayesian.get_weights()

    def _adjust_for_regime(self, signals: Dict, regime: MarketRegime) -> Dict:
        """Adjust signal weights based on market regime.

        Trending: increase trend-following weights (MACD, ADX, Ichimoku).
        Mean-reverting: increase MR weights (RSI, BB, z-score).
        Volatile: reduce all weights, increase ATR.
        Quiet: reduce signal strength overall.
        """
        adjusted = {}
        for name, (value, weight) in signals.items():
            new_weight = weight
            if regime == MarketRegime.TRENDING:
                if name in ("macd", "adx", "ichimoku"):
                    new_weight = weight * 1.5
                elif name in ("rsi", "bb", "zscore", "connors_rsi"):
                    new_weight = weight * 0.5
            elif regime == MarketRegime.MEAN_REVERTING:
                if name in ("rsi", "bb", "zscore", "connors_rsi"):
                    new_weight = weight * 1.5
                elif name in ("macd", "adx", "ichimoku"):
                    new_weight = weight * 0.5
            elif regime == MarketRegime.VOLATILE:
                new_weight = weight * 0.5
                if name == "atr":
                    new_weight = weight * 1.5
            elif regime == MarketRegime.QUIET:
                new_weight = weight * 0.7
            adjusted[name] = (value, new_weight)
        return adjusted

    def _compute_snr(self, signals: Dict) -> float:
        """Compute signal-to-noise ratio.

        SNR = |mean(signal)| / std(signal) for the sub-signal values.
        High SNR means signals agree strongly in one direction.
        """
        values = [val for val, _ in signals.values()]
        if len(values) < 2:
            return 0.0
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return 0.0
        return float(abs(mean_val) / std_val)

    def _compute_dynamic_threshold(self) -> float:
        """Compute dynamic signal threshold based on recent signal history.

        Instead of a fixed threshold, adapts based on recent signal
        distribution. Uses percentile of absolute signal values.
        """
        if not self.config.dynamic_threshold_enabled or len(self._signal_history) < 20:
            return self.config.min_signal_strength
        history = np.array(list(self._signal_history))
        abs_history = np.abs(history)
        abs_history = abs_history[np.isfinite(abs_history)]
        if len(abs_history) < 10:
            return self.config.min_signal_strength
        dynamic = float(np.percentile(abs_history, self.config.dynamic_threshold_percentile))
        # Blend with fixed threshold for stability
        return 0.5 * dynamic + 0.5 * self.config.min_signal_strength

    # --- Individual Sub-Signal Methods ---

    def _rsi_signal(self, closes: np.ndarray) -> float:
        """RSI signal: -1 to 1 (oversold=buy, overbought=sell)."""
        rsi = RSI(self.config.rsi_period).compute(closes)
        self._indicator_values["rsi"] = rsi
        if np.isnan(rsi):
            return 0.0
        if rsi > self.config.rsi_overbought:
            return -(rsi - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
        elif rsi < self.config.rsi_oversold:
            return (self.config.rsi_oversold - rsi) / self.config.rsi_oversold
        else:
            mid = (self.config.rsi_overbought + self.config.rsi_oversold) / 2
            return (mid - rsi) / mid * 0.3

    def _macd_signal(self, closes: np.ndarray) -> float:
        """MACD signal: based on histogram direction and crossover."""
        macd = MACD(self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
        result = macd.compute(closes)
        if result is None or np.isnan(result["macd"]):
            return 0.0
        self._indicator_values["macd"] = result["macd"]
        self._indicator_values["macd_signal"] = result["signal"]
        self._indicator_values["macd_histogram"] = result["histogram"]
        if "macd_line" in result:
            self._macd_histogram = result.get("macd_line", np.array([]))
        hist = result["histogram"]
        if abs(hist) < 1e-10:
            return 0.0
        price = closes[-1] if closes[-1] != 0 else 1.0
        normalized = hist / price * 1000
        return max(-1.0, min(1.0, normalized))

    def _bollinger_signal(self, closes: np.ndarray) -> float:
        """Bollinger Bands signal: mean reversion at bands."""
        bb = BollingerBands(self.config.bb_period, self.config.bb_std)
        result = bb.compute(closes)
        if result is None:
            return 0.0
        self._indicator_values["bb_upper"] = result["upper"]
        self._indicator_values["bb_middle"] = result["middle"]
        self._indicator_values["bb_lower"] = result["lower"]
        price = closes[-1]
        band_width = result["upper"] - result["lower"]
        if band_width == 0:
            return 0.0
        pct_b = (price - result["lower"]) / band_width
        self._indicator_values["bb_pct_b"] = pct_b
        if pct_b > 0.95:
            return -0.8
        elif pct_b < 0.05:
            return 0.8
        else:
            return (0.5 - pct_b) * 1.6

    def _atr_signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """ATR signal: volatility-adjusted confidence modifier."""
        atr = ATR(self.config.atr_period).compute(highs, lows, closes)
        self._indicator_values["atr"] = atr
        if np.isnan(atr) or closes[-1] == 0:
            return 0.0
        atr_pct = atr / closes[-1] * 100
        self._indicator_values["atr_pct"] = atr_pct
        if atr_pct > 8:
            return 0.0
        elif atr_pct > 5:
            return 0.3
        else:
            return 1.0

    def _adx_signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """ADX signal: trend strength filter."""
        adx = ADX(self.config.adx_period).compute(highs, lows, closes)
        self._indicator_values["adx"] = adx
        if np.isnan(adx):
            return 0.0
        if adx < self.config.adx_threshold:
            return 0.0
        return min(adx / 100.0, 1.0)

    def _stochastic_signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Stochastic oscillator signal."""
        stoch = StochasticOscillator(
            self.config.stoch_k_period, self.config.stoch_d_period
        ).compute(highs, lows, closes)
        if stoch is None:
            return 0.0
        self._indicator_values["stoch_k"] = stoch["k"]
        self._indicator_values["stoch_d"] = stoch["d"]
        k = stoch["k"]
        if k > self.config.stoch_overbought:
            return -(k - self.config.stoch_overbought) / (100 - self.config.stoch_overbought)
        elif k < self.config.stoch_oversold:
            return (self.config.stoch_oversold - k) / self.config.stoch_oversold
        else:
            return (50 - k) / 50 * 0.3

    def _ichimoku_signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Ichimoku Cloud signal."""
        ich = IchimokuCloud(
            self.config.ichimoku_tenkan,
            self.config.ichimoku_kijun,
            self.config.ichimoku_senkou_b,
        ).compute(highs, lows, closes)
        if ich is None:
            return 0.0
        self._indicator_values["ichimoku_tenkan"] = ich["tenkan"]
        self._indicator_values["ichimoku_kijun"] = ich["kijun"]
        self._indicator_values["ichimoku_senkou_a"] = ich["senkou_a"]
        self._indicator_values["ichimoku_senkou_b"] = ich["senkou_b"]
        price = closes[-1]
        cloud_top = max(ich["senkou_a"], ich["senkou_b"])
        cloud_bottom = min(ich["senkou_a"], ich["senkou_b"])
        cloud_mid = (cloud_top + cloud_bottom) / 2
        if cloud_mid == 0:
            return 0.0
        if price > cloud_top:
            return min((price - cloud_top) / cloud_mid, 1.0)
        elif price < cloud_bottom:
            return max(-(cloud_bottom - price) / cloud_mid, -1.0)
        else:
            return 0.0

    def _volume_signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Volume-based confirmation signal using CMF."""
        cmf = CMFIndicator(20).compute(highs, lows, closes, volumes)
        self._indicator_values["cmf"] = cmf
        if not np.isnan(cmf):
            return max(-1.0, min(1.0, cmf * 5))
        return 0.0

    def _divergence_signal(self, closes: np.ndarray, volumes: np.ndarray) -> float:
        """Detect price-indicator divergences (RSI, MACD, Volume)."""
        rsi_series = RSI(self.config.rsi_period).compute_series(closes)
        self._rsi_series = rsi_series
        rsi_div = self.divergence_detector.detect_rsi_divergence(closes, rsi_series)
        score = 0.0
        if rsi_div["bullish_regular"] or rsi_div["bullish_hidden"]:
            score += 0.6
            self._indicator_values["bullish_rsi_divergence"] = 1.0
        if rsi_div["bearish_regular"] or rsi_div["bearish_hidden"]:
            score -= 0.6
            self._indicator_values["bearish_rsi_divergence"] = 1.0
        vol_div = self.divergence_detector.detect_volume_divergence(closes, volumes)
        if vol_div["bullish"]:
            score += 0.3
        if vol_div["bearish"]:
            score -= 0.3
        if self._macd_histogram is not None and len(self._macd_histogram) >= self.config.divergence_lookback:
            macd_div = self.divergence_detector.detect_macd_divergence(closes, self._macd_histogram[-len(closes):] if len(self._macd_histogram) >= len(closes) else self._macd_histogram)
            if macd_div["bullish"]:
                score += 0.3
            if macd_div["bearish"]:
                score -= 0.3
        return max(-1.0, min(1.0, score))

    def _hurst_signal(self, closes: np.ndarray) -> float:
        """Hurst exponent: mean-reversion vs trending regime."""
        if len(closes) < 100:
            return 0.0
        hurst = compute_hurst_exponent(closes[-100:])
        self._indicator_values["hurst"] = hurst
        if np.isnan(hurst):
            return 0.0
        if hurst < 0.45:
            return 0.5
        elif hurst > 0.55:
            return 0.5
        else:
            return 0.0

    def _zscore_signal(self, closes: np.ndarray) -> float:
        """Z-score signal for mean reversion."""
        if len(closes) < 30:
            return 0.0
        zscore = compute_zscore(closes[-30:])
        self._indicator_values["zscore"] = zscore
        if np.isnan(zscore):
            return 0.0
        if abs(zscore) > 2.5:
            return -np.sign(zscore) * 0.8
        elif abs(zscore) > 2.0:
            return -np.sign(zscore) * 0.5
        else:
            return 0.0

    def _connors_rsi_signal(self, closes: np.ndarray) -> float:
        """Connors RSI signal for short-term momentum."""
        if len(closes) < 110:
            return 0.0
        crsi = ConnorsRSI().compute(closes)
        self._indicator_values["connors_rsi"] = crsi
        if np.isnan(crsi):
            return 0.0
        if crsi > 80:
            return -0.6
        elif crsi < 20:
            return 0.6
        else:
            return (50 - crsi) / 50 * 0.3

    def _ttm_squeeze_signal(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """TTM Squeeze momentum signal."""
        squeeze = TTMSqueeze().compute(highs, lows, closes)
        if squeeze is None:
            return 0.0
        self._indicator_values["squeeze_active"] = float(squeeze["squeeze_active"])
        self._indicator_values["squeeze_momentum"] = squeeze["momentum"]
        if squeeze["squeeze_active"]:
            return 0.0
        else:
            return max(-1.0, min(1.0, squeeze["momentum"] / (ATR(14).compute(highs, lows, closes) + 1e-10)))
