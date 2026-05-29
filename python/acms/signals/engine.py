"""Composite signal generation engine."""

import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from acms.core import Signal, SignalDirection, Candle
from acms.indicators import (
    RSI, MACD, BollingerBands, ATR, ADX, StochasticOscillator,
    IchimokuCloud, VWAPIndicator, OBVIndicator, CMFIndicator,
    ConnorsRSI, TTMSqueeze, VolumeWeightedMACD,
    compute_hurst_exponent, compute_zscore,
)
from acms.signals.config import SignalConfig
from acms.signals.bayesian import BayesianConfidenceTracker
from acms.signals.persistence import SignalPersistenceFilter
from acms.signals.divergence import DivergenceDetector
from acms.signals.regime import MarketRegime, RegimeDetector


class SignalStrength(str):
    """Signal strength classification."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


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
    ):
        """Generate signals across multiple timeframes and aggregate.

        Args:
            candles_by_tf: Dict mapping timeframe name to candle list.
            symbol: Trading symbol.

        Returns:
            MultiTimeframeSignal with aggregated results.
        """
        from acms.signals.config import MultiTimeframeSignal

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
        """Adjust signal weights based on market regime."""
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
        """Compute signal-to-noise ratio."""
        values = [val for val, _ in signals.values()]
        if len(values) < 2:
            return 0.0
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            return 0.0
        return float(abs(mean_val) / std_val)

    def _compute_dynamic_threshold(self) -> float:
        """Compute dynamic signal threshold based on recent signal history."""
        if not self.config.dynamic_threshold_enabled or len(self._signal_history) < 20:
            return self.config.min_signal_strength
        history = np.array(list(self._signal_history))
        abs_history = np.abs(history)
        abs_history = abs_history[np.isfinite(abs_history)]
        if len(abs_history) < 10:
            return self.config.min_signal_strength
        dynamic = float(np.percentile(abs_history, self.config.dynamic_threshold_percentile))
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
        """Compute continuous volatility-adjusted confidence score."""
        try:
            if len(closes) < 14:
                return 0.0
            atr = ATR(self.config.atr_period).compute(highs, lows, closes)
            self._indicator_values["atr"] = atr
            if np.isnan(atr) or closes[-1] == 0:
                return 0.0
            atr_pct = atr / closes[-1] * 100
            self._indicator_values["atr_pct"] = atr_pct
            avg_price = np.mean(closes[-14:])
            if avg_price == 0:
                return 0.0
            normalized_atr_pct = atr / avg_price
            median_atr_pct = 0.02
            confidence = 1.0 / (1.0 + np.exp(10 * (normalized_atr_pct - median_atr_pct)))
            return float(confidence)
        except Exception:
            return 0.0

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
            recent_trend = 1.0 if closes[-1] > closes[-5] else -1.0
            return -0.5 * recent_trend
        elif hurst > 0.55:
            recent_trend = 1.0 if closes[-1] > closes[-5] else -1.0
            return 0.5 * recent_trend
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


__all__ = [
    "SignalStrength",
    "SignalEngine",
]
