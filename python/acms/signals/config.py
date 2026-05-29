"""Signal configuration and multi-timeframe data types."""

from dataclasses import dataclass, field
from typing import Dict

from acms.core import SignalDirection


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


__all__ = [
    "SignalConfig",
    "MultiTimeframeSignal",
]
