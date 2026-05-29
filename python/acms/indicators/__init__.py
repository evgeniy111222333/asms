"""Technical Indicators Library for ACMS.

Implements 70+ technical indicators organized by category:
- Moving Averages (14+ types)
- Oscillators (16 types)
- Volatility Indicators (14 types)
- Volume Indicators (14 types)
- Candlestick Pattern Recognition (35+ patterns)
- Advanced Indicators (Connors RSI, TTM Squeeze, VWAP Bands, etc.)
- Statistical Measures (Hurst, z-score, divergence)
- Pattern Recognition (Support/Resistance, Pivot Points, Fibonacci)

All indicators handle edge cases (insufficient data, NaN, division by zero).
"""

from acms.indicators.moving_averages import (
    SMA, EMA, WMA, HMA, DEMA, TEMA, VWMA, KAMA, ALMA, FRAMA, ZLEMA,
    KaufmanAdaptiveMovingAverage, EhlersSuperSmoother,
)
from acms.indicators.oscillators import (
    RSI, ConnorsRSI, MACD, VolumeWeightedMACD, StochasticOscillator,
    CCI, WilliamsR, ROC, Momentum, TRIX, UltimateOscillator, MFI, ADX,
    Aroon, ChandeMomentumOscillator, EhlersFisherTransform,
)
from acms.indicators.volatility import (
    ATR, BollingerBands, KeltnerChannels, DonchianChannels,
    StandardDeviation, HistoricalVolatility, ParkinsonVolatility,
    GarmanKlassVolatility, ChaikinVolatility, TrueRange, ATRP,
)
from acms.indicators.volume import (
    OBVIndicator, CMFIndicator, ADLine, VolumeProfile, ForceIndex,
    EaseOfMovement, VolumeOscillator, NVI, PVI, IchimokuCloud,
)
from acms.indicators.advanced import (
    TTMSqueeze, VWAPIndicator, VWAPBands, EhlersDominantCycle,
    FractalDimension, Supertrend, MovingAverageRibbon,
)
from acms.indicators.patterns import CandlestickPatterns
from acms.indicators.support_resistance import (
    PivotPoints, FibonacciRetracement, SupportResistance,
)
from acms.indicators.statistics import (
    compute_hurst_exponent, compute_zscore,
    detect_bullish_divergence, detect_bearish_divergence,
)

__all__ = [
    # Moving Averages
    "SMA", "EMA", "WMA", "HMA", "DEMA", "TEMA", "VWMA", "KAMA", "ALMA",
    "FRAMA", "ZLEMA", "KaufmanAdaptiveMovingAverage", "EhlersSuperSmoother",
    # Oscillators
    "RSI", "ConnorsRSI", "MACD", "VolumeWeightedMACD", "StochasticOscillator",
    "CCI", "WilliamsR", "ROC", "Momentum", "TRIX", "UltimateOscillator",
    "MFI", "ADX", "Aroon", "ChandeMomentumOscillator", "EhlersFisherTransform",
    # Volatility
    "ATR", "BollingerBands", "KeltnerChannels", "DonchianChannels",
    "StandardDeviation", "HistoricalVolatility", "ParkinsonVolatility",
    "GarmanKlassVolatility", "ChaikinVolatility", "TrueRange", "ATRP",
    # Volume
    "OBVIndicator", "CMFIndicator", "ADLine", "VolumeProfile", "ForceIndex",
    "EaseOfMovement", "VolumeOscillator", "NVI", "PVI", "IchimokuCloud",
    # Advanced
    "TTMSqueeze", "VWAPIndicator", "VWAPBands", "EhlersDominantCycle",
    "FractalDimension", "Supertrend", "MovingAverageRibbon",
    # Patterns
    "CandlestickPatterns",
    # Support/Resistance
    "PivotPoints", "FibonacciRetracement", "SupportResistance",
    # Statistics
    "compute_hurst_exponent", "compute_zscore",
    "detect_bullish_divergence", "detect_bearish_divergence",
]
