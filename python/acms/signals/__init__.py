"""Signal Engine - Composite signal generation from multiple indicators.

Generates trading signals by combining multiple indicator readings
with configurable weights, confirmation logic, and advanced features:
- Multi-timeframe signal aggregation
- Bayesian confidence scoring
- Regime-aware signal weighting
- Signal persistence filter
- Adaptive weight adjustment
- Signal-to-noise ratio computation
- Dynamic threshold
- Signal divergence detection
"""

from acms.signals.config import SignalConfig, MultiTimeframeSignal
from acms.signals.engine import SignalEngine, SignalStrength
from acms.signals.bayesian import BayesianConfidenceTracker
from acms.signals.persistence import SignalPersistenceFilter
from acms.signals.divergence import DivergenceDetector
from acms.signals.regime import MarketRegime, RegimeDetector

__all__ = [
    # Config & data
    "SignalConfig", "MultiTimeframeSignal",
    # Engine
    "SignalEngine", "SignalStrength",
    # Components
    "BayesianConfidenceTracker",
    "SignalPersistenceFilter",
    "DivergenceDetector",
    "MarketRegime", "RegimeDetector",
]
