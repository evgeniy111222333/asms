"""Risk Engine - Comprehensive risk management.

Implements:
- 3 VaR methods (Historical, Parametric, Monte Carlo)
- CVaR (Conditional VaR / Expected Shortfall)
- Expected Shortfall with tail risk decomposition
- 8+ stress test scenarios including historical replays
- Historical scenario replay: COVID crash, FTX collapse, Luna crash, China ban
- Kill switch with propagation
- 10+ pre-trade risk checks
- Position sizing (Kelly with drawdown constraint, fixed-fractional, risk parity)
- Liquidity risk assessment: bid-ask spread widening, order book depth thinning
- Dynamic correlation risk monitoring: rolling correlation with eigenvalue decomposition
- Counterparty risk scoring: exchange risk assessment
- Portfolio heat map: marginal VaR, component VaR, percentage contribution
- Dynamic position sizing based on volatility regime
- Risk budgeting: allocate risk budget across strategies
- Circuit breaker integration
"""

from acms.risk.config import RiskConfig
from acms.risk.var import ValueAtRisk
from acms.risk.expected_shortfall import ExpectedShortfall
from acms.risk.stress_testing import StressTesting
from acms.risk.liquidity import LiquidityRiskAssessor
from acms.risk.correlation import CorrelationRiskMonitor
from acms.risk.counterparty import CounterpartyRiskScorer
from acms.risk.heatmap import PortfolioHeatMap
from acms.risk.circuit_breaker import CircuitBreaker
from acms.risk.budgeting import RiskBudgeting
from acms.risk.position_sizing import (
    kelly_size, fixed_fractional_size, volatility_regime_size, dynamic_position_size,
)
from acms.risk.engine import RiskEngine

__all__ = [
    # Config
    "RiskConfig",
    # Risk measures
    "ValueAtRisk", "ExpectedShortfall",
    # Stress testing
    "StressTesting",
    # Risk monitors
    "LiquidityRiskAssessor", "CorrelationRiskMonitor", "CounterpartyRiskScorer",
    # Risk visualization
    "PortfolioHeatMap",
    # Risk controls
    "CircuitBreaker", "RiskBudgeting",
    # Position sizing
    "kelly_size", "fixed_fractional_size", "volatility_regime_size", "dynamic_position_size",
    # Risk engine
    "RiskEngine",
]
