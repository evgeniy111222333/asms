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

from acms.orchestrator.state import OrchestratorState, DegradationLevel
from acms.orchestrator.position_sizer import PositionSizer
from acms.orchestrator.allocation import StrategyAllocationManager
from acms.orchestrator.performance import PerformanceMonitor, EquityCurveTracker
from acms.orchestrator.config import OrchestratorConfig
from acms.orchestrator.engine import Orchestrator

__all__ = [
    # State
    "OrchestratorState", "DegradationLevel",
    # Components
    "PositionSizer", "StrategyAllocationManager",
    "PerformanceMonitor", "EquityCurveTracker",
    # Config
    "OrchestratorConfig",
    # Engine
    "Orchestrator",
]
