"""Orchestrator state and degradation enums."""

from enum import Enum


class OrchestratorState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    DEGRADED = "degraded"
    CIRCUIT_BREAKER = "circuit_breaker"


class DegradationLevel(str, Enum):
    NONE = "none"
    REDUCE_POSITIONS = "reduce_positions"
    WIDEN_STOPS = "widen_stops"
    HALT_NEW_ORDERS = "halt_new_orders"
    FULL_HALT = "full_halt"


__all__ = [
    "OrchestratorState",
    "DegradationLevel",
]
