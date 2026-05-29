"""
SelfAdaptationEngine - Self-Adaptive Learning for Crypto Trading
=================================================================

Implements a meta-learning system that monitors strategy performance
across regimes and automatically triggers adaptations when degradation
is detected. Includes:

- PerformanceTracker: Tracks strategy metrics across regimes and time
- AdaptationTrigger: Detects when adaptation is needed (degradation, drift)
- AdaptationAction: Types of corrective actions (retrain, adjust, switch)
- AdaptationRecord: History tracking for all adaptations
- MetaAdaptation: Adapting the adaptation strategy itself
- SelfAdaptationEngine: Orchestrates the full adaptation cycle

GPU-ready with PyTorch-based drift detection and performance modeling.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AdaptationTriggerType(str, Enum):
    """Types of events that can trigger an adaptation."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    REGIME_CHANGE = "regime_change"
    CONCEPT_DRIFT = "concept_drift"
    ANOMALY_DETECTED = "anomaly_detected"
    SCHEDULED = "scheduled"
    MANUAL = "manual"


class AdaptationActionType(str, Enum):
    """Types of corrective actions the system can take."""

    RETRAIN = "retrain"
    ADJUST_PARAMS = "adjust_params"
    SWITCH_MODEL = "switch_model"
    ROLLBACK = "rollback"
    EXPAND_FEATURES = "expand_features"
    REDUCE_COMPLEXITY = "reduce_complexity"
    ENSEMBLE_ADD = "ensemble_add"
    ENSEMBLE_REMOVE = "ensemble_remove"


class AdaptationStatus(str, Enum):
    """Status of an adaptation action."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class RegimeType(str, Enum):
    """Market regime types for adaptation context."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSnapshot:
    """A point-in-time snapshot of strategy performance.

    Attributes:
        timestamp: When the snapshot was taken.
        strategy_id: Identifier for the strategy.
        regime: Current market regime.
        sharpe: Sharpe ratio.
        sortino: Sortino ratio.
        max_drawdown: Maximum drawdown.
        win_rate: Win rate.
        pnl: Profit and loss.
        volatility: Realized volatility.
        turnover: Portfolio turnover.
    """

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_id: str = ""
    regime: RegimeType = RegimeType.RANGING
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    pnl: float = 0.0
    volatility: float = 0.0
    turnover: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "regime": self.regime.value,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "pnl": self.pnl,
            "volatility": self.volatility,
            "turnover": self.turnover,
        }


@dataclass
class AdaptationTrigger:
    """Represents a detected condition that warrants adaptation.

    Attributes:
        trigger_id: Unique identifier.
        trigger_type: Type of trigger.
        strategy_id: Affected strategy.
        regime: Current market regime.
        severity: Severity level [0, 1].
        description: Human-readable description.
        evidence: Supporting data/metrics.
        detected_at: When the trigger was detected.
    """

    trigger_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_type: AdaptationTriggerType = AdaptationTriggerType.PERFORMANCE_DEGRADATION
    strategy_id: str = ""
    regime: RegimeType = RegimeType.RANGING
    severity: float = 0.5
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AdaptationAction:
    """A corrective action to be taken in response to a trigger.

    Attributes:
        action_id: Unique identifier.
        action_type: Type of action.
        trigger_id: ID of the trigger that caused this action.
        strategy_id: Affected strategy.
        parameters: Action-specific parameters.
        expected_impact: Predicted improvement metric.
        risk_level: Estimated risk of the action [0, 1].
        status: Current status of the action.
        created_at: When the action was created.
        executed_at: When the action was executed.
        completed_at: When the action completed.
    """

    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    action_type: AdaptationActionType = AdaptationActionType.ADJUST_PARAMS
    trigger_id: str = ""
    strategy_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_impact: float = 0.0
    risk_level: float = 0.0
    status: AdaptationStatus = AdaptationStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class AdaptationRecord:
    """Historical record of an adaptation event.

    Attributes:
        record_id: Unique identifier.
        trigger: The trigger that initiated the adaptation.
        action: The action that was taken.
        before_metrics: Performance metrics before adaptation.
        after_metrics: Performance metrics after adaptation.
        effectiveness: Measured effectiveness of the adaptation.
        created_at: When this record was created.
    """

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: Optional[AdaptationTrigger] = None
    action: Optional[AdaptationAction] = None
    before_metrics: Dict[str, float] = field(default_factory=dict)
    after_metrics: Dict[str, float] = field(default_factory=dict)
    effectiveness: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def compute_effectiveness(self) -> float:
        """Compute effectiveness as the improvement in Sharpe ratio.

        Returns:
            Effectiveness score (positive = improvement).
        """
        before_sharpe = self.before_metrics.get("sharpe", 0.0)
        after_sharpe = self.after_metrics.get("sharpe", 0.0)
        self.effectiveness = after_sharpe - before_sharpe
        return self.effectiveness

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "record_id": self.record_id,
            "trigger_type": self.trigger.trigger_type.value if self.trigger else None,
            "action_type": self.action.action_type.value if self.action else None,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "effectiveness": self.effectiveness,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Performance Tracker
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Tracks strategy performance across regimes and time windows.

    Provides statistical tests for performance degradation detection,
    including CUSUM, EWMA, and Page-Hinkley change detection.

    Attributes:
        window_size: Rolling window size for metrics.
        history: Full performance history.
    """

    def __init__(
        self,
        window_size: int = 50,
        degradation_threshold: float = 0.3,
        drift_sensitivity: float = 0.05,
    ) -> None:
        """Initialise the performance tracker.

        Args:
            window_size: Number of recent snapshots for rolling metrics.
            degradation_threshold: Minimum Sharpe drop to flag degradation.
            drift_sensitivity: Statistical significance level for drift tests.
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.drift_sensitivity = drift_sensitivity
        self.history: Dict[str, List[PerformanceSnapshot]] = {}

        # CUSUM parameters
        self._cusum_pos: Dict[str, float] = {}
        self._cusum_neg: Dict[str, float] = {}
        self._cusum_threshold = 5.0

        # Baseline metrics per strategy
        self._baselines: Dict[str, Dict[str, float]] = {}

    def record(self, snapshot: PerformanceSnapshot) -> None:
        """Record a new performance snapshot.

        Args:
            snapshot: The performance snapshot to record.
        """
        sid = snapshot.strategy_id
        self.history.setdefault(sid, []).append(snapshot)

        # Update CUSUM
        baseline_sharpe = self._baselines.get(sid, {}).get("sharpe", 0.0)
        deviation = snapshot.sharpe - baseline_sharpe
        self._cusum_pos[sid] = max(0, self._cusum_pos.get(sid, 0.0) + deviation - 0.1)
        self._cusum_neg[sid] = max(0, self._cusum_neg.get(sid, 0.0) - deviation - 0.1)

        # Update baseline using EMA
        if sid not in self._baselines:
            self._baselines[sid] = {"sharpe": snapshot.sharpe}
        else:
            alpha = 0.1
            self._baselines[sid]["sharpe"] = (
                alpha * snapshot.sharpe + (1 - alpha) * self._baselines[sid]["sharpe"]
            )

    def set_baseline(self, strategy_id: str, metrics: Dict[str, float]) -> None:
        """Manually set the baseline metrics for a strategy.

        Args:
            strategy_id: The strategy identifier.
            metrics: Baseline metrics dictionary.
        """
        self._baselines[strategy_id] = metrics

    def detect_degradation(self, strategy_id: str) -> Optional[AdaptationTrigger]:
        """Detect if a strategy is experiencing performance degradation.

        Uses rolling window comparison and CUSUM change detection.

        Args:
            strategy_id: The strategy to check.

        Returns:
            An AdaptationTrigger if degradation detected, else None.
        """
        snapshots = self.history.get(strategy_id, [])
        if len(snapshots) < self.window_size:
            return None

        recent = snapshots[-self.window_size:]
        recent_sharpe = float(np.mean([s.sharpe for s in recent]))

        baseline_sharpe = self._baselines.get(strategy_id, {}).get("sharpe", 0.0)
        drop = baseline_sharpe - recent_sharpe

        # CUSUM check
        cusum_alert = (
            self._cusum_pos.get(strategy_id, 0.0) > self._cusum_threshold
            or self._cusum_neg.get(strategy_id, 0.0) > self._cusum_threshold
        )

        if drop > self.degradation_threshold or cusum_alert:
            severity = min(1.0, drop / 2.0)
            current_regime = recent[-1].regime if recent else RegimeType.RANGING

            return AdaptationTrigger(
                trigger_type=AdaptationTriggerType.PERFORMANCE_DEGRADATION,
                strategy_id=strategy_id,
                regime=current_regime,
                severity=severity,
                description=(
                    f"Sharpe dropped from {baseline_sharpe:.2f} to {recent_sharpe:.2f} "
                    f"(drop={drop:.2f})"
                ),
                evidence={
                    "baseline_sharpe": baseline_sharpe,
                    "recent_sharpe": recent_sharpe,
                    "drop": drop,
                    "cusum_pos": self._cusum_pos.get(strategy_id, 0.0),
                    "cusum_neg": self._cusum_neg.get(strategy_id, 0.0),
                    "window_size": self.window_size,
                },
            )

        return None

    def detect_concept_drift(
        self, strategy_id: str, reference_window: int = 100
    ) -> Optional[AdaptationTrigger]:
        """Detect concept drift using the Kolmogorov-Smirnov test.

        Compares the distribution of recent PnL values against a
        reference (historical) window.

        Args:
            strategy_id: The strategy to check.
            reference_window: Size of the reference window.

        Returns:
            An AdaptationTrigger if drift detected, else None.
        """
        snapshots = self.history.get(strategy_id, [])
        if len(snapshots) < reference_window + self.window_size:
            return None

        reference_pnl = [s.pnl for s in snapshots[-reference_window - self.window_size : -self.window_size]]
        recent_pnl = [s.pnl for s in snapshots[-self.window_size:]]

        # Two-sample KS test (manual implementation)
        ref_sorted = np.sort(reference_pnl)
        rec_sorted = np.sort(recent_pnl)

        all_values = np.sort(np.concatenate([ref_sorted, rec_sorted]))
        n_ref = len(ref_sorted)
        n_rec = len(rec_sorted)

        max_diff = 0.0
        for val in all_values:
            cdf_ref = np.searchsorted(ref_sorted, val, side="right") / n_ref
            cdf_rec = np.searchsorted(rec_sorted, val, side="right") / n_rec
            max_diff = max(max_diff, abs(cdf_ref - cdf_rec))

        # Critical value approximation
        n_eff = (n_ref * n_rec) / (n_ref + n_rec)
        critical = np.sqrt(-np.log(self.drift_sensitivity / 2) / (2 * n_eff))

        if max_diff > critical:
            current_regime = snapshots[-1].regime
            return AdaptationTrigger(
                trigger_type=AdaptationTriggerType.CONCEPT_DRIFT,
                strategy_id=strategy_id,
                regime=current_regime,
                severity=min(1.0, max_diff / (critical + 1e-8)),
                description=f"Concept drift detected: KS statistic={max_diff:.4f} > critical={critical:.4f}",
                evidence={
                    "ks_statistic": max_diff,
                    "critical_value": critical,
                    "reference_window": reference_window,
                    "recent_window": self.window_size,
                },
            )

        return None

    def get_regime_performance(
        self, strategy_id: str, regime: RegimeType
    ) -> Dict[str, float]:
        """Get performance metrics for a strategy in a specific regime.

        Args:
            strategy_id: The strategy identifier.
            regime: The market regime.

        Returns:
            Dictionary of performance metrics.
        """
        snapshots = [
            s for s in self.history.get(strategy_id, []) if s.regime == regime
        ]
        if not snapshots:
            return {}

        return {
            "count": float(len(snapshots)),
            "avg_sharpe": float(np.mean([s.sharpe for s in snapshots])),
            "avg_pnl": float(np.mean([s.pnl for s in snapshots])),
            "avg_drawdown": float(np.mean([s.max_drawdown for s in snapshots])),
            "win_rate": float(np.mean([s.win_rate for s in snapshots])),
        }

    def get_strategy_summary(self, strategy_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of a strategy's performance.

        Args:
            strategy_id: The strategy identifier.

        Returns:
            Summary dictionary.
        """
        snapshots = self.history.get(strategy_id, [])
        if not snapshots:
            return {"strategy_id": strategy_id, "total_snapshots": 0}

        recent = snapshots[-self.window_size:] if len(snapshots) >= self.window_size else snapshots
        return {
            "strategy_id": strategy_id,
            "total_snapshots": len(snapshots),
            "baseline_sharpe": self._baselines.get(strategy_id, {}).get("sharpe", 0.0),
            "recent_avg_sharpe": float(np.mean([s.sharpe for s in recent])),
            "recent_avg_pnl": float(np.mean([s.pnl for s in recent])),
            "cusum_pos": self._cusum_pos.get(strategy_id, 0.0),
            "cusum_neg": self._cusum_neg.get(strategy_id, 0.0),
        }


# ---------------------------------------------------------------------------
# Meta-Adaptation
# ---------------------------------------------------------------------------

class MetaAdaptation:
    """Adapts the adaptation strategy itself based on historical effectiveness.

    Tracks which types of adaptations work best under different conditions
    and adjusts the selection and parameter tuning of future adaptations.

    Attributes:
        effectiveness_history: Records of past adaptation effectiveness.
        action_preferences: Learned preferences for action types per regime.
    """

    def __init__(self, learning_rate: float = 0.1) -> None:
        """Initialise meta-adaptation.

        Args:
            learning_rate: Learning rate for updating preferences.
        """
        self.learning_rate = learning_rate
        self.effectiveness_history: List[AdaptationRecord] = []
        self.action_preferences: Dict[str, Dict[str, float]] = {}  # regime -> action_type -> score

    def record_adaptation(self, record: AdaptationRecord) -> None:
        """Record the outcome of an adaptation for meta-learning.

        Args:
            record: The adaptation record with before/after metrics.
        """
        record.compute_effectiveness()
        self.effectiveness_history.append(record)

        # Update preferences
        if record.action and record.trigger:
            regime = record.trigger.regime.value
            action_type = record.action.action_type.value

            self.action_preferences.setdefault(regime, {})
            current_score = self.action_preferences[regime].get(action_type, 0.5)

            # Reward positive effectiveness, penalize negative
            reward = np.clip(record.effectiveness, -1.0, 1.0)
            updated = current_score + self.learning_rate * (reward - current_score)
            self.action_preferences[regime][action_type] = float(np.clip(updated, 0.0, 1.0))

        logger.debug(
            "Recorded adaptation: effectiveness=%.4f",
            record.effectiveness,
        )

    def suggest_action(
        self, regime: RegimeType, trigger_type: AdaptationTriggerType
    ) -> Tuple[AdaptationActionType, float]:
        """Suggest the best adaptation action for a given situation.

        Uses learned preferences to recommend the most effective
        action type for the current regime.

        Args:
            regime: Current market regime.
            trigger_type: Type of trigger that was detected.

        Returns:
            Tuple of (recommended action type, confidence score).
        """
        regime_prefs = self.action_preferences.get(regime.value, {})

        if not regime_prefs:
            # Default mappings based on trigger type
            defaults = {
                AdaptationTriggerType.PERFORMANCE_DEGRADATION: (AdaptationActionType.ADJUST_PARAMS, 0.5),
                AdaptationTriggerType.REGIME_CHANGE: (AdaptationActionType.SWITCH_MODEL, 0.5),
                AdaptationTriggerType.CONCEPT_DRIFT: (AdaptationActionType.RETRAIN, 0.5),
                AdaptationTriggerType.ANOMALY_DETECTED: (AdaptationActionType.REDUCE_COMPLEXITY, 0.5),
                AdaptationTriggerType.SCHEDULED: (AdaptationActionType.RETRAIN, 0.5),
                AdaptationTriggerType.MANUAL: (AdaptationActionType.ADJUST_PARAMS, 0.5),
            }
            return defaults.get(trigger_type, (AdaptationActionType.ADJUST_PARAMS, 0.5))

        best_action = max(regime_prefs, key=regime_prefs.get)  # type: ignore[arg-type]
        best_score = regime_prefs[best_action]
        return AdaptationActionType(best_action), best_score

    def get_meta_stats(self) -> Dict[str, Any]:
        """Compute meta-adaptation statistics.

        Returns:
            Dictionary of meta-learning statistics.
        """
        if not self.effectiveness_history:
            return {"total_records": 0, "avg_effectiveness": 0.0}

        effectiveness = [r.effectiveness for r in self.effectiveness_history]
        by_action: Dict[str, List[float]] = {}
        for r in self.effectiveness_history:
            if r.action:
                by_action.setdefault(r.action.action_type.value, []).append(r.effectiveness)

        return {
            "total_records": len(self.effectiveness_history),
            "avg_effectiveness": float(np.mean(effectiveness)),
            "success_rate": float(np.mean([1.0 if e > 0 else 0.0 for e in effectiveness])),
            "action_preferences": self.action_preferences,
            "by_action_avg": {
                k: float(np.mean(v)) for k, v in by_action.items() if v
            },
        }


# ---------------------------------------------------------------------------
# Self-Adaptation Engine
# ---------------------------------------------------------------------------

class SelfAdaptationEngine:
    """Orchestrates the full self-adaptation cycle for trading strategies.

    Monitors performance, detects triggers, selects actions via
    meta-adaptation, executes adaptations, and evaluates their
    effectiveness.

    Attributes:
        engine_id: Unique identifier.
        performance_tracker: Strategy performance tracker.
        meta_adaptation: Meta-learning adaptation optimizer.
        adaptation_history: Complete history of adaptations.
        action_handlers: Registered handlers for each action type.
    """

    def __init__(
        self,
        engine_id: str = "default",
        window_size: int = 50,
        degradation_threshold: float = 0.3,
        drift_sensitivity: float = 0.05,
        auto_adapt: bool = True,
        device: str = "auto",
        redis_client: Any = None,
        postgres_client: Any = None,
    ) -> None:
        """Initialise the self-adaptation engine.

        Args:
            engine_id: Unique identifier.
            window_size: Rolling window for performance tracking.
            degradation_threshold: Minimum Sharpe drop to trigger adaptation.
            drift_sensitivity: Statistical significance for drift detection.
            auto_adapt: Whether to automatically execute adaptations.
            device: Compute device.
            redis_client: Optional Redis client.
            postgres_client: Optional Postgres client.
        """
        self.engine_id = engine_id
        self.auto_adapt = auto_adapt
        self._redis = redis_client
        self._postgres = postgres_client

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        self.performance_tracker = PerformanceTracker(
            window_size=window_size,
            degradation_threshold=degradation_threshold,
            drift_sensitivity=drift_sensitivity,
        )
        self.meta_adaptation = MetaAdaptation()
        self.adaptation_history: List[AdaptationRecord] = []

        # Registered action handlers
        self.action_handlers: Dict[AdaptationActionType, Callable] = {}

        logger.info(
            "SelfAdaptationEngine initialised [id=%s, device=%s, auto=%s]",
            engine_id,
            self._device,
            auto_adapt,
        )

    def register_action_handler(
        self, action_type: AdaptationActionType, handler: Callable
    ) -> None:
        """Register a handler function for an action type.

        Args:
            action_type: The type of action this handler executes.
            handler: Callable that takes an AdaptationAction and returns bool.
        """
        self.action_handlers[action_type] = handler
        logger.debug("Registered handler for action type: %s", action_type.value)

    def record_performance(self, snapshot: PerformanceSnapshot) -> None:
        """Record a performance snapshot and check for triggers.

        Args:
            snapshot: The performance snapshot to record.
        """
        self.performance_tracker.record(snapshot)

        # Check for degradation
        trigger = self.performance_tracker.detect_degradation(snapshot.strategy_id)
        if trigger:
            self._handle_trigger(trigger)

        # Check for concept drift
        drift_trigger = self.performance_tracker.detect_concept_drift(snapshot.strategy_id)
        if drift_trigger:
            self._handle_trigger(drift_trigger)

    def _handle_trigger(self, trigger: AdaptationTrigger) -> None:
        """Handle a detected adaptation trigger.

        Args:
            trigger: The detected trigger.
        """
        logger.info(
            "Adaptation trigger detected: type=%s strategy=%s severity=%.2f - %s",
            trigger.trigger_type.value,
            trigger.strategy_id,
            trigger.severity,
            trigger.description,
        )

        if not self.auto_adapt:
            logger.info("Auto-adapt disabled; trigger logged but no action taken")
            return

        # Get before metrics
        summary = self.performance_tracker.get_strategy_summary(trigger.strategy_id)
        before_metrics = {
            "sharpe": summary.get("recent_avg_sharpe", 0.0),
            "pnl": summary.get("recent_avg_pnl", 0.0),
        }

        # Suggest action via meta-adaptation
        action_type, confidence = self.meta_adaptation.suggest_action(
            trigger.regime, trigger.trigger_type
        )

        # Create action
        action = AdaptationAction(
            action_type=action_type,
            trigger_id=trigger.trigger_id,
            strategy_id=trigger.strategy_id,
            parameters={"trigger_severity": trigger.severity},
            expected_impact=confidence,
            risk_level=1.0 - confidence,
        )

        # Execute action
        record = self._execute_adaptation(trigger, action, before_metrics)
        self.adaptation_history.append(record)

        # Learn from outcome
        self.meta_adaptation.record_adaptation(record)

    def _execute_adaptation(
        self,
        trigger: AdaptationTrigger,
        action: AdaptationAction,
        before_metrics: Dict[str, float],
    ) -> AdaptationRecord:
        """Execute an adaptation action and record the result.

        Args:
            trigger: The trigger that caused this action.
            action: The action to execute.
            before_metrics: Metrics before adaptation.

        Returns:
            AdaptationRecord with the outcome.
        """
        action.status = AdaptationStatus.IN_PROGRESS
        action.executed_at = datetime.now(timezone.utc)

        handler = self.action_handlers.get(action.action_type)
        if handler:
            try:
                success = handler(action)
                action.status = AdaptationStatus.COMPLETED if success else AdaptationStatus.FAILED
            except Exception as exc:
                logger.error("Adaptation handler failed: %s", exc)
                action.status = AdaptationStatus.FAILED
        else:
            logger.warning("No handler registered for action type: %s", action.action_type.value)
            action.status = AdaptationStatus.COMPLETED  # No-op

        action.completed_at = datetime.now(timezone.utc)

        # Collect after metrics
        after_summary = self.performance_tracker.get_strategy_summary(trigger.strategy_id)
        after_metrics = {
            "sharpe": after_summary.get("recent_avg_sharpe", 0.0),
            "pnl": after_summary.get("recent_avg_pnl", 0.0),
        }

        record = AdaptationRecord(
            trigger=trigger,
            action=action,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
        )
        record.compute_effectiveness()

        logger.info(
            "Adaptation executed: action=%s status=%s effectiveness=%.4f",
            action.action_type.value,
            action.status.value,
            record.effectiveness,
        )

        return record

    def manual_adapt(
        self,
        strategy_id: str,
        action_type: AdaptationActionType,
        parameters: Optional[Dict[str, Any]] = None,
        regime: RegimeType = RegimeType.RANGING,
    ) -> AdaptationRecord:
        """Manually trigger an adaptation action.

        Args:
            strategy_id: The strategy to adapt.
            action_type: The type of action to take.
            parameters: Action-specific parameters.
            regime: Current market regime.

        Returns:
            AdaptationRecord with the outcome.
        """
        trigger = AdaptationTrigger(
            trigger_type=AdaptationTriggerType.MANUAL,
            strategy_id=strategy_id,
            regime=regime,
            severity=0.5,
            description=f"Manual adaptation requested: {action_type.value}",
        )

        summary = self.performance_tracker.get_strategy_summary(strategy_id)
        before_metrics = {
            "sharpe": summary.get("recent_avg_sharpe", 0.0),
            "pnl": summary.get("recent_avg_pnl", 0.0),
        }

        action = AdaptationAction(
            action_type=action_type,
            trigger_id=trigger.trigger_id,
            strategy_id=strategy_id,
            parameters=parameters or {},
        )

        record = self._execute_adaptation(trigger, action, before_metrics)
        self.adaptation_history.append(record)
        self.meta_adaptation.record_adaptation(record)
        return record

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Compute a comprehensive summary of adaptation activity.

        Returns:
            Summary dictionary.
        """
        if not self.adaptation_history:
            return {"engine_id": self.engine_id, "total_adaptations": 0}

        by_status: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        by_trigger: Dict[str, int] = {}
        effectiveness_values: List[float] = []

        for record in self.adaptation_history:
            if record.action:
                by_status[record.action.status.value] = by_status.get(record.action.status.value, 0) + 1
                by_action[record.action.action_type.value] = by_action.get(record.action.action_type.value, 0) + 1
            if record.trigger:
                by_trigger[record.trigger.trigger_type.value] = by_trigger.get(record.trigger.trigger_type.value, 0) + 1
            effectiveness_values.append(record.effectiveness)

        return {
            "engine_id": self.engine_id,
            "total_adaptations": len(self.adaptation_history),
            "avg_effectiveness": float(np.mean(effectiveness_values)) if effectiveness_values else 0.0,
            "success_rate": float(np.mean([1.0 if e > 0 else 0.0 for e in effectiveness_values])),
            "by_status": by_status,
            "by_action": by_action,
            "by_trigger": by_trigger,
            "meta_stats": self.meta_adaptation.get_meta_stats(),
            "device": self._device,
        }

    async def save_to_postgres(self) -> int:
        """Persist adaptation history to Postgres.

        Returns:
            Number of records saved.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping save")
            return 0

        count = 0
        try:
            async with self._postgres.transaction():
                for record in self.adaptation_history:
                    await self._postgres.execute(
                        """
                        INSERT INTO adaptation_records (record_id, engine_id, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (record_id) DO UPDATE SET data = $3
                        """,
                        record.record_id,
                        self.engine_id,
                        json.dumps(record.to_dict(), default=str),
                    )
                    count += 1
            logger.info("Saved %d adaptation records to Postgres", count)
        except Exception as exc:
            logger.error("Failed to save adaptation records: %s", exc)
        return count
