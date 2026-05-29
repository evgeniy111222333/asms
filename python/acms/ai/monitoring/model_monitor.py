"""Advanced Model Monitoring for ACMS AI models.

Implements:
- AIModelMonitor: Extends the base ModelMonitor with degradation detection,
  performance regression testing, model comparison, and alert generation.
- PredictionAccuracyTracker: Per-model, per-regime accuracy tracking with
  rolling windows and statistical significance tests.
- ModelHealthDashboard: Generates dashboard-ready data for model health.
- CalibrationMonitor: Tracks prediction calibration over time.
- DegradationAlert: Alert generation for model issues.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from acms.ml import ModelMonitor

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels for model monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ModelStatus(str, Enum):
    """Model health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DegradationAlert:
    """Alert generated when model degradation is detected.

    Attributes:
        model_id: Identifier of the affected model.
        alert_type: Type of degradation (e.g., 'accuracy_drop', 'drift_detected').
        severity: Alert severity level.
        message: Human-readable alert message.
        metric_value: Current value of the degraded metric.
        threshold: Threshold that was breached.
        timestamp: When the alert was generated.
        regime: Market regime at the time of alert (if applicable).
    """
    model_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    regime: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize alert to dictionary."""
        return {
            "model_id": self.model_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime,
        }


@dataclass
class AccuracyRecord:
    """A single accuracy measurement with context.

    Attributes:
        timestamp: When the measurement was taken.
        accuracy: Accuracy value [0, 1].
        regime: Market regime label.
        n_samples: Number of samples used for measurement.
        model_version: Version identifier of the model.
    """
    timestamp: datetime
    accuracy: float
    regime: str = "default"
    n_samples: int = 0
    model_version: str = ""


@dataclass
class CalibrationRecord:
    """A single calibration measurement.

    Attributes:
        timestamp: When the measurement was taken.
        expected_accuracy: Predicted probability (confidence).
        actual_accuracy: Observed accuracy at that confidence level.
        n_bins: Number of calibration bins used.
        ece: Expected Calibration Error.
    """
    timestamp: datetime
    expected_accuracy: float
    actual_accuracy: float
    n_bins: int = 10
    ece: float = 0.0


# ============================================================================
# Prediction Accuracy Tracker
# ============================================================================

class PredictionAccuracyTracker:
    """Per-model, per-regime accuracy tracking with rolling windows.

    Tracks prediction accuracy over time for each model and market regime,
    supporting statistical significance tests for accuracy changes and
    automated alerting when accuracy degrades beyond configurable thresholds.
    """

    def __init__(self, window_size: int = 1000, degradation_threshold: float = 0.05,
                 min_samples: int = 50, regime_labels: Optional[List[str]] = None):
        """Initialize the accuracy tracker.

        Args:
            window_size: Rolling window size for accuracy computation.
            degradation_threshold: Relative accuracy drop to trigger alerts.
            min_samples: Minimum samples before alerting.
            regime_labels: Known regime labels for per-regime tracking.
        """
        self.window_size = window_size
        self.degradation_threshold = degradation_threshold
        self.min_samples = min_samples
        self.regime_labels = regime_labels or ["default"]

        # Per-model, per-regime tracking
        # Structure: {model_id: {regime: deque of (correct, total)}}
        self._predictions: Dict[str, Dict[str, Deque[Tuple[bool, int]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )
        # Baseline accuracies: {model_id: {regime: baseline_accuracy}}
        self._baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        # Accuracy history for charting: {model_id: List[AccuracyRecord]}
        self._history: Dict[str, List[AccuracyRecord]] = defaultdict(list)

    def set_baseline(self, model_id: str, regime: str, accuracy: float) -> None:
        """Set baseline accuracy for a model/regime combination.

        Args:
            model_id: Model identifier.
            regime: Market regime label.
            accuracy: Baseline accuracy value.
        """
        self._baselines[model_id][regime] = accuracy
        logger.info("Baseline set for model '%s' regime '%s': %.4f", model_id, regime, accuracy)

    def record_prediction(self, model_id: str, predicted: Any, actual: Any,
                          regime: str = "default", model_version: str = "") -> None:
        """Record a single prediction result.

        Args:
            model_id: Model identifier.
            predicted: Model prediction.
            actual: Ground truth value.
            regime: Market regime at prediction time.
            model_version: Model version identifier.
        """
        correct = predicted == actual
        self._predictions[model_id][regime].append((correct, 1))

        # Record to history periodically
        window = self._predictions[model_id][regime]
        if len(window) % 100 == 0 and len(window) >= self.min_samples:
            accuracy = self.get_accuracy(model_id, regime)
            if accuracy is not None:
                self._history[model_id].append(AccuracyRecord(
                    timestamp=datetime.utcnow(),
                    accuracy=accuracy,
                    regime=regime,
                    n_samples=len(window),
                    model_version=model_version,
                ))

    def record_batch(self, model_id: str, predictions: np.ndarray,
                     actuals: np.ndarray, regime: str = "default",
                     model_version: str = "") -> Dict[str, Any]:
        """Record a batch of predictions.

        Args:
            model_id: Model identifier.
            predictions: Array of model predictions.
            actuals: Array of ground truth values.
            regime: Market regime.
            model_version: Model version identifier.

        Returns:
            Dict with batch accuracy and alert status.
        """
        if len(predictions) != len(actuals):
            logger.warning("Prediction/actual length mismatch for model '%s'", model_id)
            return {"accuracy": 0.0, "alert": False}

        correct_mask = predictions == actuals
        for correct in correct_mask:
            self._predictions[model_id][regime].append((bool(correct), 1))

        batch_accuracy = float(np.mean(correct_mask))
        alert = self.check_degradation(model_id, regime)

        self._history[model_id].append(AccuracyRecord(
            timestamp=datetime.utcnow(),
            accuracy=batch_accuracy,
            regime=regime,
            n_samples=len(predictions),
            model_version=model_version,
        ))

        return {"accuracy": batch_accuracy, "alert": alert, "n_samples": len(predictions)}

    def get_accuracy(self, model_id: str, regime: str = "default") -> Optional[float]:
        """Get current rolling accuracy for a model/regime.

        Args:
            model_id: Model identifier.
            regime: Market regime label.

        Returns:
            Rolling accuracy or None if insufficient data.
        """
        window = self._predictions[model_id].get(regime, deque())
        if len(window) < self.min_samples:
            return None
        correct = sum(1 for c, _ in window if c)
        total = len(window)
        return correct / total if total > 0 else None

    def get_all_accuracies(self, model_id: str) -> Dict[str, Optional[float]]:
        """Get accuracies across all regimes for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Dict mapping regime to accuracy value.
        """
        result = {}
        for regime in self._predictions.get(model_id, {}):
            result[regime] = self.get_accuracy(model_id, regime)
        return result

    def check_degradation(self, model_id: str, regime: str = "default") -> bool:
        """Check if model accuracy has degraded beyond threshold.

        Compares current accuracy against baseline. If no baseline is set,
        uses the first recorded accuracy as an implicit baseline.

        Args:
            model_id: Model identifier.
            regime: Market regime label.

        Returns:
            True if degradation detected.
        """
        current = self.get_accuracy(model_id, regime)
        if current is None:
            return False

        baseline = self._baselines.get(model_id, {}).get(regime)
        if baseline is None:
            # Use first history entry as baseline
            history = self._history.get(model_id, [])
            if history:
                baseline = history[0].accuracy
                self._baselines[model_id][regime] = baseline
            else:
                return False

        relative_drop = (baseline - current) / baseline if baseline > 0 else 0.0
        return relative_drop > self.degradation_threshold

    def get_regression_test_result(self, model_id: str, regime: str = "default",
                                   confidence: float = 0.95) -> Dict[str, Any]:
        """Statistical test for accuracy regression.

        Uses a two-proportion z-test to determine if the current
        accuracy is significantly lower than baseline.

        Args:
            model_id: Model identifier.
            regime: Market regime label.
            confidence: Confidence level for the test.

        Returns:
            Dict with test results including p-value and conclusion.
        """
        current = self.get_accuracy(model_id, regime)
        baseline = self._baselines.get(model_id, {}).get(regime)

        if current is None or baseline is None:
            return {"test": "z-test", "conclusion": "insufficient_data"}

        window = self._predictions[model_id][regime]
        n = len(window)
        n_correct = sum(1 for c, _ in window if c)

        # Two-proportion z-test
        p1, p2 = baseline, current
        n1 = max(n, self.min_samples)  # Baseline sample estimate
        n2 = n

        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
        z_score = (p1 - p2) / se if se > 0 else 0.0

        from scipy.stats import norm
        p_value = 1 - norm.cdf(z_score)

        alpha = 1 - confidence
        is_regression = p_value < alpha and current < baseline

        return {
            "test": "two_proportion_z_test",
            "baseline_accuracy": baseline,
            "current_accuracy": current,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "is_regression": is_regression,
            "confidence": confidence,
            "conclusion": "regression_detected" if is_regression else "no_regression",
        }

    def get_history(self, model_id: str, regime: Optional[str] = None,
                    limit: int = 100) -> List[Dict[str, Any]]:
        """Get accuracy history for a model.

        Args:
            model_id: Model identifier.
            regime: Optional regime filter.
            limit: Maximum number of records to return.

        Returns:
            List of accuracy record dicts.
        """
        history = self._history.get(model_id, [])
        if regime:
            history = [r for r in history if r.regime == regime]
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "accuracy": r.accuracy,
                "regime": r.regime,
                "n_samples": r.n_samples,
                "model_version": r.model_version,
            }
            for r in history[-limit:]
        ]


# ============================================================================
# Calibration Monitor
# ============================================================================

class CalibrationMonitor:
    """Monitors prediction calibration over time.

    Tracks how well predicted probabilities match actual outcomes
    using Expected Calibration Error (ECE) and reliability diagrams.
    """

    def __init__(self, n_bins: int = 10, window_size: int = 5000,
                 ece_warning_threshold: float = 0.1,
                 ece_critical_threshold: float = 0.2):
        """Initialize the calibration monitor.

        Args:
            n_bins: Number of bins for calibration computation.
            window_size: Rolling window for calibration tracking.
            ece_warning_threshold: ECE value triggering warning alert.
            ece_critical_threshold: ECE value triggering critical alert.
        """
        self.n_bins = n_bins
        self.window_size = window_size
        self.ece_warning_threshold = ece_warning_threshold
        self.ece_critical_threshold = ece_critical_threshold

        # Per-model: {model_id: deque of (predicted_prob, correct)}
        self._records: Dict[str, Deque[Tuple[float, bool]]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        # Per-model calibration history
        self._calibration_history: Dict[str, List[CalibrationRecord]] = defaultdict(list)

    def record(self, model_id: str, predicted_prob: float, correct: bool) -> None:
        """Record a single calibration data point.

        Args:
            model_id: Model identifier.
            predicted_prob: Predicted probability (confidence).
            correct: Whether the prediction was correct.
        """
        self._records[model_id].append((predicted_prob, correct))

    def compute_ece(self, model_id: str) -> Optional[float]:
        """Compute Expected Calibration Error for a model.

        ECE = sum(bins) (n_b / N) * |acc_b - conf_b|

        Args:
            model_id: Model identifier.

        Returns:
            ECE value or None if insufficient data.
        """
        records = self._records.get(model_id, deque())
        if len(records) < 50:
            return None

        probs = np.array([r[0] for r in records])
        correct = np.array([r[1] for r in records])

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        n_total = len(probs)

        for i in range(self.n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            n_bin = np.sum(mask)
            if n_bin > 0:
                bin_accuracy = np.mean(correct[mask])
                bin_confidence = np.mean(probs[mask])
                ece += (n_bin / n_total) * abs(bin_accuracy - bin_confidence)

        # Record calibration history
        self._calibration_history[model_id].append(CalibrationRecord(
            timestamp=datetime.utcnow(),
            expected_accuracy=float(np.mean(probs)),
            actual_accuracy=float(np.mean(correct)),
            n_bins=self.n_bins,
            ece=float(ece),
        ))

        return float(ece)

    def get_reliability_data(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Generate reliability diagram data for visualization.

        Args:
            model_id: Model identifier.

        Returns:
            Dict with bin accuracies, confidences, and counts, or None.
        """
        records = self._records.get(model_id, deque())
        if len(records) < 50:
            return None

        probs = np.array([r[0] for r in records])
        correct = np.array([r[1] for r in records])

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bins_data = []

        for i in range(self.n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            n_bin = int(np.sum(mask))
            if n_bin > 0:
                bins_data.append({
                    "bin_lower": float(bin_boundaries[i]),
                    "bin_upper": float(bin_boundaries[i + 1]),
                    "accuracy": float(np.mean(correct[mask])),
                    "confidence": float(np.mean(probs[mask])),
                    "count": n_bin,
                })
            else:
                bins_data.append({
                    "bin_lower": float(bin_boundaries[i]),
                    "bin_upper": float(bin_boundaries[i + 1]),
                    "accuracy": 0.0,
                    "confidence": 0.0,
                    "count": 0,
                })

        ece = self.compute_ece(model_id)
        return {"bins": bins_data, "ece": ece, "n_samples": len(records)}

    def check_calibration_alert(self, model_id: str) -> Optional[DegradationAlert]:
        """Check if calibration has degraded enough to trigger an alert.

        Args:
            model_id: Model identifier.

        Returns:
            DegradationAlert if calibration is poor, None otherwise.
        """
        ece = self.compute_ece(model_id)
        if ece is None:
            return None

        if ece > self.ece_critical_threshold:
            return DegradationAlert(
                model_id=model_id,
                alert_type="calibration_critical",
                severity=AlertSeverity.CRITICAL,
                message=f"Model '{model_id}' ECE={ece:.4f} exceeds critical threshold "
                        f"({self.ece_critical_threshold})",
                metric_value=ece,
                threshold=self.ece_critical_threshold,
            )
        elif ece > self.ece_warning_threshold:
            return DegradationAlert(
                model_id=model_id,
                alert_type="calibration_warning",
                severity=AlertSeverity.WARNING,
                message=f"Model '{model_id}' ECE={ece:.4f} exceeds warning threshold "
                        f"({self.ece_warning_threshold})",
                metric_value=ece,
                threshold=self.ece_warning_threshold,
            )
        return None


# ============================================================================
# AI Model Monitor
# ============================================================================

class AIModelMonitor(ModelMonitor):
    """Advanced model monitoring extending the base ModelMonitor.

    Adds:
    - Automated model degradation detection with alerts
    - Performance regression testing
    - Model comparison over time
    - Prediction calibration monitoring
    - Per-model, per-regime accuracy tracking
    - Model health status reporting
    """

    def __init__(self, reference_features: Optional[np.ndarray] = None,
                 reference_predictions: Optional[np.ndarray] = None,
                 drift_threshold: float = 0.05,
                 psi_threshold: float = 0.2,
                 accuracy_degradation_threshold: float = 0.05,
                 ece_warning_threshold: float = 0.1,
                 ece_critical_threshold: float = 0.2):
        """Initialize the advanced model monitor.

        Args:
            reference_features: Reference feature distribution for drift detection.
            reference_predictions: Reference prediction distribution.
            drift_threshold: P-value threshold for KS test drift detection.
            psi_threshold: PSI threshold for feature drift.
            accuracy_degradation_threshold: Relative accuracy drop for alerts.
            ece_warning_threshold: ECE value for calibration warning.
            ece_critical_threshold: ECE value for calibration critical alert.
        """
        super().__init__(reference_features, reference_predictions, drift_threshold, psi_threshold)
        self.accuracy_tracker = PredictionAccuracyTracker(
            degradation_threshold=accuracy_degradation_threshold,
        )
        self.calibration_monitor = CalibrationMonitor(
            ece_warning_threshold=ece_warning_threshold,
            ece_critical_threshold=ece_critical_threshold,
        )
        self._alerts: List[DegradationAlert] = []
        self._model_statuses: Dict[str, ModelStatus] = {}
        self._model_versions: Dict[str, str] = {}
        self._comparison_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._check_interval_seconds = 300  # 5 minutes default
        self._last_check: Dict[str, float] = defaultdict(float)

    def register_model(self, model_id: str, version: str = "1.0.0",
                       baseline_accuracy: Optional[float] = None,
                       regime: str = "default") -> None:
        """Register a model for monitoring.

        Args:
            model_id: Unique model identifier.
            version: Model version string.
            baseline_accuracy: Baseline accuracy for degradation checks.
            regime: Default regime for this model.
        """
        self._model_versions[model_id] = version
        self._model_statuses[model_id] = ModelStatus.UNKNOWN
        if baseline_accuracy is not None:
            self.accuracy_tracker.set_baseline(model_id, regime, baseline_accuracy)
        logger.info("Registered model '%s' v%s for monitoring", model_id, version)

    def record_prediction(self, model_id: str, predicted: Any, actual: Any,
                          predicted_prob: Optional[float] = None,
                          regime: str = "default") -> List[DegradationAlert]:
        """Record a prediction and run monitoring checks.

        Args:
            model_id: Model identifier.
            predicted: Model prediction.
            actual: Ground truth value.
            predicted_prob: Predicted probability (for calibration).
            regime: Market regime.

        Returns:
            List of any new alerts generated.
        """
        new_alerts = []

        # Record accuracy
        self.accuracy_tracker.record_prediction(
            model_id, predicted, actual, regime, self._model_versions.get(model_id, "")
        )

        # Record calibration
        if predicted_prob is not None:
            self.calibration_monitor.record(model_id, predicted_prob, predicted == actual)

        # Check for alerts periodically
        now = time.time()
        if now - self._last_check.get(model_id, 0) > self._check_interval_seconds:
            self._last_check[model_id] = now
            new_alerts.extend(self._run_checks(model_id, regime))

        return new_alerts

    def _run_checks(self, model_id: str, regime: str = "default") -> List[DegradationAlert]:
        """Run all monitoring checks for a model.

        Args:
            model_id: Model identifier.
            regime: Market regime.

        Returns:
            List of new alerts.
        """
        alerts = []

        # 1. Accuracy degradation check
        if self.accuracy_tracker.check_degradation(model_id, regime):
            current = self.accuracy_tracker.get_accuracy(model_id, regime)
            baseline = self.accuracy_tracker._baselines.get(model_id, {}).get(regime, 0.0)
            alerts.append(DegradationAlert(
                model_id=model_id,
                alert_type="accuracy_degradation",
                severity=AlertSeverity.WARNING,
                message=f"Model '{model_id}' accuracy degraded from {baseline:.4f} to "
                        f"{current:.4f} in regime '{regime}'",
                metric_value=current or 0.0,
                threshold=baseline,
                regime=regime,
            ))

        # 2. Calibration check
        cal_alert = self.calibration_monitor.check_calibration_alert(model_id)
        if cal_alert is not None:
            alerts.append(cal_alert)

        # 3. Feature drift check
        if self.reference_features is not None:
            # This would be called with current features separately
            pass

        # 4. Performance regression test
        regression = self.accuracy_tracker.get_regression_test_result(model_id, regime)
        if regression.get("is_regression"):
            alerts.append(DegradationAlert(
                model_id=model_id,
                alert_type="performance_regression",
                severity=AlertSeverity.WARNING,
                message=f"Model '{model_id}' shows statistically significant accuracy "
                        f"regression (p={regression['p_value']:.4f})",
                metric_value=regression["current_accuracy"],
                threshold=regression["baseline_accuracy"],
                regime=regime,
            ))

        # Update model status
        if alerts:
            critical = any(a.severity == AlertSeverity.CRITICAL for a in alerts)
            self._model_statuses[model_id] = ModelStatus.UNHEALTHY if critical else ModelStatus.DEGRADED
        else:
            self._model_statuses[model_id] = ModelStatus.HEALTHY

        self._alerts.extend(alerts)
        return alerts

    def run_feature_drift_check(self, model_id: str,
                                 current_features: np.ndarray) -> Optional[DegradationAlert]:
        """Run feature drift check and generate alert if drift detected.

        Args:
            model_id: Model identifier.
            current_features: Current feature matrix.

        Returns:
            DegradationAlert if drift detected, None otherwise.
        """
        drift_result = self.detect_feature_drift(current_features)
        if drift_result.get("drift_detected"):
            n_drifted = sum(
                1 for v in drift_result.get("details", {}).values()
                if v.get("drifted")
            )
            return DegradationAlert(
                model_id=model_id,
                alert_type="feature_drift",
                severity=AlertSeverity.WARNING if n_drifted <= 3 else AlertSeverity.CRITICAL,
                message=f"Feature drift detected in {n_drifted} features for model '{model_id}'",
                metric_value=float(n_drifted),
                threshold=0,
            )
        return None

    def run_prediction_drift_check(self, model_id: str,
                                    current_predictions: np.ndarray) -> Optional[DegradationAlert]:
        """Run prediction drift check and generate alert if drift detected.

        Args:
            model_id: Model identifier.
            current_predictions: Current model predictions.

        Returns:
            DegradationAlert if drift detected, None otherwise.
        """
        drift_result = self.detect_prediction_drift(current_predictions)
        if drift_result.get("drift_detected"):
            return DegradationAlert(
                model_id=model_id,
                alert_type="prediction_drift",
                severity=AlertSeverity.WARNING,
                message=f"Prediction drift detected for model '{model_id}' "
                        f"(KS stat={drift_result['ks_statistic']:.4f})",
                metric_value=drift_result["ks_statistic"],
                threshold=drift_result.get("p_value", 0.0),
            )
        return None

    def compare_models(self, model_ids: List[str], regime: str = "default") -> Dict[str, Any]:
        """Compare models on current accuracy metrics.

        Args:
            model_ids: List of model identifiers to compare.
            regime: Market regime for comparison.

        Returns:
            Dict with comparison results and ranking.
        """
        results = []
        for model_id in model_ids:
            accuracy = self.accuracy_tracker.get_accuracy(model_id, regime)
            ece = self.calibration_monitor.compute_ece(model_id)
            status = self._model_statuses.get(model_id, ModelStatus.UNKNOWN)
            results.append({
                "model_id": model_id,
                "version": self._model_versions.get(model_id, "unknown"),
                "accuracy": accuracy,
                "ece": ece,
                "status": status.value if isinstance(status, ModelStatus) else status,
            })

        # Sort by accuracy (highest first), None last
        results.sort(key=lambda x: x["accuracy"] if x["accuracy"] is not None else -1, reverse=True)

        comparison = {
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime,
            "models": results,
            "best_model": results[0]["model_id"] if results else None,
        }

        # Store in comparison history
        for model_id in model_ids:
            self._comparison_history[model_id].append(comparison)

        return comparison

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a model.

        Args:
            model_id: Model identifier.

        Returns:
            Dict with model health status, accuracy, calibration, and recent alerts.
        """
        status = self._model_statuses.get(model_id, ModelStatus.UNKNOWN)
        accuracies = self.accuracy_tracker.get_all_accuracies(model_id)
        ece = self.calibration_monitor.compute_ece(model_id)
        recent_alerts = [
            a.to_dict() for a in self._alerts
            if a.model_id == model_id
        ][-20:]

        return {
            "model_id": model_id,
            "version": self._model_versions.get(model_id, "unknown"),
            "status": status.value if isinstance(status, ModelStatus) else status,
            "accuracies": accuracies,
            "ece": ece,
            "recent_alerts": recent_alerts,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ============================================================================
# Model Health Dashboard
# ============================================================================

class ModelHealthDashboard:
    """Generates dashboard-ready data for model health monitoring.

    Aggregates data from AIModelMonitor into dashboard formats
    suitable for visualization tools (Grafana, custom dashboards).
    """

    def __init__(self, monitor: AIModelMonitor):
        """Initialize the dashboard data generator.

        Args:
            monitor: AIModelMonitor instance to pull data from.
        """
        self.monitor = monitor

    def get_overview(self) -> Dict[str, Any]:
        """Get dashboard overview with all models' status.

        Returns:
            Dict with model counts by status, total alerts, and per-model summaries.
        """
        model_ids = list(self.monitor._model_versions.keys())
        status_counts = defaultdict(int)
        model_summaries = []

        for model_id in model_ids:
            model_status = self.monitor.get_model_status(model_id)
            status = model_status.get("status", "unknown")
            status_counts[status] += 1
            model_summaries.append({
                "model_id": model_id,
                "version": model_status.get("version", "unknown"),
                "status": status,
                "default_accuracy": model_status.get("accuracies", {}).get("default"),
                "ece": model_status.get("ece"),
                "alert_count": len(model_status.get("recent_alerts", [])),
            })

        recent_alerts = [a.to_dict() for a in self.monitor._alerts[-50:]]

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_models": len(model_ids),
            "status_counts": dict(status_counts),
            "models": model_summaries,
            "recent_alerts": recent_alerts,
        }

    def get_model_detail(self, model_id: str) -> Dict[str, Any]:
        """Get detailed dashboard data for a single model.

        Args:
            model_id: Model identifier.

        Returns:
            Dict with accuracy history, calibration data, and alerts.
        """
        accuracy_history = self.monitor.accuracy_tracker.get_history(model_id, limit=200)
        calibration_data = self.monitor.calibration_monitor.get_reliability_data(model_id)
        model_status = self.monitor.get_model_status(model_id)

        return {
            "model_id": model_id,
            "status": model_status.get("status", "unknown"),
            "version": model_status.get("version", "unknown"),
            "accuracy_history": accuracy_history,
            "calibration": calibration_data,
            "accuracies_by_regime": model_status.get("accuracies", {}),
            "ece": model_status.get("ece"),
            "alerts": model_status.get("recent_alerts", []),
        }

    def get_comparison_chart_data(self, model_ids: List[str],
                                   regime: str = "default") -> Dict[str, Any]:
        """Get chart-ready data for comparing models.

        Args:
            model_ids: Models to compare.
            regime: Market regime for comparison.

        Returns:
            Dict with comparison data suitable for bar/line charts.
        """
        comparison = self.monitor.compare_models(model_ids, regime)

        return {
            "chart_type": "comparison",
            "regime": regime,
            "models": [
                {
                    "label": m["model_id"],
                    "accuracy": m["accuracy"],
                    "ece": m["ece"],
                    "status": m["status"],
                }
                for m in comparison["models"]
            ],
            "timestamp": comparison["timestamp"],
        }
