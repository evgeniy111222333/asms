"""
ACMS AI Feature Drift Detection
=================================

Multi-dimensional feature and concept drift detection for the
Algorithmic Crypto Management System. Distinguishes between
covariate drift (input distribution change) and concept drift
(input-output relationship change), with automated retraining
triggers and drift visualization data generation.

Components
----------
FeatureDriftMonitor : Main drift monitoring orchestrator
DriftResult : Result of a drift detection test
DriftType : Enum for drift classification
DriftAlert : Alert emitted when drift is detected
DriftVisualizer : Generates visualization data for drift dashboards
RetrainingTrigger : Automated model retraining trigger based on drift
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------

class DriftType(Enum):
    """Classification of drift type."""
    COVARIATE = "covariate"       # Input feature distribution has changed
    CONCEPT = "concept"           # Input-output relationship has changed
    PRIOR = "prior"               # P(Y) distribution has changed
    LIKELIHOOD = "likelihood"     # P(X|Y) distribution has changed
    TWO_SAMPLE = "two_sample"     # General two-sample distribution shift


class DriftSeverity(Enum):
    """Severity level of detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of a drift detection test for a single feature or model.

    Attributes
    ----------
    feature_name : str
        Name of the feature tested (or ``"model"`` for concept drift).
    drift_type : DriftType
        Classification of the drift type.
    statistic : float
        Test statistic value.
    p_value : float
        P-value of the test.
    threshold : float
        Significance threshold used.
    is_drift : bool
        Whether drift was detected.
    severity : DriftSeverity
        Severity of detected drift.
    reference_mean : float
        Mean of reference distribution.
    current_mean : float
        Mean of current distribution.
    reference_std : float
        Std of reference distribution.
    current_std : float
        Std of current distribution.
    effect_size : float
        Cohen's d or similar effect size measure.
    timestamp : float
        When this test was performed.
    metadata : dict
        Additional metadata.
    """
    feature_name: str = ""
    drift_type: DriftType = DriftType.COVARIATE
    statistic: float = 0.0
    p_value: float = 1.0
    threshold: float = 0.05
    is_drift: bool = False
    severity: DriftSeverity = DriftSeverity.NONE
    reference_mean: float = 0.0
    current_mean: float = 0.0
    reference_std: float = 0.0
    current_std: float = 0.0
    effect_size: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftAlert:
    """Alert emitted when drift is detected.

    Attributes
    ----------
    alert_id : str
        Unique alert identifier.
    feature_name : str
        Feature that triggered the alert.
    drift_type : DriftType
        Type of drift detected.
    severity : DriftSeverity
        Alert severity level.
    message : str
        Human-readable alert message.
    timestamp : float
        When the alert was generated.
    action_required : str
        Suggested action to take.
    metadata : dict
        Additional alert metadata.
    """
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    feature_name: str = ""
    drift_type: DriftType = DriftType.COVARIATE
    severity: DriftSeverity = DriftSeverity.NONE
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    action_required: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Drift Detectors (Statistical Tests)
# ---------------------------------------------------------------------------

class _KSDetector:
    """Kolmogorov-Smirnov two-sample test for drift detection."""

    @staticmethod
    def detect(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Run KS test and return (statistic, p-value).

        Uses a permutation-based approximation for the p-value.
        """
        ref_sorted = np.sort(reference)
        cur_sorted = np.sort(current)
        n_ref = len(ref_sorted)
        n_cur = len(cur_sorted)

        # Compute KS statistic
        all_vals = np.sort(np.concatenate([ref_sorted, cur_sorted]))
        max_diff = 0.0
        for val in all_vals[::max(1, len(all_vals) // 200)]:
            ref_cdf = np.searchsorted(ref_sorted, val, side="right") / n_ref
            cur_cdf = np.searchsorted(cur_sorted, val, side="right") / n_cur
            diff = abs(ref_cdf - cur_cdf)
            max_diff = max(max_diff, diff)

        # Approximate p-value using Kolmogorov distribution
        n_eff = (n_ref * n_cur) / (n_ref + n_cur)
        lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * max_diff
        p_value = _KSDetector._kolmogorov_sf(lam)
        return float(max_diff), float(p_value)

    @staticmethod
    def _kolmogorov_sf(lam: float) -> float:
        """Survival function of Kolmogorov distribution (approximation)."""
        if lam < 0.01:
            return 1.0
        # Asymptotic formula
        result = 0.0
        for k in range(1, 101):
            term = (-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)
            result += term
        return float(max(0.0, 2.0 * result))


class _CVMTest:
    """Cramér-von Mises two-sample test."""

    @staticmethod
    def detect(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """Run CvM test and return (statistic, p-value)."""
        m, n = len(reference), len(current)
        combined = np.concatenate([reference, current])
        ranks = np.argsort(np.argsort(combined)) + 1

        u = np.sum(ranks[:m])
        statistic = u / (m * n) - (2 * m - 1) / (2 * n)

        # Approximate p-value (simplified)
        n_total = m + n
        expected = m * (n_total + 1) / 2
        var = m * n * (n_total + 1) / 12
        if var < 1e-10:
            return 0.0, 1.0
        z = (u - expected) / np.sqrt(var)
        p_value = 2.0 * min(_normal_cdf(z), 1.0 - _normal_cdf(z))
        return float(abs(statistic)), float(p_value)


class _PSIDetector:
    """Population Stability Index detector."""

    @staticmethod
    def detect(reference: np.ndarray, current: np.ndarray,
               n_bins: int = 10) -> Tuple[float, str]:
        """Compute PSI and return (psi_value, severity_label)."""
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        ref_pct = ref_hist / (len(reference) + 1e-8)
        cur_pct = cur_hist / (len(current) + 1e-8)

        psi = 0.0
        for p, q in zip(ref_pct, cur_pct):
            if p > 0 and q > 0:
                psi += (p - q) * np.log(p / q)

        if psi < 0.1:
            severity = "none"
        elif psi < 0.2:
            severity = "low"
        elif psi < 0.3:
            severity = "medium"
        else:
            severity = "high"

        return float(psi), severity


def _normal_cdf(z: float) -> float:
    """Standard normal CDF approximation."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if z >= 0 else -1
    z = abs(z)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-z * z)
    return 0.5 * (1.0 + sign * y)


# ---------------------------------------------------------------------------
# Drift Visualizer
# ---------------------------------------------------------------------------

class DriftVisualizer:
    """Generates visualization data for drift dashboards.

    Produces data structures suitable for rendering distribution
    comparisons, drift timelines, and heatmaps.
    """

    def __init__(self, n_bins: int = 30) -> None:
        self._n_bins = n_bins
        self._history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        logger.info("DriftVisualizer initialized (bins=%d)", n_bins)

    def generate_distribution_comparison(
        self,
        feature_name: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate distribution comparison data for visualization.

        Returns
        -------
        dict
            Data for rendering overlapping histograms or KDE plots.
        """
        bins = np.linspace(
            min(np.min(reference), np.min(current)) if len(reference) > 0 and len(current) > 0 else 0,
            max(np.max(reference), np.max(current)) if len(reference) > 0 and len(current) > 0 else 1,
            self._n_bins + 1,
        )

        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        comparison = {
            "feature_name": feature_name,
            "bin_centers": bin_centers.tolist(),
            "reference_density": ref_hist.tolist(),
            "current_density": cur_hist.tolist(),
            "reference_mean": float(np.mean(reference)),
            "current_mean": float(np.mean(current)),
            "reference_std": float(np.std(reference)),
            "current_std": float(np.std(current)),
            "timestamp": time.time(),
        }

        self._history[feature_name].append(comparison)
        return comparison

    def generate_drift_heatmap(
        self,
        drift_results: List[DriftResult],
    ) -> Dict[str, Any]:
        """Generate drift heatmap data across features and time.

        Returns
        -------
        dict
            Data for rendering a drift severity heatmap.
        """
        feature_names = list(set(r.feature_name for r in drift_results))
        feature_names.sort()

        matrix = []
        for fname in feature_names:
            feature_results = [r for r in drift_results if r.feature_name == fname]
            row = {
                "feature": fname,
                "values": [
                    {
                        "severity": r.severity.value,
                        "statistic": r.statistic,
                        "p_value": r.p_value,
                        "effect_size": r.effect_size,
                    }
                    for r in feature_results
                ],
            }
            matrix.append(row)

        return {
            "features": feature_names,
            "matrix": matrix,
            "timestamp": time.time(),
        }

    def generate_timeline(self, feature_name: str) -> Dict[str, Any]:
        """Generate timeline data showing drift evolution over time."""
        history = self._history.get(feature_name, [])
        return {
            "feature_name": feature_name,
            "snapshots": [
                {
                    "timestamp": s.get("timestamp", 0),
                    "reference_mean": s.get("reference_mean", 0),
                    "current_mean": s.get("current_mean", 0),
                    "mean_shift": s.get("current_mean", 0) - s.get("reference_mean", 0),
                }
                for s in history
            ],
        }

    @property
    def history(self) -> Dict[str, List[Dict[str, Any]]]:
        return dict(self._history)


# ---------------------------------------------------------------------------
# Retraining Trigger
# ---------------------------------------------------------------------------

class RetrainingTrigger:
    """Automated model retraining trigger based on drift detection.

    Monitors drift alerts and triggers model retraining when
    conditions are met (severity threshold, duration, or number
    of drifting features).

    Parameters
    ----------
    min_drift_severity : DriftSeverity
        Minimum severity to consider for triggering.
    min_drifting_features : int
        Minimum number of drifting features to trigger retraining.
    cooldown_hours : float
        Minimum hours between retraining triggers.
    callback : callable, optional
        Function to call when retraining is triggered.
    """

    def __init__(
        self,
        min_drift_severity: DriftSeverity = DriftSeverity.MEDIUM,
        min_drifting_features: int = 3,
        cooldown_hours: float = 24.0,
        callback: Optional[Callable] = None,
    ) -> None:
        self._min_severity = min_drift_severity
        self._min_drifting = min_drifting_features
        self._cooldown = cooldown_hours
        self._callback = callback
        self._last_trigger: float = 0.0
        self._trigger_count: int = 0
        self._drifting_features: Dict[str, DriftSeverity] = {}
        self._severity_order = {
            DriftSeverity.NONE: 0,
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4,
        }
        logger.info(
            "RetrainingTrigger initialized (min_severity=%s, min_features=%d)",
            min_drift_severity.value, min_drifting_features,
        )

    def evaluate(self, alerts: List[DriftAlert]) -> Optional[Dict[str, Any]]:
        """Evaluate drift alerts and determine if retraining should be triggered.

        Parameters
        ----------
        alerts : list of DriftAlert
            Recent drift alerts.

        Returns
        -------
        dict or None
            Trigger decision with details, or None if no trigger.
        """
        # Update drifting features
        for alert in alerts:
            if self._severity_order.get(alert.severity, 0) >= self._severity_order.get(self._min_severity, 2):
                self._drifting_features[alert.feature_name] = alert.severity
            else:
                self._drifting_features.pop(alert.feature_name, None)

        # Check cooldown
        hours_since_last = (time.time() - self._last_trigger) / 3600.0
        if hours_since_last < self._cooldown:
            return None

        # Check if enough features are drifting
        if len(self._drifting_features) < self._min_drifting:
            return None

        # Check for critical drift (immediate trigger)
        has_critical = any(
            s == DriftSeverity.CRITICAL for s in self._drifting_features.values()
        )

        if has_critical or len(self._drifting_features) >= self._min_drifting:
            trigger_info = {
                "triggered_at": time.time(),
                "drifting_features": list(self._drifting_features.keys()),
                "severity_per_feature": {k: v.value for k, v in self._drifting_features.items()},
                "total_drifting": len(self._drifting_features),
                "has_critical": has_critical,
                "cooldown_remaining_hours": 0.0,
            }

            self._last_trigger = time.time()
            self._trigger_count += 1
            self._drifting_features.clear()

            logger.warning(
                "Retraining triggered: %d features drifting (critical=%s)",
                trigger_info["total_drifting"], has_critical,
            )

            if self._callback:
                try:
                    self._callback(trigger_info)
                except Exception as exc:
                    logger.error("Retraining callback error: %s", exc)

            return trigger_info

        return None

    @property
    def trigger_count(self) -> int:
        return self._trigger_count

    @property
    def last_trigger_time(self) -> Optional[float]:
        return self._last_trigger if self._trigger_count > 0 else None


# ---------------------------------------------------------------------------
# Feature Drift Monitor
# ---------------------------------------------------------------------------

class FeatureDriftMonitor:
    """Multi-dimensional feature and concept drift monitor.

    Orchestrates drift detection across multiple features simultaneously,
    distinguishing between covariate drift (feature distribution change)
    and concept drift (feature-target relationship change). Emits alerts
    when drift is detected and can trigger automated model retraining.

    Parameters
    ----------
    reference_window_size : int
        Number of observations in the reference (baseline) window.
    current_window_size : int
        Number of observations in the current (monitoring) window.
    significance_level : float
        P-value threshold for drift detection.
    check_interval_seconds : float
        Interval between periodic drift checks.
    enable_concept_drift : bool
        Whether to also monitor concept drift (requires target values).
    psi_threshold : float
        PSI threshold for drift severity classification.

    Examples
    --------
    >>> monitor = FeatureDriftMonitor(reference_window_size=1000)
    >>> monitor.set_reference("btc_returns", reference_data)
    >>> monitor.add_current("btc_returns", new_data)
    >>> results = monitor.detect_drift("btc_returns")
    >>> all_results = monitor.detect_all()
    """

    def __init__(
        self,
        reference_window_size: int = 1000,
        current_window_size: int = 500,
        significance_level: float = 0.05,
        check_interval_seconds: float = 300.0,
        enable_concept_drift: bool = True,
        psi_threshold: float = 0.2,
    ) -> None:
        self._ref_size = reference_window_size
        self._cur_size = current_window_size
        self._alpha = significance_level
        self._check_interval = check_interval_seconds
        self._enable_concept = enable_concept_drift
        self._psi_threshold = psi_threshold

        # Data storage
        self._reference: Dict[str, np.ndarray] = {}
        self._current: Dict[str, np.ndarray] = {}
        self._reference_target: Optional[np.ndarray] = None
        self._current_target: Optional[np.ndarray] = None

        # Results and alerts
        self._drift_results: Dict[str, List[DriftResult]] = defaultdict(list)
        self._alerts: List[DriftAlert] = []
        self._alert_callbacks: List[Callable] = []

        # Sub-components
        self._visualizer = DriftVisualizer()
        self._retraining_trigger = RetrainingTrigger()

        # Monitoring state
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

        logger.info(
            "FeatureDriftMonitor initialized (ref=%d, cur=%d, alpha=%.3f, concept=%s)",
            reference_window_size, current_window_size, significance_level,
            enable_concept_drift,
        )

    # -- Reference Data Management --

    def set_reference(self, feature_name: str, data: np.ndarray) -> None:
        """Set the reference (baseline) distribution for a feature.

        Parameters
        ----------
        feature_name : str
            Feature name.
        data : np.ndarray
            Reference distribution values.
        """
        self._reference[feature_name] = np.asarray(data, dtype=np.float64)[-self._ref_size:]
        logger.debug("Reference set for '%s' (%d samples)", feature_name, len(self._reference[feature_name]))

    def set_reference_batch(self, features: Dict[str, np.ndarray]) -> None:
        """Set reference distributions for multiple features at once."""
        for fname, data in features.items():
            self.set_reference(fname, data)

    def set_reference_target(self, target: np.ndarray) -> None:
        """Set the reference target values for concept drift detection."""
        self._reference_target = np.asarray(target, dtype=np.float64)[-self._ref_size:]

    # -- Current Data Management --

    def add_current(self, feature_name: str, data: np.ndarray) -> None:
        """Add current observations for a feature.

        Parameters
        ----------
        feature_name : str
            Feature name.
        data : np.ndarray
            New observations to append.
        """
        data = np.asarray(data, dtype=np.float64)
        if feature_name in self._current:
            self._current[feature_name] = np.concatenate([self._current[feature_name], data])
        else:
            self._current[feature_name] = data
        # Keep only the most recent window
        self._current[feature_name] = self._current[feature_name][-self._cur_size:]

    def add_current_target(self, target: np.ndarray) -> None:
        """Add current target observations for concept drift detection."""
        target = np.asarray(target, dtype=np.float64)
        if self._current_target is not None:
            self._current_target = np.concatenate([self._current_target, target])
        else:
            self._current_target = target
        self._current_target = self._current_target[-self._cur_size:]

    # -- Drift Detection --

    def detect_drift(self, feature_name: str) -> DriftResult:
        """Run drift detection for a single feature.

        Executes multiple statistical tests (KS, CvM, PSI) and
        aggregates the results into a single DriftResult.

        Parameters
        ----------
        feature_name : str
            Feature name to test.

        Returns
        -------
        DriftResult
            Drift detection result.
        """
        ref = self._reference.get(feature_name)
        cur = self._current.get(feature_name)

        if ref is None or cur is None:
            return DriftResult(
                feature_name=feature_name,
                metadata={"error": "Missing reference or current data"},
            )

        if len(cur) < 30:
            return DriftResult(
                feature_name=feature_name,
                metadata={"error": f"Insufficient current data ({len(cur)} < 30)"},
            )

        # Clean data
        ref_clean = ref[~np.isnan(ref) & ~np.isinf(ref)]
        cur_clean = cur[~np.isnan(cur) & ~np.isinf(cur)]

        if len(ref_clean) < 10 or len(cur_clean) < 10:
            return DriftResult(
                feature_name=feature_name,
                metadata={"error": "Insufficient clean data"},
            )

        # KS test
        ks_stat, ks_pval = _KSDetector.detect(ref_clean, cur_clean)

        # CvM test
        cvm_stat, cvm_pval = _CVMTest.detect(ref_clean, cur_clean)

        # PSI
        psi_val, psi_severity = _PSIDetector.detect(ref_clean, cur_clean)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(ref_clean, ddof=1) * (len(ref_clean) - 1) +
             np.var(cur_clean, ddof=1) * (len(cur_clean) - 1))
            / (len(ref_clean) + len(cur_clean) - 2)
        )
        effect_size = float((np.mean(cur_clean) - np.mean(ref_clean)) / (pooled_std + 1e-8))

        # Aggregate decision: drift if any test is significant
        is_drift = ks_pval < self._alpha or cvm_pval < self._alpha or psi_val > self._psi_threshold

        # Severity classification
        min_pval = min(ks_pval, cvm_pval)
        if not is_drift:
            severity = DriftSeverity.NONE
        elif min_pval < 0.001 or psi_val > 0.3:
            severity = DriftSeverity.CRITICAL
        elif min_pval < 0.01 or psi_val > 0.2:
            severity = DriftSeverity.HIGH
        elif min_pval < 0.01:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        result = DriftResult(
            feature_name=feature_name,
            drift_type=DriftType.COVARIATE,
            statistic=float(ks_stat),
            p_value=float(min_pval),
            threshold=self._alpha,
            is_drift=is_drift,
            severity=severity,
            reference_mean=float(np.mean(ref_clean)),
            current_mean=float(np.mean(cur_clean)),
            reference_std=float(np.std(ref_clean)),
            current_std=float(np.std(cur_clean)),
            effect_size=effect_size,
            metadata={
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_pval),
                "cvm_statistic": float(cvm_stat),
                "cvm_p_value": float(cvm_pval),
                "psi": float(psi_val),
                "psi_severity": psi_severity,
                "n_reference": len(ref_clean),
                "n_current": len(cur_clean),
            },
        )

        self._drift_results[feature_name].append(result)

        # Generate alert if drift detected
        if is_drift:
            self._emit_alert(result)

        # Generate visualization data
        self._visualizer.generate_distribution_comparison(feature_name, ref_clean, cur_clean)

        return result

    def detect_concept_drift(self) -> Optional[DriftResult]:
        """Detect concept drift (change in feature-target relationship).

        Compares the correlation structure between features and target
        in the reference vs current windows.

        Returns
        -------
        DriftResult or None
            Concept drift result, or None if insufficient data.
        """
        if not self._enable_concept:
            return None

        if self._reference_target is None or self._current_target is None:
            return None

        ref_target = self._reference_target[~np.isnan(self._reference_target)]
        cur_target = self._current_target[~np.isnan(self._current_target)]

        if len(ref_target) < 30 or len(cur_target) < 30:
            return None

        # Compare target distributions
        ks_stat, ks_pval = _KSDetector.detect(ref_target, cur_target)

        # Compare feature-target correlations
        ref_corrs: List[float] = []
        cur_corrs: List[float] = []
        for fname in self._reference:
            ref_feat = self._reference.get(fname)
            cur_feat = self._current.get(fname)
            if ref_feat is None or cur_feat is None:
                continue

            ref_feat_clean = ref_feat[~np.isnan(ref_feat)]
            cur_feat_clean = cur_feat[~np.isnan(cur_feat)]
            min_ref = min(len(ref_feat_clean), len(ref_target))
            min_cur = min(len(cur_feat_clean), len(cur_target))

            if min_ref > 10 and min_cur > 10:
                ref_c = abs(float(np.corrcoef(ref_feat_clean[:min_ref], ref_target[:min_ref])[0, 1]))
                cur_c = abs(float(np.corrcoef(cur_feat_clean[:min_cur], cur_target[:min_cur])[0, 1]))
                if not np.isnan(ref_c) and not np.isnan(cur_c):
                    ref_corrs.append(ref_c)
                    cur_corrs.append(cur_c)

        # If correlation structure has changed significantly
        avg_corr_shift = 0.0
        if ref_corrs and cur_corrs:
            avg_corr_shift = float(abs(np.mean(cur_corrs) - np.mean(ref_corrs)))

        is_drift = ks_pval < self._alpha or avg_corr_shift > 0.1

        severity = DriftSeverity.NONE
        if is_drift:
            if ks_pval < 0.001 or avg_corr_shift > 0.3:
                severity = DriftSeverity.CRITICAL
            elif ks_pval < 0.01 or avg_corr_shift > 0.2:
                severity = DriftSeverity.HIGH
            else:
                severity = DriftSeverity.MEDIUM

        result = DriftResult(
            feature_name="model",
            drift_type=DriftType.CONCEPT,
            statistic=float(ks_stat),
            p_value=float(ks_pval),
            threshold=self._alpha,
            is_drift=is_drift,
            severity=severity,
            reference_mean=float(np.mean(ref_target)),
            current_mean=float(np.mean(cur_target)),
            reference_std=float(np.std(ref_target)),
            current_std=float(np.std(cur_target)),
            effect_size=avg_corr_shift,
            metadata={
                "avg_correlation_shift": avg_corr_shift,
                "reference_avg_corr": float(np.mean(ref_corrs)) if ref_corrs else 0.0,
                "current_avg_corr": float(np.mean(cur_corrs)) if cur_corrs else 0.0,
                "n_ref_correlations": len(ref_corrs),
                "n_cur_correlations": len(cur_corrs),
            },
        )

        self._drift_results["concept"].append(result)

        if is_drift:
            self._emit_alert(result)

        return result

    def detect_all(self) -> Dict[str, DriftResult]:
        """Run drift detection for all registered features.

        Returns
        -------
        dict
            Mapping of feature name to DriftResult.
        """
        results: Dict[str, DriftResult] = {}

        for feature_name in self._reference:
            results[feature_name] = self.detect_drift(feature_name)

        # Also check concept drift
        concept_result = self.detect_concept_drift()
        if concept_result is not None:
            results["concept"] = concept_result

        # Evaluate retraining trigger
        self._retraining_trigger.evaluate(self._alerts[-len(results):])

        return results

    # -- Alert Management --

    def _emit_alert(self, result: DriftResult) -> None:
        """Emit a drift alert."""
        action_map = {
            DriftSeverity.LOW: "Monitor closely",
            DriftSeverity.MEDIUM: "Consider model retraining",
            DriftSeverity.HIGH: "Schedule model retraining",
            DriftSeverity.CRITICAL: "Immediate retraining required",
        }

        alert = DriftAlert(
            feature_name=result.feature_name,
            drift_type=result.drift_type,
            severity=result.severity,
            message=(
                f"{result.drift_type.value} drift detected in '{result.feature_name}': "
                f"p={result.p_value:.4f}, effect={result.effect_size:.3f}, "
                f"severity={result.severity.value}"
            ),
            action_required=action_map.get(result.severity, "Review"),
            metadata=result.metadata,
        )

        self._alerts.append(alert)
        logger.warning("Drift alert: %s", alert.message)

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as exc:
                logger.error("Alert callback error: %s", exc)

    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback to invoke when drift alerts are emitted."""
        self._alert_callbacks.append(callback)

    # -- Async Monitoring --

    async def start_monitoring(self) -> None:
        """Start periodic drift monitoring."""
        self._running = True
        self._check_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Drift monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop periodic drift monitoring."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                logger.debug("Drift monitoring task cancelled during stop")
        logger.info("Drift monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background loop for periodic drift checks."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)
                results = self.detect_all()
                drifting = {k: v for k, v in results.items() if v.is_drift}
                if drifting:
                    logger.info("Drift check: %d/%d features drifting",
                                len(drifting), len(results))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Monitoring loop error: %s", exc)

    # -- Query --

    def get_drift_results(self, feature_name: str) -> List[DriftResult]:
        """Get drift detection history for a feature."""
        return list(self._drift_results.get(feature_name, []))

    def get_recent_alerts(self, n: int = 50) -> List[DriftAlert]:
        """Get the most recent drift alerts."""
        return self._alerts[-n:]

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of drift monitoring status."""
        features_status: Dict[str, Any] = {}
        for fname, results in self._drift_results.items():
            if results:
                latest = results[-1]
                features_status[fname] = {
                    "is_drift": latest.is_drift,
                    "severity": latest.severity.value,
                    "p_value": latest.p_value,
                    "effect_size": latest.effect_size,
                    "last_checked": latest.timestamp,
                }

        return {
            "monitored_features": list(self._reference.keys()),
            "current_data_available": list(self._current.keys()),
            "features_status": features_status,
            "total_alerts": len(self._alerts),
            "retraining_triggers": self._retraining_trigger.trigger_count,
            "last_retraining": self._retraining_trigger.last_trigger_time,
            "timestamp": time.time(),
        }

    def get_visualization_data(self, feature_name: str) -> Dict[str, Any]:
        """Get visualization data for a specific feature."""
        return self._visualizer.generate_timeline(feature_name)

    def get_drift_heatmap(self) -> Dict[str, Any]:
        """Get drift heatmap data across all features."""
        all_results = []
        for results in self._drift_results.values():
            all_results.extend(results)
        return self._visualizer.generate_drift_heatmap(all_results)

    @property
    def visualizer(self) -> DriftVisualizer:
        return self._visualizer

    @property
    def retraining_trigger(self) -> RetrainingTrigger:
        return self._retraining_trigger
