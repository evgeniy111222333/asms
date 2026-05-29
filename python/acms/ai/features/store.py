"""
ACMS AI Feature Store
======================

Redis-backed feature store with real-time computation, versioning,
lineage tracking, quality scoring, and dependency graph management
for the Algorithmic Crypto Management System.

Components
----------
FeatureStore : Central feature storage with Redis backend
RealTimeFeatureComputer : On-demand feature computation
FeatureGroup : Logical grouping and management of features
FeatureStatistics : Statistical profiling of feature distributions
FeatureFreshnessMonitor : Monitoring feature staleness
FeatureVersion : Versioning and lineage tracking
FeatureQualityScorer : Automated feature quality assessment
FeatureDependencyGraph : Feature computation dependency management
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureVersion:
    """Versioned snapshot of a feature definition.

    Attributes
    ----------
    feature_name : str
        Name of the feature.
    version : int
        Monotonically increasing version number.
    definition : dict
        Feature computation definition (transform, source, params).
    created_at : float
        Unix timestamp of creation.
    created_by : str
        Creator identifier.
    parent_version : int, optional
        Previous version this was derived from.
    lineage : list of str
        List of upstream feature/data source names.
    """
    feature_name: str
    version: int
    definition: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    parent_version: Optional[int] = None
    lineage: List[str] = field(default_factory=list)

    def fingerprint(self) -> str:
        """Return a unique fingerprint for this feature version."""
        raw = f"{self.feature_name}:v{self.version}:{json.dumps(self.definition, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class FeatureStatistics:
    """Statistical profile of a feature's distribution.

    Computed over a configurable window of recent values.
    """
    feature_name: str = ""
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    zero_count: int = 0
    computed_at: float = field(default_factory=time.time)

    @classmethod
    def from_values(cls, feature_name: str, values: np.ndarray) -> "FeatureStatistics":
        """Compute statistics from an array of feature values."""
        clean = values[~np.isnan(values)]
        clean = clean[~np.isinf(clean)]

        return cls(
            feature_name=feature_name,
            count=len(values),
            mean=float(np.mean(clean)) if len(clean) > 0 else 0.0,
            std=float(np.std(clean)) if len(clean) > 1 else 0.0,
            min_val=float(np.min(clean)) if len(clean) > 0 else 0.0,
            max_val=float(np.max(clean)) if len(clean) > 0 else 0.0,
            p25=float(np.percentile(clean, 25)) if len(clean) > 0 else 0.0,
            p50=float(np.percentile(clean, 50)) if len(clean) > 0 else 0.0,
            p75=float(np.percentile(clean, 75)) if len(clean) > 0 else 0.0,
            p95=float(np.percentile(clean, 95)) if len(clean) > 0 else 0.0,
            p99=float(np.percentile(clean, 99)) if len(clean) > 0 else 0.0,
            nan_count=int(np.sum(np.isnan(values))),
            inf_count=int(np.sum(np.isinf(values))),
            zero_count=int(np.sum(values == 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p95": self.p95,
            "p99": self.p99,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "zero_count": self.zero_count,
            "computed_at": self.computed_at,
        }


# ---------------------------------------------------------------------------
# Feature Group
# ---------------------------------------------------------------------------

class FeatureGroup:
    """Logical grouping of related features.

    Feature groups enable batch computation, shared configuration,
    and coordinated freshness monitoring.

    Parameters
    ----------
    name : str
        Group name (e.g., ``"market_microstructure"``).
    description : str
        Human-readable description.
    feature_names : list of str
        Names of features in this group.
    compute_interval_s : float
        Recommended recomputation interval in seconds.
    tags : dict
        Arbitrary metadata tags.
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        feature_names: Optional[List[str]] = None,
        compute_interval_s: float = 60.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self._features: List[str] = list(feature_names or [])
        self._compute_interval = compute_interval_s
        self._tags = tags or {}
        self._last_computed: Optional[float] = None
        logger.info("FeatureGroup '%s' created with %d features", name, len(self._features))

    def add_feature(self, feature_name: str) -> None:
        """Add a feature to this group."""
        if feature_name not in self._features:
            self._features.append(feature_name)

    def remove_feature(self, feature_name: str) -> None:
        """Remove a feature from this group."""
        self._features = [f for f in self._features if f != feature_name]

    @property
    def feature_names(self) -> List[str]:
        return list(self._features)

    @property
    def last_computed(self) -> Optional[float]:
        return self._last_computed

    @last_computed.setter
    def last_computed(self, ts: float) -> None:
        self._last_computed = ts

    @property
    def is_stale(self) -> bool:
        """Check if the group needs recomputation."""
        if self._last_computed is None:
            return True
        return (time.time() - self._last_computed) > self._compute_interval

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "feature_names": self._features,
            "compute_interval_s": self._compute_interval,
            "tags": self._tags,
            "last_computed": self._last_computed,
            "is_stale": self.is_stale,
        }


# ---------------------------------------------------------------------------
# Feature Quality Scorer
# ---------------------------------------------------------------------------

class FeatureQualityScorer:
    """Automated feature quality assessment.

    Scores features on multiple dimensions:

    - **Completeness**: Fraction of non-null values.
    - **Uniqueness**: Fraction of unique values (detects constants).
    - **Stability**: Temporal consistency of distribution.
    - **Predictiveness**: Correlation with target (if available).

    Parameters
    ----------
    stability_window : int
        Number of recent observations for stability scoring.
    """

    def __init__(self, stability_window: int = 1000) -> None:
        self._window = stability_window
        self._history: Dict[str, List[np.ndarray]] = defaultdict(list)
        logger.info("FeatureQualityScorer initialized (window=%d)", stability_window)

    def score(self, feature_name: str, values: np.ndarray,
              target: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute quality scores for a feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature being scored.
        values : np.ndarray
            Current feature values.
        target : np.ndarray, optional
            Target values for predictiveness scoring.

        Returns
        -------
        dict
            Quality scores in [0, 1] for each dimension.
        """
        scores: Dict[str, float] = {}

        # Completeness
        if len(values) > 0:
            non_null = np.sum(~np.isnan(values) & ~np.isinf(values))
            scores["completeness"] = float(non_null / len(values))
        else:
            scores["completeness"] = 0.0

        # Uniqueness
        if len(values) > 0:
            n_unique = len(np.unique(values[~np.isnan(values)]))
            scores["uniqueness"] = float(min(1.0, n_unique / max(len(values), 1)))
        else:
            scores["uniqueness"] = 0.0

        # Stability
        self._history[feature_name].append(values.copy())
        if len(self._history[feature_name]) > 10:
            self._history[feature_name] = self._history[feature_name][-10:]
        scores["stability"] = self._compute_stability(feature_name)

        # Predictiveness
        if target is not None and len(values) == len(target):
            scores["predictiveness"] = self._compute_predictiveness(values, target)
        else:
            scores["predictiveness"] = 0.5  # neutral

        # Overall quality
        weights = {"completeness": 0.3, "uniqueness": 0.2, "stability": 0.3, "predictiveness": 0.2}
        scores["overall"] = sum(scores.get(k, 0.0) * w for k, w in weights.items())

        return scores

    def _compute_stability(self, feature_name: str) -> float:
        """Compute temporal stability of feature distribution."""
        history = self._history.get(feature_name, [])
        if len(history) < 2:
            return 1.0  # Assume stable with insufficient data

        means = [float(np.nanmean(h)) if len(h) > 0 else 0.0 for h in history]
        stds = [float(np.nanstd(h)) if len(h) > 1 else 0.0 for h in history]

        mean_cv = np.std(means) / (np.mean(means) + 1e-8)
        std_cv = np.std(stds) / (np.mean(stds) + 1e-8)

        stability = 1.0 - min(1.0, (mean_cv + std_cv) / 2.0)
        return float(max(0.0, stability))

    @staticmethod
    def _compute_predictiveness(values: np.ndarray, target: np.ndarray) -> float:
        """Compute absolute correlation with target as predictiveness score."""
        clean_mask = ~np.isnan(values) & ~np.isnan(target)
        clean_v = values[clean_mask]
        clean_t = target[clean_mask]

        if len(clean_v) < 10:
            return 0.5

        corr = np.corrcoef(clean_v, clean_t)[0, 1]
        if np.isnan(corr):
            return 0.5
        return float(min(1.0, abs(corr) * 2.0))  # Scale so 0.5 correlation → 1.0


# ---------------------------------------------------------------------------
# Feature Dependency Graph
# ---------------------------------------------------------------------------

class FeatureDependencyGraph:
    """Directed acyclic graph (DAG) of feature computation dependencies.

    Ensures features are computed in topological order and enables
    efficient recomputation when upstream features change.

    Attributes
    ----------
    adjacency : dict
        Mapping of feature name to set of downstream dependents.
    reverse_adjacency : dict
        Mapping of feature name to set of upstream dependencies.
    """

    def __init__(self) -> None:
        self._adj: Dict[str, Set[str]] = defaultdict(set)
        self._rev: Dict[str, Set[str]] = defaultdict(set)
        self._computation_fns: Dict[str, Callable] = {}
        logger.info("FeatureDependencyGraph initialized")

    def add_dependency(self, feature: str, depends_on: str,
                       compute_fn: Optional[Callable] = None) -> None:
        """Register a dependency: *feature* depends on *depends_on*.

        Parameters
        ----------
        feature : str
            The downstream feature name.
        depends_on : str
            The upstream feature or data source name.
        compute_fn : callable, optional
            Function to compute *feature* from its dependencies.
        """
        self._adj[depends_on].add(feature)
        self._rev[feature].add(depends_on)
        if compute_fn is not None:
            self._computation_fns[feature] = compute_fn

    def remove_feature(self, feature: str) -> None:
        """Remove a feature and all its dependency edges."""
        for dep in list(self._rev.get(feature, set())):
            self._adj[dep].discard(feature)
        self._rev.pop(feature, None)
        for downstream in list(self._adj.get(feature, set())):
            self._rev[downstream].discard(feature)
        self._adj.pop(feature, None)
        self._computation_fns.pop(feature, None)

    def get_dependencies(self, feature: str) -> Set[str]:
        """Return all direct upstream dependencies of a feature."""
        return set(self._rev.get(feature, set()))

    def get_dependents(self, feature: str) -> Set[str]:
        """Return all direct downstream dependents of a feature."""
        return set(self._adj.get(feature, set()))

    def get_all_dependents(self, feature: str) -> Set[str]:
        """Return all transitive downstream dependents of a feature."""
        visited: Set[str] = set()
        queue = [feature]
        while queue:
            current = queue.pop(0)
            for dep in self._adj.get(current, set()):
                if dep not in visited:
                    visited.add(dep)
                    queue.append(dep)
        return visited

    def topological_order(self) -> List[str]:
        """Return features in topological (dependency) order.

        Returns
        -------
        list of str
            Feature names ordered so that dependencies come first.
        """
        # Kahn's algorithm
        in_degree: Dict[str, int] = defaultdict(int)
        all_nodes: Set[str] = set()

        for node in list(self._rev.keys()) + list(self._adj.keys()):
            all_nodes.add(node)

        for node in all_nodes:
            in_degree.setdefault(node, 0)
        for node, deps in self._rev.items():
            in_degree[node] = len(deps)

        queue = [n for n in all_nodes if in_degree[n] == 0]
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for dependent in self._adj.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def detect_cycles(self) -> bool:
        """Check if the dependency graph contains cycles.

        Returns
        -------
        bool
            True if cycles are detected.
        """
        return len(self.topological_order()) < len(
            set(list(self._rev.keys()) + list(self._adj.keys()))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the graph structure."""
        return {
            "dependencies": {k: list(v) for k, v in self._rev.items()},
            "dependents": {k: list(v) for k, v in self._adj.items()},
            "topological_order": self.topological_order(),
            "has_cycles": self.detect_cycles(),
        }


# ---------------------------------------------------------------------------
# Feature Freshness Monitor
# ---------------------------------------------------------------------------

class FeatureFreshnessMonitor:
    """Monitors feature staleness and emits alerts.

    Tracks when each feature was last computed and alerts when
    features exceed their freshness threshold.

    Parameters
    ----------
    default_threshold_s : float
        Default staleness threshold in seconds.
    check_interval_s : float
        Interval between freshness checks.
    """

    def __init__(
        self,
        default_threshold_s: float = 300.0,
        check_interval_s: float = 60.0,
    ) -> None:
        self._default_threshold = default_threshold_s
        self._check_interval = check_interval_s
        self._thresholds: Dict[str, float] = {}
        self._last_updated: Dict[str, float] = {}
        self._stale_features: Set[str] = set()
        self._callbacks: List[Callable] = []
        logger.info("FeatureFreshnessMonitor initialized (threshold=%.0fs)", default_threshold_s)

    def register(self, feature_name: str, threshold_s: Optional[float] = None) -> None:
        """Register a feature for freshness monitoring."""
        self._thresholds[feature_name] = threshold_s or self._default_threshold
        self._last_updated[feature_name] = time.time()

    def update(self, feature_name: str) -> None:
        """Mark a feature as freshly updated."""
        self._last_updated[feature_name] = time.time()
        self._stale_features.discard(feature_name)

    def set_threshold(self, feature_name: str, threshold_s: float) -> None:
        """Update the staleness threshold for a feature."""
        self._thresholds[feature_name] = threshold_s

    def add_callback(self, callback: Callable) -> None:
        """Add a callback to invoke when features become stale."""
        self._callbacks.append(callback)

    def check_freshness(self) -> Dict[str, Any]:
        """Check all registered features for staleness.

        Returns
        -------
        dict
            Freshness report with stale features and details.
        """
        now = time.time()
        newly_stale: List[str] = []

        for fname, threshold in self._thresholds.items():
            last = self._last_updated.get(fname, 0)
            age = now - last
            if age > threshold:
                if fname not in self._stale_features:
                    newly_stale.append(fname)
                self._stale_features.add(fname)

        # Invoke callbacks for newly stale features
        for fname in newly_stale:
            age = now - self._last_updated.get(fname, 0)
            for cb in self._callbacks:
                try:
                    cb(fname, age)
                except Exception as exc:
                    logger.warning("Freshness callback error for %s: %s", fname, exc)

        report = {
            "total_monitored": len(self._thresholds),
            "stale_count": len(self._stale_features),
            "stale_features": list(self._stale_features),
            "newly_stale": newly_stale,
            "details": {},
        }

        for fname in self._stale_features:
            age = now - self._last_updated.get(fname, 0)
            threshold = self._thresholds.get(fname, self._default_threshold)
            report["details"][fname] = {
                "age_seconds": age,
                "threshold_seconds": threshold,
                "staleness_ratio": age / threshold,
            }

        return report


# ---------------------------------------------------------------------------
# Real-Time Feature Computer
# ---------------------------------------------------------------------------

class RealTimeFeatureComputer:
    """On-demand feature computation with caching and dependency resolution.

    Computes features from raw data sources using registered computation
    functions, respecting the dependency graph for ordering.

    Parameters
    ----------
    dependency_graph : FeatureDependencyGraph
        The feature dependency DAG.
    cache_ttl_s : float
        Cache TTL for computed features.
    """

    def __init__(
        self,
        dependency_graph: FeatureDependencyGraph,
        cache_ttl_s: float = 60.0,
    ) -> None:
        self._graph = dependency_graph
        self._cache_ttl = cache_ttl_s
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._data_sources: Dict[str, Callable] = {}
        logger.info("RealTimeFeatureComputer initialized (cache_ttl=%.0fs)", cache_ttl_s)

    def register_data_source(self, name: str, fetch_fn: Callable) -> None:
        """Register a raw data source fetch function."""
        self._data_sources[name] = fetch_fn

    async def compute(self, feature_name: str, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Compute a feature on demand.

        Recursively computes upstream dependencies first, then
        applies the feature's computation function.

        Parameters
        ----------
        feature_name : str
            Name of the feature to compute.
        context : dict, optional
            Additional context passed to computation functions.

        Returns
        -------
        Any
            The computed feature value.
        """
        # Check cache
        cached = self._cache.get(feature_name)
        if cached is not None:
            ts, value = cached
            if time.time() - ts < self._cache_ttl:
                return value

        # Compute dependencies first
        deps = self._graph.get_dependencies(feature_name)
        dep_values: Dict[str, Any] = {}
        for dep in deps:
            if dep in self._data_sources:
                try:
                    dep_values[dep] = self._data_sources[dep]()
                except Exception as exc:
                    logger.error("Data source '%s' error: %s", dep, exc)
                    dep_values[dep] = None
            elif dep in self._graph._computation_fns:
                dep_val = await self.compute(dep, context)
                dep_values[dep] = dep_val
            else:
                dep_values[dep] = None

        # Compute the feature
        compute_fn = self._graph._computation_fns.get(feature_name)
        if compute_fn is None:
            logger.warning("No computation function for feature '%s'", feature_name)
            return None

        try:
            result = compute_fn(dep_values, context)
            self._cache[feature_name] = (time.time(), result)
            return result
        except Exception as exc:
            logger.error("Computation error for '%s': %s", feature_name, exc)
            return None

    async def compute_group(self, group: FeatureGroup,
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute all features in a group.

        Returns
        -------
        dict
            Mapping of feature name to computed value.
        """
        results: Dict[str, Any] = {}
        for fname in group.feature_names:
            results[fname] = await self.compute(fname, context)
        group.last_computed = time.time()
        return results

    def invalidate(self, feature_name: str) -> None:
        """Invalidate cached value for a feature and all its dependents."""
        dependents = self._graph.get_all_dependents(feature_name)
        for fname in dependents | {feature_name}:
            self._cache.pop(fname, None)

    def clear_cache(self) -> None:
        """Clear all cached feature values."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Feature Store (Redis Backend)
# ---------------------------------------------------------------------------

class FeatureStore:
    """Central feature store with Redis backend, versioning, and lineage.

    Provides a unified interface for storing, retrieving, and managing
    features used across the ACMS AI pipeline. Supports:

    - **Real-time** feature storage and retrieval
    - **Versioned** feature definitions with lineage tracking
    - **Grouped** feature management
    - **Statistical** profiling and quality scoring
    - **Freshness** monitoring and staleness alerts

    Parameters
    ----------
    redis_url : str
        Redis connection URL.
    namespace : str
        Key namespace prefix for all Redis keys.
    default_ttl : int
        Default TTL for stored feature values (seconds).
    enable_versioning : bool
        Whether to track feature version history.

    Examples
    --------
    >>> store = FeatureStore(redis_url="redis://localhost:6379/1")
    >>> await store.connect()
    >>> await store.put("btc_price", 42150.0, group="market")
    >>> price = await store.get("btc_price")
    >>> stats = await store.compute_statistics("btc_price")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/1",
        namespace: str = "acms:features",
        default_ttl: int = 3600,
        enable_versioning: bool = True,
    ) -> None:
        self._redis_url = redis_url
        self._namespace = namespace
        self._default_ttl = default_ttl
        self._enable_versioning = enable_versioning
        self._redis: Any = None

        # Local state
        self._groups: Dict[str, FeatureGroup] = {}
        self._versions: Dict[str, List[FeatureVersion]] = defaultdict(list)
        self._statistics: Dict[str, FeatureStatistics] = {}
        self._quality_scores: Dict[str, Dict[str, float]] = {}
        self._dependency_graph = FeatureDependencyGraph()
        self._freshness_monitor = FeatureFreshnessMonitor()
        self._quality_scorer = FeatureQualityScorer()
        self._computer: Optional[RealTimeFeatureComputer] = None

        # Local fallback cache
        self._local_store: Dict[str, Tuple[float, Any]] = {}

        logger.info("FeatureStore initialized (namespace=%s, ttl=%ds)", namespace, default_ttl)

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
            self._redis = await aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("FeatureStore connected to Redis at %s", self._redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); using local store", exc)
            self._redis = None

        # Initialize computer
        self._computer = RealTimeFeatureComputer(self._dependency_graph)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    def _key(self, feature_name: str) -> str:
        return f"{self._namespace}:val:{feature_name}"

    def _stats_key(self, feature_name: str) -> str:
        return f"{self._namespace}:stats:{feature_name}"

    # -- Core Operations --

    async def put(
        self,
        feature_name: str,
        value: Any,
        group: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a feature value.

        Parameters
        ----------
        feature_name : str
            Feature name.
        value : Any
            Feature value (will be JSON-serialized).
        group : str, optional
            Feature group to assign to.
        ttl : int, optional
            Custom TTL in seconds.
        """
        effective_ttl = ttl or self._default_ttl
        serialized = json.dumps(value, default=_feature_json_default)
        full_key = self._key(feature_name)

        if self._redis is not None:
            try:
                await self._redis.setex(full_key, effective_ttl, serialized)
            except Exception as exc:
                logger.warning("Redis PUT error for %s: %s", feature_name, exc)

        # Local fallback
        self._local_store[feature_name] = (time.time(), value)

        # Update freshness
        self._freshness_monitor.update(feature_name)

        # Assign to group
        if group and group not in self._groups:
            self._groups[group] = FeatureGroup(name=group)
        if group:
            self._groups[group].add_feature(feature_name)

    async def get(self, feature_name: str) -> Optional[Any]:
        """Retrieve a feature value.

        Parameters
        ----------
        feature_name : str
            Feature name.

        Returns
        -------
        Any or None
            The stored feature value, or None if not found.
        """
        # Try Redis first
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._key(feature_name))
                if raw is not None:
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis GET error for %s: %s", feature_name, exc)

        # Local fallback
        entry = self._local_store.get(feature_name)
        if entry is not None:
            ts, value = entry
            return value

        # Try computing on demand
        if self._computer:
            return await self._computer.compute(feature_name)

        return None

    async def get_batch(self, feature_names: List[str]) -> Dict[str, Any]:
        """Retrieve multiple feature values.

        Parameters
        ----------
        feature_names : list of str
            Feature names to retrieve.

        Returns
        -------
        dict
            Mapping of feature name to value (missing features omitted).
        """
        results: Dict[str, Any] = {}
        for fname in feature_names:
            val = await self.get(fname)
            if val is not None:
                results[fname] = val
        return results

    async def delete(self, feature_name: str) -> None:
        """Delete a feature value."""
        if self._redis is not None:
            try:
                await self._redis.delete(self._key(feature_name))
            except Exception as exc:
                logger.warning("Redis DEL error for %s: %s", feature_name, exc)
        self._local_store.pop(feature_name, None)

    # -- Statistics --

    async def compute_statistics(self, feature_name: str,
                                  values: Optional[np.ndarray] = None) -> FeatureStatistics:
        """Compute and store statistics for a feature.

        Parameters
        ----------
        feature_name : str
            Feature name.
        values : np.ndarray, optional
            Raw values to compute stats from. If None, will attempt
            to fetch from the store's history.

        Returns
        -------
        FeatureStatistics
            Computed statistics.
        """
        if values is None:
            val = await self.get(feature_name)
            if val is None:
                return FeatureStatistics(feature_name=feature_name)
            values = np.atleast_1d(np.asarray(val, dtype=np.float64))

        stats = FeatureStatistics.from_values(feature_name, values)
        self._statistics[feature_name] = stats

        # Persist to Redis
        if self._redis is not None:
            try:
                stats_json = json.dumps(stats.to_dict(), default=str)
                await self._redis.set(self._stats_key(feature_name), stats_json)
            except Exception as exc:
                logger.warning("Redis stats PUT error for %s: %s", feature_name, exc)

        return stats

    async def get_statistics(self, feature_name: str) -> Optional[FeatureStatistics]:
        """Retrieve stored statistics for a feature."""
        if feature_name in self._statistics:
            return self._statistics[feature_name]

        # Try Redis
        if self._redis is not None:
            try:
                raw = await self._redis.get(self._stats_key(feature_name))
                if raw is not None:
                    data = json.loads(raw)
                    return FeatureStatistics(
                        feature_name=data.get("feature_name", feature_name),
                        count=data.get("count", 0),
                        mean=data.get("mean", 0.0),
                        std=data.get("std", 0.0),
                        min_val=data.get("min", 0.0),
                        max_val=data.get("max", 0.0),
                        p25=data.get("p25", 0.0),
                        p50=data.get("p50", 0.0),
                        p75=data.get("p75", 0.0),
                        p95=data.get("p95", 0.0),
                        p99=data.get("p99", 0.0),
                        nan_count=data.get("nan_count", 0),
                        inf_count=data.get("inf_count", 0),
                        zero_count=data.get("zero_count", 0),
                    )
            except Exception as exc:
                logger.warning("Redis stats GET error for %s: %s", feature_name, exc)

        return None

    # -- Quality Scoring --

    async def score_quality(self, feature_name: str,
                             values: np.ndarray,
                             target: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Score the quality of a feature."""
        scores = self._quality_scorer.score(feature_name, values, target)
        self._quality_scores[feature_name] = scores
        return scores

    async def get_quality_scores(self, feature_name: str) -> Optional[Dict[str, float]]:
        """Retrieve quality scores for a feature."""
        return self._quality_scores.get(feature_name)

    # -- Versioning --

    async def create_version(self, feature_name: str, definition: Dict[str, Any],
                              created_by: str = "system") -> FeatureVersion:
        """Create a new version of a feature definition."""
        versions = self._versions[feature_name]
        new_version = FeatureVersion(
            feature_name=feature_name,
            version=len(versions) + 1,
            definition=definition,
            created_by=created_by,
            parent_version=versions[-1].version if versions else None,
            lineage=list(self._dependency_graph.get_dependencies(feature_name)),
        )
        versions.append(new_version)

        # Persist
        if self._redis is not None:
            try:
                ver_key = f"{self._namespace}:ver:{feature_name}"
                ver_data = json.dumps({
                    "version": new_version.version,
                    "definition": new_version.definition,
                    "created_at": new_version.created_at,
                    "fingerprint": new_version.fingerprint(),
                    "lineage": new_version.lineage,
                }, default=str)
                await self._redis.rpush(ver_key, ver_data)
            except Exception as exc:
                logger.warning("Redis version PUT error for %s: %s", feature_name, exc)

        logger.info("Created version %d for feature '%s'", new_version.version, feature_name)
        return new_version

    async def get_versions(self, feature_name: str) -> List[FeatureVersion]:
        """Get all versions of a feature."""
        return list(self._versions.get(feature_name, []))

    async def get_lineage(self, feature_name: str) -> Dict[str, Any]:
        """Get the full lineage of a feature including all upstream dependencies."""
        versions = self._versions.get(feature_name, [])
        deps = self._dependency_graph.get_dependencies(feature_name)
        all_deps = set()
        for dep in deps:
            all_deps.add(dep)
            all_deps |= self._dependency_graph.get_all_dependents(dep)

        return {
            "feature": feature_name,
            "current_version": versions[-1].version if versions else 0,
            "direct_dependencies": list(deps),
            "all_upstream": list(all_deps),
            "downstream_dependents": list(self._dependency_graph.get_dependents(feature_name)),
            "version_history": [
                {
                    "version": v.version,
                    "fingerprint": v.fingerprint(),
                    "created_at": v.created_at,
                    "lineage": v.lineage,
                }
                for v in versions
            ],
        }

    # -- Group Management --

    def create_group(self, name: str, description: str = "",
                     feature_names: Optional[List[str]] = None,
                     compute_interval_s: float = 60.0) -> FeatureGroup:
        """Create a feature group."""
        group = FeatureGroup(
            name=name,
            description=description,
            feature_names=feature_names,
            compute_interval_s=compute_interval_s,
        )
        self._groups[name] = group
        return group

    def get_group(self, name: str) -> Optional[FeatureGroup]:
        """Get a feature group by name."""
        return self._groups.get(name)

    def list_groups(self) -> List[Dict[str, Any]]:
        """List all feature groups."""
        return [g.to_dict() for g in self._groups.values()]

    async def compute_group(self, group_name: str,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compute all features in a group."""
        group = self._groups.get(group_name)
        if group is None:
            return {"error": f"Group '{group_name}' not found"}

        if self._computer is None:
            return {"error": "Feature computer not initialized"}

        return await self._computer.compute_group(group, context)

    # -- Dependency Management --

    @property
    def dependency_graph(self) -> FeatureDependencyGraph:
        return self._dependency_graph

    @property
    def freshness_monitor(self) -> FeatureFreshnessMonitor:
        return self._freshness_monitor

    @property
    def quality_scorer(self) -> FeatureQualityScorer:
        return self._quality_scorer

    @property
    def computer(self) -> Optional[RealTimeFeatureComputer]:
        return self._computer

    # -- Health --

    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the feature store."""
        redis_ok = False
        if self._redis is not None:
            try:
                await self._redis.ping()
                redis_ok = True
            except Exception:
                redis_ok = False

        freshness = self._freshness_monitor.check_freshness()

        return {
            "redis_connected": redis_ok,
            "local_store_size": len(self._local_store),
            "groups_count": len(self._groups),
            "versioned_features": len(self._versions),
            "statistics_count": len(self._statistics),
            "quality_scored": len(self._quality_scores),
            "freshness": freshness,
            "dependency_graph": self._dependency_graph.to_dict(),
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _feature_json_default(obj: Any) -> Any:
    """JSON serializer for feature store types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
