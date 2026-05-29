"""
ACMS AI Inference Server
========================

GPU-ready model serving for real-time and batch inference in the
Algorithmic Crypto Management System.

Components
----------
ModelServer : Real-time inference serving with request queuing and prioritization
BatchInferenceEngine : Bulk prediction processing with GPU batching
PredictionCache : Redis-backed prediction caching with TTL management
ModelWarmup : Startup model warm-up for consistent latency
InferencePipeline : Pre/post processing orchestration
ModelVersion : Model version routing and management
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
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes and enums
# ---------------------------------------------------------------------------

class RequestPriority(Enum):
    """Priority levels for inference requests."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ModelStatus(Enum):
    """Model serving status."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    WARMING_UP = "warming_up"
    READY = "ready"
    ERROR = "error"
    DRAINING = "draining"


@dataclass
class ModelVersion:
    """Represents a versioned model with metadata."""
    model_id: str
    version: str
    path: str
    framework: str = "pytorch"
    device: str = "cuda"
    created_at: float = field(default_factory=time.time)
    sha256: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    is_default: bool = False

    def fingerprint(self) -> str:
        """Return a unique fingerprint for this model version."""
        raw = f"{self.model_id}:{self.version}:{self.path}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class InferenceRequest:
    """A single inference request with metadata."""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    version: Optional[str] = None
    features: Optional[np.ndarray] = None
    feature_dict: Optional[Dict[str, Any]] = None
    priority: RequestPriority = RequestPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    deadline_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enable_cache: bool = True
    enable_uncertainty: bool = False

    def cache_key(self) -> str:
        """Generate a deterministic cache key from request content."""
        parts = [self.model_id, self.version or "latest"]
        if self.features is not None:
            parts.append(hashlib.md5(self.features.tobytes()).hexdigest()[:12])
        if self.feature_dict is not None:
            serialized = json.dumps(self.feature_dict, sort_keys=True, default=str)
            parts.append(hashlib.md5(serialized.encode()).hexdigest()[:12])
        return ":".join(parts)


@dataclass
class InferenceResponse:
    """A single inference response with timing and metadata."""
    request_id: str = ""
    model_id: str = ""
    version: str = ""
    prediction: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    uncertainty: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    queue_time_ms: float = 0.0
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Prometheus-style metrics collector
# ---------------------------------------------------------------------------

class InferenceMetrics:
    """Lightweight Prometheus-compatible metrics for inference serving.

    Tracks request counts, latency histograms (bucketized), error rates,
    cache hit rates, and GPU utilization markers.
    """

    def __init__(self) -> None:
        self._request_count: Dict[str, int] = defaultdict(int)
        self._error_count: Dict[str, int] = defaultdict(int)
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._latency_buckets: Dict[str, List[float]] = defaultdict(list)
        self._latency_sum: Dict[str, float] = defaultdict(float)
        self._active_requests: int = 0
        self._batch_sizes: List[int] = []
        self._start_time: float = time.time()

    def record_request(self, model_id: str, latency_ms: float, error: bool = False,
                       cache_hit: bool = False, batch_size: int = 1) -> None:
        """Record a completed inference request."""
        self._request_count[model_id] += 1
        self._latency_sum[model_id] += latency_ms
        self._latency_buckets[model_id].append(latency_ms)
        if error:
            self._error_count[model_id] += 1
        if cache_hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        self._batch_sizes.append(batch_size)

    def increment_active(self) -> None:
        self._active_requests += 1

    def decrement_active(self) -> None:
        self._active_requests = max(0, self._active_requests - 1)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for Prometheus exposition."""
        total = sum(self._request_count.values()) or 1
        total_errors = sum(self._error_count.values())
        total_cache = self._cache_hits + self._cache_misses or 1
        uptime = time.time() - self._start_time

        per_model: Dict[str, Any] = {}
        for mid, count in self._request_count.items():
            lats = self._latency_buckets.get(mid, [])
            per_model[mid] = {
                "request_count": count,
                "error_count": self._error_count.get(mid, 0),
                "avg_latency_ms": (self._latency_sum[mid] / count) if count else 0.0,
                "p50_latency_ms": float(np.percentile(lats, 50)) if lats else 0.0,
                "p99_latency_ms": float(np.percentile(lats, 99)) if lats else 0.0,
            }

        return {
            "uptime_seconds": uptime,
            "total_requests": total - 1 + 1,
            "total_errors": total_errors,
            "error_rate": total_errors / total,
            "cache_hit_rate": self._cache_hits / total_cache,
            "active_requests": self._active_requests,
            "avg_batch_size": float(np.mean(self._batch_sizes)) if self._batch_sizes else 0.0,
            "requests_per_second": (total - 1 + 1) / max(uptime, 1.0),
            "per_model": per_model,
        }

    def prometheus_format(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        summary = self.get_summary()
        lines: List[str] = [
            "# HELP acms_inference_total Total inference requests",
            "# TYPE acms_inference_total counter",
            f"acms_inference_total {summary['total_requests']}",
            "# HELP acms_inference_errors Total inference errors",
            "# TYPE acms_inference_errors counter",
            f"acms_inference_errors {summary['total_errors']}",
            "# HELP acms_inference_error_rate Current error rate",
            "# TYPE acms_inference_error_rate gauge",
            f"acms_inference_error_rate {summary['error_rate']:.6f}",
            "# HELP acms_inference_cache_hit_rate Cache hit rate",
            "# TYPE acms_inference_cache_hit_rate gauge",
            f"acms_inference_cache_hit_rate {summary['cache_hit_rate']:.6f}",
            "# HELP acms_inference_active Active inference requests",
            "# TYPE acms_inference_active gauge",
            f"acms_inference_active {summary['active_requests']}",
            "# HELP acms_inference_rps Requests per second",
            "# TYPE acms_inference_rps gauge",
            f"acms_inference_rps {summary['requests_per_second']:.2f}",
        ]
        for mid, stats in summary["per_model"].items():
            lines.append(f'acms_inference_latency_avg_ms{{model="{mid}"}} {stats["avg_latency_ms"]:.2f}')
            lines.append(f'acms_inference_latency_p99_ms{{model="{mid}"}} {stats["p99_latency_ms"]:.2f}')
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Prediction Cache (Redis backend)
# ---------------------------------------------------------------------------

class PredictionCache:
    """Redis-backed prediction cache with TTL and size management.

    Stores inference results keyed by a deterministic hash of the input
    features, model ID, and version. Supports configurable TTL and
    maximum cache size with LRU eviction.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (e.g. ``redis://localhost:6379/0``).
    ttl_seconds : int
        Default time-to-live for cached predictions.
    max_size : int
        Maximum number of entries before LRU eviction.
    prefix : str
        Key prefix for all cache entries in Redis.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 300,
        max_size: int = 100_000,
        prefix: str = "acms:cache:pred",
    ) -> None:
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._prefix = prefix
        self._redis: Any = None
        self._local_fallback: Dict[str, Tuple[float, Any]] = {}
        self._hits = 0
        self._misses = 0
        logger.info("PredictionCache initialized (ttl=%ds, max_size=%d)", ttl_seconds, max_size)

    async def connect(self) -> None:
        """Establish Redis connection; fall back to local dict on failure."""
        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
            self._redis = await aioredis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("PredictionCache connected to Redis at %s", self._redis_url)
        except Exception as exc:
            logger.warning("Redis unavailable (%s); falling back to local cache", exc)
            self._redis = None

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached prediction by key."""
        full_key = self._full_key(key)
        if self._redis is not None:
            try:
                raw = await self._redis.get(full_key)
                if raw is not None:
                    self._hits += 1
                    return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis GET error: %s", exc)
        # local fallback
        entry = self._local_fallback.get(key)
        if entry is not None:
            ts, value = entry
            if time.time() - ts < self._ttl:
                self._hits += 1
                return value
            del self._local_fallback[key]
        self._misses += 1
        return None

    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Store a prediction in the cache."""
        full_key = self._full_key(key)
        effective_ttl = ttl or self._ttl
        serialized = json.dumps(value, default=_json_default)
        if self._redis is not None:
            try:
                await self._redis.setex(full_key, effective_ttl, serialized)
            except Exception as exc:
                logger.warning("Redis SET error: %s", exc)
        # local fallback
        self._local_fallback[key] = (time.time(), value)
        if len(self._local_fallback) > self._max_size:
            oldest_key = min(self._local_fallback, key=lambda k: self._local_fallback[k][0])
            del self._local_fallback[oldest_key]

    async def delete(self, key: str) -> None:
        """Remove a cached prediction."""
        full_key = self._full_key(key)
        if self._redis is not None:
            try:
                await self._redis.delete(full_key)
            except Exception as exc:
                logger.warning("Redis DEL error: %s", exc)
        self._local_fallback.pop(key, None)

    async def clear(self) -> None:
        """Clear all cached predictions."""
        if self._redis is not None:
            try:
                keys = await self._redis.keys(f"{self._prefix}:*")
                if keys:
                    await self._redis.delete(*keys)
            except Exception as exc:
                logger.warning("Redis CLEAR error: %s", exc)
        self._local_fallback.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "local_size": len(self._local_fallback),
            "backend": "redis" if self._redis else "local",
        }


# ---------------------------------------------------------------------------
# Model Warm-up
# ---------------------------------------------------------------------------

class ModelWarmup:
    """Warm-up engine that runs dummy inference passes on startup.

    Ensures GPU kernels are compiled, memory is allocated, and the model
    is ready for production latency SLAs before accepting live traffic.

    Parameters
    ----------
    warmup_iterations : int
        Number of warm-up forward passes.
    warmup_timeout_s : float
        Maximum seconds to wait for warm-up completion.
    input_shapes : dict
        Mapping of model_id to expected input shape tuples.
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        warmup_timeout_s: float = 120.0,
        input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> None:
        self._iterations = warmup_iterations
        self._timeout = warmup_timeout_s
        self._input_shapes = input_shapes or {}
        self._warmup_results: Dict[str, Dict[str, Any]] = {}
        logger.info("ModelWarmup configured (iterations=%d)", warmup_iterations)

    def register_shape(self, model_id: str, shape: Tuple[int, ...]) -> None:
        """Register an expected input shape for a model."""
        self._input_shapes[model_id] = shape

    async def warmup_model(self, model_id: str, model: Any, device: str = "cuda") -> Dict[str, Any]:
        """Execute warm-up forward passes for a single model.

        Parameters
        ----------
        model_id : str
            Identifier of the model being warmed up.
        model : Any
            The loaded model object (must support ``__call__`` or ``forward``).
        device : str
            Device string (e.g. ``"cuda"``, ``"cpu"``).

        Returns
        -------
        dict
            Warm-up result with latency statistics.
        """
        shape = self._input_shapes.get(model_id, (1, 64))
        logger.info("Warming up model %s with shape %s on %s", model_id, shape, device)

        latencies: List[float] = []
        try:
            for i in range(self._iterations):
                dummy = np.random.randn(*shape).astype(np.float32)
                t0 = time.perf_counter()
                try:
                    if hasattr(model, "forward"):
                        import torch  # type: ignore[import-untyped]
                        tensor = torch.from_numpy(dummy).to(device)
                        with torch.no_grad():
                            _ = model(tensor)
                    elif callable(model):
                        _ = model(dummy)
                except Exception as exc:
                    logger.warning("Warmup iteration %d error: %s", i, exc)
                elapsed = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed)

            result = {
                "status": "completed",
                "iterations": self._iterations,
                "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
                "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
                "max_latency_ms": float(np.max(latencies)) if latencies else 0.0,
            }
        except Exception as exc:
            result = {"status": "failed", "error": str(exc)}
            logger.error("Warmup failed for %s: %s", model_id, exc)

        self._warmup_results[model_id] = result
        logger.info("Warmup complete for %s: %s", model_id, result)
        return result

    async def warmup_all(self, models: Dict[str, Any], device: str = "cuda") -> Dict[str, Dict[str, Any]]:
        """Warm up all registered models concurrently."""
        tasks = {
            mid: self.warmup_model(mid, m, device)
            for mid, m in models.items()
        }
        results = {}
        for mid, task in tasks.items():
            results[mid] = await task
        return results

    @property
    def results(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._warmup_results)


# ---------------------------------------------------------------------------
# Request Queue with Priority
# ---------------------------------------------------------------------------

class RequestQueue:
    """Priority-based request queue for inference serving.

    Requests are ordered by priority (CRITICAL > HIGH > NORMAL > LOW)
    and then by submission time (FIFO within same priority).

    Parameters
    ----------
    max_size : int
        Maximum queue size before rejecting requests.
    sla_timeout_ms : float
        Default SLA timeout in milliseconds.
    """

    def __init__(self, max_size: int = 10_000, sla_timeout_ms: float = 200.0) -> None:
        self._max_size = max_size
        self._sla_ms = sla_timeout_ms
        self._queues: Dict[RequestPriority, List[InferenceRequest]] = defaultdict(list)
        self._dropped: int = 0
        self._timed_out: int = 0
        self._lock = asyncio.Lock()

    async def enqueue(self, request: InferenceRequest) -> bool:
        """Add a request to the queue. Returns False if queue is full."""
        async with self._lock:
            total = sum(len(q) for q in self._queues.values())
            if total >= self._max_size:
                self._dropped += 1
                logger.warning("Request queue full; dropping request %s", request.request_id)
                return False
            self._queues[request.priority].append(request)
            return True

    async def dequeue(self) -> Optional[InferenceRequest]:
        """Dequeue the highest-priority, oldest request."""
        async with self._lock:
            for priority in [RequestPriority.CRITICAL, RequestPriority.HIGH,
                             RequestPriority.NORMAL, RequestPriority.LOW]:
                queue = self._queues[priority]
                while queue:
                    req = queue.pop(0)
                    # Check SLA deadline
                    if req.deadline_ms is not None:
                        elapsed = (time.time() - req.timestamp) * 1000.0
                        if elapsed > req.deadline_ms:
                            self._timed_out += 1
                            continue
                    return req
        return None

    async def dequeue_batch(self, max_batch: int = 32) -> List[InferenceRequest]:
        """Dequeue up to ``max_batch`` requests for batch inference."""
        batch: List[InferenceRequest] = []
        while len(batch) < max_batch:
            req = await self.dequeue()
            if req is None:
                break
            batch.append(req)
        return batch

    @property
    def size(self) -> int:
        return sum(len(q) for q in self._queues.values())

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_size": self.size,
            "dropped": self._dropped,
            "timed_out": self._timed_out,
            "by_priority": {p.name: len(q) for p, q in self._queues.items()},
        }


# ---------------------------------------------------------------------------
# Model Server
# ---------------------------------------------------------------------------

class ModelServer:
    """Real-time inference server with GPU support, request queuing,
    latency SLA enforcement, and model version routing.

    Parameters
    ----------
    device : str
        Target device for inference (``"cuda"`` or ``"cpu"``).
    max_batch_size : int
        Maximum batch size for GPU inference.
    sla_latency_ms : float
        Target SLA latency in milliseconds per request.
    enable_cache : bool
        Whether to enable prediction caching.
    cache_ttl : int
        Default cache TTL in seconds.

    Examples
    --------
    >>> server = ModelServer(device="cuda", sla_latency_ms=50.0)
    >>> await server.start()
    >>> resp = await server.predict(InferenceRequest(model_id="btc_v1", features=features))
    """

    def __init__(
        self,
        device: str = "cuda",
        max_batch_size: int = 32,
        sla_latency_ms: float = 100.0,
        enable_cache: bool = True,
        cache_ttl: int = 300,
    ) -> None:
        self._device = device
        self._max_batch = max_batch_size
        self._sla_ms = sla_latency_ms
        self._models: Dict[str, Any] = {}
        self._model_versions: Dict[str, Dict[str, ModelVersion]] = defaultdict(dict)
        self._default_versions: Dict[str, str] = {}
        self._status: Dict[str, ModelStatus] = {}
        self._queue = RequestQueue(sla_timeout_ms=sla_latency_ms)
        self._metrics = InferenceMetrics()
        self._cache = PredictionCache(ttl_seconds=cache_ttl) if enable_cache else None
        self._warmup = ModelWarmup()
        self._running = False
        self._server_task: Optional[asyncio.Task] = None
        logger.info("ModelServer initialized (device=%s, sla=%.1fms)", device, sla_latency_ms)

    # -- Model Management --

    def register_model(
        self,
        model_id: str,
        model: Any,
        version: str = "v1",
        is_default: bool = True,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Register a model for serving.

        Parameters
        ----------
        model_id : str
            Logical model identifier.
        model : Any
            The model object (PyTorch module or callable).
        version : str
            Version string for this model.
        is_default : bool
            Whether this version is the default.
        input_shape : tuple, optional
            Expected input shape for warm-up.
        """
        key = f"{model_id}:{version}"
        self._models[key] = model
        mv = ModelVersion(model_id=model_id, version=version, path=key, is_default=is_default)
        self._model_versions[model_id][version] = mv
        self._status[key] = ModelStatus.NOT_LOADED
        if is_default:
            self._default_versions[model_id] = version
        if input_shape:
            self._warmup.register_shape(model_id, input_shape)
        logger.info("Registered model %s version %s (default=%s)", model_id, version, is_default)

    def get_model(self, model_id: str, version: Optional[str] = None) -> Optional[Any]:
        """Retrieve a model by ID and optional version."""
        ver = version or self._default_versions.get(model_id)
        if ver is None:
            # Return any available version
            versions = self._model_versions.get(model_id, {})
            if versions:
                ver = next(iter(versions))
            else:
                return None
        return self._models.get(f"{model_id}:{ver}")

    def set_default_version(self, model_id: str, version: str) -> None:
        """Change the default version for a model."""
        if version in self._model_versions.get(model_id, {}):
            self._default_versions[model_id] = version
            logger.info("Default version for %s set to %s", model_id, version)
        else:
            raise ValueError(f"Version {version} not found for model {model_id}")

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all registered models and their versions."""
        result: Dict[str, Dict[str, Any]] = {}
        for mid, versions in self._model_versions.items():
            default = self._default_versions.get(mid, "unknown")
            result[mid] = {
                "default_version": default,
                "versions": {
                    v: {"fingerprint": mv.fingerprint(), "is_default": mv.is_default}
                    for v, mv in versions.items()
                },
            }
        return result

    # -- Server Lifecycle --

    async def start(self) -> None:
        """Start the inference server: connect cache, warm up models, begin processing."""
        if self._running:
            logger.warning("ModelServer already running")
            return
        self._running = True
        if self._cache:
            await self._cache.connect()
        # Warm up models
        warmup_models = {}
        for key, model in self._models.items():
            mid, ver = key.split(":", 1)
            warmup_models[mid] = model
            self._status[key] = ModelStatus.WARMING_UP
        if warmup_models:
            await self._warmup.warmup_all(warmup_models, self._device)
        for key in self._models:
            self._status[key] = ModelStatus.READY
        # Start queue processor
        self._server_task = asyncio.create_task(self._process_queue_loop())
        logger.info("ModelServer started with %d models", len(self._models))

    async def stop(self) -> None:
        """Gracefully stop the inference server."""
        self._running = False
        for key in self._models:
            self._status[key] = ModelStatus.DRAINING
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        if self._cache:
            await self._cache.disconnect()
        logger.info("ModelServer stopped")

    # -- Inference --

    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Execute a single inference request.

        Attempts cache lookup first, then falls through to model inference
        with pre/post processing and latency tracking.
        """
        self._metrics.increment_active()
        t_start = time.perf_counter()
        response = InferenceResponse(request_id=request.request_id, model_id=request.model_id)

        try:
            # Cache lookup
            if self._cache and request.enable_cache:
                cache_key = request.cache_key()
                cached = await self._cache.get(cache_key)
                if cached is not None:
                    response.prediction = np.array(cached.get("prediction", []))
                    response.confidence = cached.get("confidence")
                    response.cache_hit = True
                    response.latency_ms = (time.perf_counter() - t_start) * 1000.0
                    self._metrics.record_request(request.model_id, response.latency_ms, cache_hit=True)
                    return response

            # Queue request
            enqueue_start = time.perf_counter()
            queued = await self._queue.enqueue(request)
            if not queued:
                response.error = "Queue full; request dropped"
                response.latency_ms = (time.perf_counter() - t_start) * 1000.0
                self._metrics.record_request(request.model_id, response.latency_ms, error=True)
                return response

            # Wait for processing (simplified - in production would use asyncio events)
            result = await self._execute_inference(request)
            response = result
            response.queue_time_ms = (enqueue_start - request.timestamp) * 1000.0

            # Store in cache
            if self._cache and request.enable_cache and not response.cache_hit:
                cache_data = {
                    "prediction": response.prediction.tolist() if response.prediction is not None else [],
                    "confidence": response.confidence,
                    "version": response.version,
                }
                await self._cache.set(request.cache_key(), cache_data)

            response.latency_ms = (time.perf_counter() - t_start) * 1000.0

            # SLA check
            if response.latency_ms > self._sla_ms:
                logger.warning("SLA breach: %.1fms > %.1fms for %s",
                               response.latency_ms, self._sla_ms, request.request_id)

            self._metrics.record_request(
                request.model_id, response.latency_ms,
                error=response.error is not None,
                cache_hit=response.cache_hit,
            )
        except Exception as exc:
            response.error = str(exc)
            response.latency_ms = (time.perf_counter() - t_start) * 1000.0
            self._metrics.record_request(request.model_id, response.latency_ms, error=True)
            logger.error("Inference error for %s: %s", request.request_id, exc)
        finally:
            self._metrics.decrement_active()

        return response

    async def _execute_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Execute the actual model inference."""
        t0 = time.perf_counter()
        response = InferenceResponse(request_id=request.request_id, model_id=request.model_id)

        version = request.version or self._default_versions.get(request.model_id, "v1")
        response.version = version
        model = self.get_model(request.model_id, version)

        if model is None:
            response.error = f"Model {request.model_id}:{version} not found"
            return response

        key = f"{request.model_id}:{version}"
        if self._status.get(key) != ModelStatus.READY:
            response.error = f"Model {key} not ready (status={self._status.get(key)})"
            return response

        # Preprocess
        t_pre = time.perf_counter()
        features = request.features
        if features is None and request.feature_dict is not None:
            features = np.array(list(request.feature_dict.values()), dtype=np.float32)
        if features is None:
            response.error = "No features provided"
            return response
        if features.ndim == 1:
            features = features.reshape(1, -1)
        response.preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

        # Inference
        t_inf = time.perf_counter()
        try:
            if hasattr(model, "forward"):
                import torch  # type: ignore[import-untyped]
                tensor = torch.from_numpy(features).to(self._device)
                with torch.no_grad():
                    output = model(tensor)
                if hasattr(output, "cpu"):
                    output = output.cpu().numpy()
                response.prediction = np.atleast_1d(np.asarray(output))
            elif callable(model):
                response.prediction = np.atleast_1d(np.asarray(model(features)))
        except Exception as exc:
            response.error = f"Inference failed: {exc}"
            response.inference_ms = (time.perf_counter() - t_inf) * 1000.0
            return response
        response.inference_ms = (time.perf_counter() - t_inf) * 1000.0

        # Postprocess - compute confidence
        t_post = time.perf_counter()
        if response.prediction is not None:
            if response.prediction.ndim >= 1 and len(response.prediction) > 0:
                abs_vals = np.abs(response.prediction.flatten())
                response.confidence = float(min(1.0, np.max(abs_vals) / (np.max(abs_vals) + np.std(abs_vals) + 1e-8)))
        response.postprocess_ms = (time.perf_counter() - t_post) * 1000.0
        return response

    async def _process_queue_loop(self) -> None:
        """Background loop that processes queued requests."""
        while self._running:
            try:
                batch = await self._queue.dequeue_batch(max_batch=self._max_batch)
                if not batch:
                    await asyncio.sleep(0.001)
                    continue
                for req in batch:
                    await self._execute_inference(req)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Queue processor error: %s", exc)
                await asyncio.sleep(0.01)

    # -- Health Check --

    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the inference server."""
        models_health = {}
        for key, status in self._status.items():
            models_health[key] = {
                "status": status.value,
                "warmup": self._warmup.results.get(key.split(":")[0], {}),
            }

        return {
            "server": "healthy" if self._running else "stopped",
            "device": self._device,
            "models_registered": len(self._models),
            "queue": self._queue.stats,
            "cache": self._cache.stats if self._cache else {"enabled": False},
            "metrics": self._metrics.get_summary(),
            "models": models_health,
            "timestamp": time.time(),
        }

    # -- Metrics --

    @property
    def metrics(self) -> InferenceMetrics:
        return self._metrics

    def prometheus_metrics(self) -> str:
        """Export Prometheus-format metrics."""
        return self._metrics.prometheus_format()


# ---------------------------------------------------------------------------
# Batch Inference Engine
# ---------------------------------------------------------------------------

class BatchInferenceEngine:
    """Bulk prediction engine that batches requests for GPU efficiency.

    Automatically groups incoming requests by model and version,
    batches their features into a single GPU forward pass, and
    distributes results back to individual requestors.

    Parameters
    ----------
    server : ModelServer
        The underlying model server for inference execution.
    max_batch_size : int
        Maximum number of requests in a single batch.
    batch_timeout_ms : float
        Maximum time to wait before dispatching a partial batch.
    """

    def __init__(
        self,
        server: ModelServer,
        max_batch_size: int = 64,
        batch_timeout_ms: float = 50.0,
    ) -> None:
        self._server = server
        self._max_batch = max_batch_size
        self._timeout_ms = batch_timeout_ms
        self._pending: Dict[str, List[Tuple[InferenceRequest, asyncio.Future]]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._running = False
        self._batch_task: Optional[asyncio.Task] = None
        logger.info("BatchInferenceEngine initialized (max_batch=%d, timeout=%.1fms)",
                     max_batch_size, batch_timeout_ms)

    async def start(self) -> None:
        """Start the batch processing loop."""
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop the batch processing loop."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

    async def predict_batch(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Execute a list of inference requests as a batch."""
        futures: List[asyncio.Future] = []
        async with self._lock:
            for req in requests:
                group_key = f"{req.model_id}:{req.version or 'default'}"
                future: asyncio.Future = asyncio.get_event_loop().create_future()
                self._pending[group_key].append((req, future))
                futures.append(future)
        results = await asyncio.gather(*futures, return_exceptions=True)
        responses: List[InferenceResponse] = []
        for r in results:
            if isinstance(r, Exception):
                responses.append(InferenceResponse(error=str(r)))
            else:
                responses.append(r)
        return responses

    async def _batch_loop(self) -> None:
        """Background loop that dispatches accumulated requests in batches."""
        while self._running:
            try:
                await asyncio.sleep(self._timeout_ms / 1000.0)
                async with self._lock:
                    if not self._pending:
                        continue
                    # Process each model group
                    for group_key, items in list(self._pending.items()):
                        if not items:
                            continue
                        batch = items[:self._max_batch]
                        self._pending[group_key] = items[self._max_batch:]
                        asyncio.create_task(self._process_batch(group_key, batch))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Batch loop error: %s", exc)

    async def _process_batch(
        self,
        group_key: str,
        batch: List[Tuple[InferenceRequest, asyncio.Future]],
    ) -> None:
        """Process a single batch of requests for one model group."""
        t0 = time.perf_counter()
        try:
            # Stack features
            feature_arrays = []
            for req, _ in batch:
                if req.features is not None:
                    feature_arrays.append(req.features.flatten())
                elif req.feature_dict is not None:
                    feature_arrays.append(np.array(list(req.feature_dict.values()), dtype=np.float32))

            if not feature_arrays:
                for req, future in batch:
                    if not future.done():
                        future.set_result(InferenceResponse(request_id=req.request_id, error="No features"))
                return

            # Pad to same length
            max_len = max(len(f) for f in feature_arrays)
            padded = np.zeros((len(feature_arrays), max_len), dtype=np.float32)
            for i, arr in enumerate(feature_arrays):
                padded[i, :len(arr)] = arr

            # Single batch inference
            mid, ver = group_key.split(":", 1) if ":" in group_key else (group_key, None)
            model = self._server.get_model(mid, ver if ver != "default" else None)
            if model is None:
                for req, future in batch:
                    if not future.done():
                        future.set_result(InferenceResponse(request_id=req.request_id, error="Model not found"))
                return

            if hasattr(model, "forward"):
                import torch  # type: ignore[import-untyped]
                tensor = torch.from_numpy(padded).to(self._server._device)
                with torch.no_grad():
                    output = model(tensor)
                predictions = output.cpu().numpy() if hasattr(output, "cpu") else np.asarray(output)
            elif callable(model):
                predictions = np.asarray(model(padded))
            else:
                predictions = np.zeros((len(batch), 1))

            # Distribute results
            for i, (req, future) in enumerate(batch):
                if i < len(predictions):
                    pred = predictions[i]
                else:
                    pred = np.array([0.0])
                resp = InferenceResponse(
                    request_id=req.request_id,
                    model_id=mid,
                    version=ver or "default",
                    prediction=np.atleast_1d(pred),
                    inference_ms=(time.perf_counter() - t0) * 1000.0,
                )
                if not future.done():
                    future.set_result(resp)

        except Exception as exc:
            logger.error("Batch processing error for %s: %s", group_key, exc)
            for req, future in batch:
                if not future.done():
                    future.set_result(InferenceResponse(request_id=req.request_id, error=str(exc)))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
