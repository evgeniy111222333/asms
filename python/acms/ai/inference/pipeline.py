"""
ACMS AI Inference Pipeline
===========================

Orchestrated inference pipeline with pre/post processing, uncertainty
estimation, latency budgets, and async concurrency for the Algorithmic
Crypto Management System.

Components
----------
InferencePipeline : Main pipeline orchestrator
PreprocessorStep : Feature computation and normalization
ModelInferenceStep : GPU forward pass
PostprocessorStep : Denormalization and confidence scoring
UncertaintyEstimationStep : MC dropout and ensemble variance
PipelineConfig : Per-step latency budget configuration
PipelineProfiler : Step-level profiling and bottleneck detection
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class StepLatencyBudget:
    """Latency budget for a single pipeline step.

    Parameters
    ----------
    name : str
        Step name.
    budget_ms : float
        Target maximum latency in milliseconds.
    warning_threshold_pct : float
        Percentage of budget at which to emit a warning (0-1).
    """
    name: str
    budget_ms: float
    warning_threshold_pct: float = 0.8


@dataclass
class PipelineConfig:
    """Configuration for an inference pipeline with per-step latency budgets.

    Attributes
    ----------
    total_budget_ms : float
        End-to-end latency budget for the entire pipeline.
    step_budgets : dict
        Mapping of step name to StepLatencyBudget.
    enable_uncertainty : bool
        Whether to run the uncertainty estimation step.
    uncertainty_samples : int
        Number of MC dropout samples for uncertainty estimation.
    device : str
        Compute device (``"cuda"`` or ``"cpu"``).
    max_concurrent : int
        Maximum concurrent pipeline executions.
    """
    total_budget_ms: float = 100.0
    step_budgets: Dict[str, StepLatencyBudget] = field(default_factory=lambda: {
        "preprocess": StepLatencyBudget("preprocess", budget_ms=10.0),
        "inference": StepLatencyBudget("inference", budget_ms=60.0),
        "uncertainty": StepLatencyBudget("uncertainty", budget_ms=20.0),
        "postprocess": StepLatencyBudget("postprocess", budget_ms=10.0),
    })
    enable_uncertainty: bool = False
    uncertainty_samples: int = 30
    device: str = "cuda"
    max_concurrent: int = 16


# ---------------------------------------------------------------------------
# Pipeline Context
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Shared context passed through all pipeline steps.

    Carries features, predictions, metadata, and timing information
    from step to step.
    """
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    model_id: str = ""
    version: str = ""

    # Data flow
    raw_features: Optional[np.ndarray] = None
    raw_feature_dict: Optional[Dict[str, Any]] = None
    preprocessed_features: Optional[np.ndarray] = None
    raw_prediction: Optional[np.ndarray] = None
    postprocessed_prediction: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    uncertainty: Optional[np.ndarray] = None
    uncertainty_mean: Optional[np.ndarray] = None
    uncertainty_std: Optional[np.ndarray] = None

    # Metadata
    normalization_params: Dict[str, Any] = field(default_factory=dict)
    step_timings: Dict[str, float] = field(default_factory=dict)
    step_warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Abstract Pipeline Step
# ---------------------------------------------------------------------------

class PipelineStep(ABC):
    """Abstract base class for pipeline steps.

    Each step must implement :meth:`execute` and :meth:`name`.
    Steps receive a :class:`PipelineContext` and mutate it in place.
    """

    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute this pipeline step, mutating *context* in place."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the step name (must match a key in PipelineConfig.step_budgets)."""
        ...


# ---------------------------------------------------------------------------
# Preprocessor Step
# ---------------------------------------------------------------------------

class PreprocessorStep(PipelineStep):
    """Preprocessing step: feature computation, normalization, and validation.

    Parameters
    ----------
    feature_means : np.ndarray, optional
        Precomputed feature means for normalization.
    feature_stds : np.ndarray, optional
        Precomputed feature standard deviations for normalization.
    clip_value : float
        Absolute value at which to clip features after normalization.
    fill_na : bool
        Whether to fill NaN values with zeros before processing.
    """

    def __init__(
        self,
        feature_means: Optional[np.ndarray] = None,
        feature_stds: Optional[np.ndarray] = None,
        clip_value: float = 5.0,
        fill_na: bool = True,
    ) -> None:
        self._means = feature_means
        self._stds = feature_stds
        self._clip = clip_value
        self._fill_na = fill_na
        logger.info("PreprocessorStep initialized (clip=%.1f, fill_na=%s)", clip_value, fill_na)

    @property
    def name(self) -> str:
        return "preprocess"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute preprocessing: convert features, normalize, clip, validate."""
        t0 = time.perf_counter()

        try:
            # Resolve features from raw input
            features = context.raw_features
            if features is None and context.raw_feature_dict is not None:
                values = list(context.raw_feature_dict.values())
                features = np.array(values, dtype=np.float32)

            if features is None:
                context.error = "No features provided for preprocessing"
                return context

            features = np.asarray(features, dtype=np.float32)

            # Fill NaN
            if self._fill_na:
                nan_mask = np.isnan(features)
                if np.any(nan_mask):
                    fill_val = self._means[:features.shape[-1]] if self._means is not None else 0.0
                    features[nan_mask] = fill_val if np.isscalar(fill_val) else 0.0

            # Ensure 2D
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Z-score normalization
            if self._means is not None and self._stds is not None:
                # Adjust shapes if needed
                means = self._means[:features.shape[-1]]
                stds = self._stds[:features.shape[-1]]
                stds = np.where(stds < 1e-8, 1.0, stds)  # prevent division by zero
                features = (features - means) / stds
                context.normalization_params = {
                    "method": "zscore",
                    "means_shape": list(self._means.shape),
                    "stds_shape": list(self._stds.shape),
                }

            # Clip extreme values
            features = np.clip(features, -self._clip, self._clip)

            context.preprocessed_features = features

        except Exception as exc:
            context.error = f"Preprocessing failed: {exc}"
            logger.error("PreprocessorStep error: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        context.step_timings[self.name] = elapsed_ms
        return context


# ---------------------------------------------------------------------------
# Model Inference Step
# ---------------------------------------------------------------------------

class ModelInferenceStep(PipelineStep):
    """GPU forward pass step.

    Executes the model's forward method on preprocessed features.
    Supports PyTorch models and generic callables.

    Parameters
    ----------
    model_registry : dict
        Mapping of ``"model_id:version"`` to model objects.
    device : str
        Compute device (``"cuda"`` or ``"cpu"``).
    half_precision : bool
        Whether to use FP16 for inference.
    """

    def __init__(
        self,
        model_registry: Dict[str, Any],
        device: str = "cuda",
        half_precision: bool = False,
    ) -> None:
        self._models = model_registry
        self._device = device
        self._half = half_precision
        logger.info("ModelInferenceStep initialized (device=%s, fp16=%s)", device, half_precision)

    @property
    def name(self) -> str:
        return "inference"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute model forward pass on GPU."""
        t0 = time.perf_counter()

        if context.error:
            return context

        try:
            features = context.preprocessed_features
            if features is None:
                context.error = "No preprocessed features available for inference"
                return context

            # Resolve model
            version = context.version or "v1"
            model_key = f"{context.model_id}:{version}"
            model = self._models.get(model_key)
            if model is None:
                # Try default
                model = self._models.get(context.model_id)
            if model is None:
                context.error = f"Model not found: {model_key}"
                return context

            # Execute forward pass
            if hasattr(model, "forward"):
                import torch  # type: ignore[import-untyped]
                tensor = torch.from_numpy(features).to(self._device)
                if self._half:
                    tensor = tensor.half()
                with torch.no_grad():
                    output = model(tensor)
                if hasattr(output, "cpu"):
                    output = output.cpu().numpy()
                context.raw_prediction = np.atleast_1d(np.asarray(output, dtype=np.float32))
            elif callable(model):
                result = model(features)
                context.raw_prediction = np.atleast_1d(np.asarray(result, dtype=np.float32))
            else:
                context.error = f"Model {model_key} is not callable"
                return context

        except Exception as exc:
            context.error = f"Model inference failed: {exc}"
            logger.error("ModelInferenceStep error: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        context.step_timings[self.name] = elapsed_ms
        return context


# ---------------------------------------------------------------------------
# Postprocessor Step
# ---------------------------------------------------------------------------

class PostprocessorStep(PipelineStep):
    """Postprocessing step: denormalization, confidence scoring, and formatting.

    Parameters
    ----------
    denormalize : bool
        Whether to apply inverse normalization to predictions.
    target_means : np.ndarray, optional
        Means for denormalization.
    target_stds : np.ndarray, optional
        Standard deviations for denormalization.
    confidence_method : str
        Confidence scoring method (``"softmax"`` or ``"sigmoid"`` or ``"magnitude"``).
    """

    def __init__(
        self,
        denormalize: bool = False,
        target_means: Optional[np.ndarray] = None,
        target_stds: Optional[np.ndarray] = None,
        confidence_method: str = "magnitude",
    ) -> None:
        self._denormalize = denormalize
        self._means = target_means
        self._stds = target_stds
        self._conf_method = confidence_method
        logger.info("PostprocessorStep initialized (denorm=%s, conf=%s)", denormalize, confidence_method)

    @property
    def name(self) -> str:
        return "postprocess"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute postprocessing: denormalize, compute confidence."""
        t0 = time.perf_counter()

        if context.error:
            return context

        try:
            prediction = context.raw_prediction
            if prediction is None:
                context.error = "No raw prediction to postprocess"
                return context

            # Denormalize if needed
            if self._denormalize and self._means is not None and self._stds is not None:
                stds = np.where(self._stds < 1e-8, 1.0, self._stds)
                prediction = prediction * stds + self._means

            context.postprocessed_prediction = prediction

            # Compute confidence
            context.confidence = self._compute_confidence(prediction)

        except Exception as exc:
            context.error = f"Postprocessing failed: {exc}"
            logger.error("PostprocessorStep error: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        context.step_timings[self.name] = elapsed_ms
        return context

    def _compute_confidence(self, prediction: np.ndarray) -> float:
        """Compute a confidence score for the prediction.

        Parameters
        ----------
        prediction : np.ndarray
            Raw or denormalized prediction vector.

        Returns
        -------
        float
            Confidence in [0, 1].
        """
        flat = prediction.flatten()
        if len(flat) == 0:
            return 0.0

        if self._conf_method == "softmax":
            exp_vals = np.exp(flat - np.max(flat))
            probs = exp_vals / (np.sum(exp_vals) + 1e-8)
            return float(np.max(probs))

        if self._conf_method == "sigmoid":
            sigmoid_vals = 1.0 / (1.0 + np.exp(-flat))
            return float(np.max(sigmoid_vals))

        # Default: magnitude-based
        abs_vals = np.abs(flat)
        mean_abs = np.mean(abs_vals) + 1e-8
        std_abs = np.std(abs_vals) + 1e-8
        return float(np.clip(mean_abs / (mean_abs + std_abs), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Uncertainty Estimation Step
# ---------------------------------------------------------------------------

class UncertaintyEstimationStep(PipelineStep):
    """Uncertainty estimation via MC dropout or ensemble variance.

    Parameters
    ----------
    n_samples : int
        Number of forward passes for MC dropout.
    method : str
        Estimation method (``"mc_dropout"`` or ``"ensemble"``).
    model_registry : dict
        Model registry (needed for ensemble method).
    device : str
        Compute device.
    """

    def __init__(
        self,
        n_samples: int = 30,
        method: str = "mc_dropout",
        model_registry: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ) -> None:
        self._n_samples = n_samples
        self._method = method
        self._models = model_registry or {}
        self._device = device
        logger.info("UncertaintyEstimationStep initialized (method=%s, samples=%d)", method, n_samples)

    @property
    def name(self) -> str:
        return "uncertainty"

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Estimate prediction uncertainty."""
        t0 = time.perf_counter()

        if context.error:
            return context

        try:
            if self._method == "mc_dropout":
                await self._mc_dropout(context)
            elif self._method == "ensemble":
                await self._ensemble_variance(context)
            else:
                context.step_warnings.append(f"Unknown uncertainty method: {self._method}")
        except Exception as exc:
            context.step_warnings.append(f"Uncertainty estimation failed: {exc}")
            logger.warning("UncertaintyEstimationStep error: %s", exc)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        context.step_timings[self.name] = elapsed_ms
        return context

    async def _mc_dropout(self, context: PipelineContext) -> None:
        """Monte Carlo dropout uncertainty estimation.

        Runs multiple forward passes with dropout enabled, then
        computes mean and standard deviation of predictions.
        """
        features = context.preprocessed_features
        if features is None:
            return

        version = context.version or "v1"
        model_key = f"{context.model_id}:{version}"
        model = self._models.get(model_key)
        if model is None:
            return

        predictions: List[np.ndarray] = []
        if hasattr(model, "forward") and hasattr(model, "train"):
            import torch  # type: ignore[import-untyped]
            model.train()  # Enable dropout
            tensor = torch.from_numpy(features).to(self._device)
            for _ in range(self._n_samples):
                with torch.no_grad():
                    output = model(tensor)
                if hasattr(output, "cpu"):
                    output = output.cpu().numpy()
                predictions.append(np.atleast_1d(np.asarray(output)).flatten())
            model.eval()  # Disable dropout
        else:
            # Cannot do MC dropout without a PyTorch model
            context.step_warnings.append("MC dropout requires a PyTorch model with dropout layers")
            return

        if predictions:
            stacked = np.stack(predictions, axis=0)
            context.uncertainty = stacked
            context.uncertainty_mean = np.mean(stacked, axis=0)
            context.uncertainty_std = np.std(stacked, axis=0)

    async def _ensemble_variance(self, context: PipelineContext) -> None:
        """Ensemble-based uncertainty estimation.

        Runs inference on multiple model versions and computes variance.
        """
        features = context.preprocessed_features
        if features is None:
            return

        predictions: List[np.ndarray] = []
        for key, model in self._models.items():
            if not key.startswith(context.model_id):
                continue
            try:
                if hasattr(model, "forward"):
                    import torch  # type: ignore[import-untyped]
                    tensor = torch.from_numpy(features).to(self._device)
                    with torch.no_grad():
                        output = model(tensor)
                    if hasattr(output, "cpu"):
                        output = output.cpu().numpy()
                    predictions.append(np.atleast_1d(np.asarray(output)).flatten())
                elif callable(model):
                    predictions.append(np.atleast_1d(np.asarray(model(features))).flatten())
            except Exception as exc:
                logger.warning("Ensemble member %s failed: %s", key, exc)

        if len(predictions) >= 2:
            stacked = np.stack(predictions, axis=0)
            context.uncertainty = stacked
            context.uncertainty_mean = np.mean(stacked, axis=0)
            context.uncertainty_std = np.std(stacked, axis=0)
        else:
            context.step_warnings.append(
                f"Ensemble requires >= 2 models, found {len(predictions)}"
            )


# ---------------------------------------------------------------------------
# Pipeline Profiler
# ---------------------------------------------------------------------------

class PipelineProfiler:
    """Step-level profiler for inference pipelines.

    Collects per-step latency statistics, identifies bottlenecks,
    and emits warnings when latency budgets are exceeded.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with step budgets.
    window_size : int
        Number of recent executions to track for rolling statistics.
    """

    def __init__(self, config: PipelineConfig, window_size: int = 1000) -> None:
        self._config = config
        self._window = window_size
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._budget_violations: Dict[str, int] = defaultdict(int)
        self._total_executions: int = 0
        logger.info("PipelineProfiler initialized (window=%d)", window_size)

    def record(self, step_name: str, latency_ms: float) -> None:
        """Record a step execution latency."""
        self._history[step_name].append(latency_ms)
        if len(self._history[step_name]) > self._window:
            self._history[step_name] = self._history[step_name][-self._window:]

        # Check budget
        budget = self._config.step_budgets.get(step_name)
        if budget and latency_ms > budget.budget_ms:
            self._budget_violations[step_name] += 1
            logger.warning(
                "Step '%s' exceeded budget: %.1fms > %.1fms (violation #%d)",
                step_name, latency_ms, budget.budget_ms,
                self._budget_violations[step_name],
            )

    def record_pipeline(self, context: PipelineContext) -> None:
        """Record all step timings from a completed pipeline execution."""
        self._total_executions += 1
        for step_name, latency_ms in context.step_timings.items():
            self.record(step_name, latency_ms)

    def get_bottleneck(self) -> Optional[str]:
        """Identify the step with the highest average latency."""
        if not self._history:
            return None
        avg_latencies = {
            step: np.mean(lats) for step, lats in self._history.items() if lats
        }
        if not avg_latencies:
            return None
        return max(avg_latencies, key=avg_latencies.get)  # type: ignore[arg-type]

    def get_summary(self) -> Dict[str, Any]:
        """Return profiling summary."""
        summary: Dict[str, Any] = {
            "total_executions": self._total_executions,
            "budget_violations": dict(self._budget_violations),
            "bottleneck": self.get_bottleneck(),
        }
        for step, lats in self._history.items():
            if lats:
                arr = np.array(lats)
                budget = self._config.step_budgets.get(step)
                summary[step] = {
                    "count": len(arr),
                    "avg_ms": float(np.mean(arr)),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "max_ms": float(np.max(arr)),
                    "budget_ms": budget.budget_ms if budget else None,
                    "budget_violations": self._budget_violations.get(step, 0),
                }
        return summary


# ---------------------------------------------------------------------------
# Inference Pipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Orchestrated inference pipeline with configurable steps, latency
    budgets, and async concurrency control.

    The pipeline chains :class:`PreprocessorStep`, :class:`ModelInferenceStep`,
    optional :class:`UncertaintyEstimationStep`, and :class:`PostprocessorStep`
    in order, passing a :class:`PipelineContext` through each.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration including step budgets.
    model_registry : dict
        Mapping of model keys to model objects.
    profiler : PipelineProfiler, optional
        Step-level profiler. Created automatically if not provided.

    Examples
    --------
    >>> config = PipelineConfig(device="cuda", enable_uncertainty=True)
    >>> pipeline = InferencePipeline(config, models)
    >>> ctx = await pipeline.run(context)
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_registry: Dict[str, Any],
        profiler: Optional[PipelineProfiler] = None,
    ) -> None:
        self._config = config
        self._models = model_registry
        self._profiler = profiler or PipelineProfiler(config)
        self._semaphore = asyncio.Semaphore(config.max_concurrent)

        # Build step chain
        self._steps: List[PipelineStep] = [
            PreprocessorStep(),
            ModelInferenceStep(model_registry, device=config.device),
        ]
        if config.enable_uncertainty:
            self._steps.append(
                UncertaintyEstimationStep(
                    n_samples=config.uncertainty_samples,
                    model_registry=model_registry,
                    device=config.device,
                )
            )
        self._steps.append(PostprocessorStep())

        logger.info(
            "InferencePipeline initialized with %d steps: %s",
            len(self._steps),
            [s.name for s in self._steps],
        )

    async def run(self, context: PipelineContext) -> PipelineContext:
        """Execute the full pipeline on the given context.

        Parameters
        ----------
        context : PipelineContext
            Input context with raw features and model information.

        Returns
        -------
        PipelineContext
            Output context with predictions, confidence, and timing.
        """
        async with self._semaphore:
            t_start = time.perf_counter()

            # Check total budget deadline
            deadline = time.perf_counter() + self._config.total_budget_ms / 1000.0

            for step in self._steps:
                # Skip uncertainty if disabled and this is the uncertainty step
                if step.name == "uncertainty" and not self._config.enable_uncertainty:
                    continue

                # Skip post-uncertainty steps if we already errored
                if context.error and step.name != "postprocess":
                    continue

                t_step = time.perf_counter()
                context = await step.execute(context)
                step_ms = (time.perf_counter() - t_step) * 1000.0

                # Budget warning
                budget = self._config.step_budgets.get(step.name)
                if budget and step_ms > budget.budget_ms * budget.warning_threshold_pct:
                    if step_ms > budget.budget_ms:
                        context.step_warnings.append(
                            f"Step '{step.name}' exceeded budget: {step_ms:.1f}ms > {budget.budget_ms:.1f}ms"
                        )

                # Check deadline
                if time.perf_counter() > deadline:
                    context.step_warnings.append(
                        f"Pipeline total budget exceeded after step '{step.name}'"
                    )
                    break

            context.total_latency_ms = (time.perf_counter() - t_start) * 1000.0

            # Record in profiler
            self._profiler.record_pipeline(context)

        return context

    async def run_batch(self, contexts: List[PipelineContext]) -> List[PipelineContext]:
        """Execute the pipeline on multiple contexts concurrently.

        Respects the ``max_concurrent`` setting from the pipeline config.

        Parameters
        ----------
        contexts : list of PipelineContext
            Batch of input contexts.

        Returns
        -------
        list of PipelineContext
            Batch of output contexts.
        """
        tasks = [self.run(ctx) for ctx in contexts]
        return await asyncio.gather(*tasks, return_exceptions=False)  # type: ignore[return-value]

    @property
    def profiler(self) -> PipelineProfiler:
        """Access the pipeline profiler."""
        return self._profiler

    @property
    def steps(self) -> List[str]:
        """Return the ordered list of step names."""
        return [s.name for s in self._steps]

    def update_normalization(
        self,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
    ) -> None:
        """Update normalization parameters on the preprocessor step."""
        for step in self._steps:
            if isinstance(step, PreprocessorStep):
                step._means = feature_means
                step._stds = feature_stds
                logger.info("Updated normalization parameters on preprocessor")
                break
