"""
ACMS AI Inference Module
========================

GPU-ready, self-learning AI inference system for algorithmic crypto trading.

Submodules
----------
server : Model serving, prediction caching, and inference orchestration
pipeline : Inference pipeline with pre/post processing steps
ab_testing : Model A/B testing and traffic splitting

Key Components
--------------
- ModelServer: Real-time inference serving with GPU support
- BatchInferenceEngine: Bulk prediction processing
- PredictionCache: Redis-backed prediction caching
- InferencePipeline: Orchestrated pre/post processing
- ABTestManager: Model comparison and traffic routing
"""

from acms.ai.inference.server import (
    ModelServer,
    BatchInferenceEngine,
    PredictionCache,
    ModelWarmup,
    InferenceRequest,
    InferenceResponse,
    ModelVersion,
)
from acms.ai.inference.pipeline import (
    InferencePipeline,
    PreprocessorStep,
    ModelInferenceStep,
    PostprocessorStep,
    UncertaintyEstimationStep,
    PipelineConfig,
    PipelineProfiler,
)
from acms.ai.inference.ab_testing import (
    ABTestManager,
    TrafficSplitter,
    StatisticalSignificanceTester,
    ModelComparison,
    ABTestConfig,
    ABTestStatus,
)

__all__ = [
    # Server
    "ModelServer",
    "BatchInferenceEngine",
    "PredictionCache",
    "ModelWarmup",
    "InferenceRequest",
    "InferenceResponse",
    "ModelVersion",
    # Pipeline
    "InferencePipeline",
    "PreprocessorStep",
    "ModelInferenceStep",
    "PostprocessorStep",
    "UncertaintyEstimationStep",
    "PipelineConfig",
    "PipelineProfiler",
    # A/B Testing
    "ABTestManager",
    "TrafficSplitter",
    "StatisticalSignificanceTester",
    "ModelComparison",
    "ABTestConfig",
    "ABTestStatus",
]

__version__ = "1.0.0"
