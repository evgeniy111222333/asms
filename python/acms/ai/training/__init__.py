"""
ACMS AI Training Subpackage
===========================

Comprehensive training infrastructure for the Algorithmic Crypto Management System.

Modules
-------
trainer
    Core training engine with GPU support, mixed precision, gradient accumulation,
    learning rate scheduling, early stopping, checkpointing, and profiling.

curriculum
    Curriculum learning strategies that progressively increase training difficulty
    across market regimes, time horizons, and asset counts.

online_learning
    Continuous/online learning with experience replay, concept drift detection,
    model update strategies, and A/B testing of model updates.

hyperopt
    Hyperparameter optimization using Optuna with multi-objective search,
    pruning strategies, and distributed optimization support.

walkforward_trainer
    Walk-forward training pipeline for proper time-series model evaluation
    with expanding/sliding windows and automatic retraining triggers.

distributed
    Distributed training using PyTorch DDP with multi-node coordination,
    fault tolerance, and resource monitoring.

Example
-------
>>> from acms.ai.training import Trainer, TrainingConfig
>>> from acms.ai.training.curriculum import MarketRegimeCurriculum
>>> from acms.ai.training.hyperopt import HyperoptManager
>>>
>>> config = TrainingConfig(epochs=100, learning_rate=1e-3, device="cuda")
>>> trainer = Trainer(model, config, train_loader, val_loader)
>>> trainer.fit()
"""

from acms.ai.training.trainer import (
    Trainer,
    TrainingConfig,
    TrainingState,
    TrainingLoop,
    TrainingCallback,
    EpochCallback,
    StepCallback,
    EarlyStoppingCallback,
    CheckpointCallback,
    LRSchedulerType,
)
from acms.ai.training.curriculum import (
    CurriculumScheduler,
    CurriculumStage,
    MarketRegimeCurriculum,
    TimeHorizonCurriculum,
    AssetCountCurriculum,
    DifficultyLevel,
)
from acms.ai.training.online_learning import (
    OnlineLearner,
    ExperienceReplayBuffer,
    ConceptDriftDetector,
    DriftDetectionMethod,
    UpdateStrategy,
    UpdateTrigger,
    ModelUpdateResult,
)
from acms.ai.training.hyperopt import (
    HyperoptManager,
    SearchSpace,
    MultiObjective,
    PruningStrategy,
    TrialResult,
)
from acms.ai.training.walkforward_trainer import (
    WalkForwardTrainer,
    WindowStrategy,
    WalkForwardResult,
    RetrainingTrigger,
    WindowMetrics,
)
from acms.ai.training.distributed import (
    DistributedTrainer,
    DistributedConfig,
    NodeInfo,
    ResourceMonitor,
)

__all__ = [
    # Core training
    "Trainer",
    "TrainingConfig",
    "TrainingState",
    "TrainingLoop",
    "TrainingCallback",
    "EpochCallback",
    "StepCallback",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "LRSchedulerType",
    # Curriculum learning
    "CurriculumScheduler",
    "CurriculumStage",
    "MarketRegimeCurriculum",
    "TimeHorizonCurriculum",
    "AssetCountCurriculum",
    "DifficultyLevel",
    # Online learning
    "OnlineLearner",
    "ExperienceReplayBuffer",
    "ConceptDriftDetector",
    "DriftDetectionMethod",
    "UpdateStrategy",
    "UpdateTrigger",
    "ModelUpdateResult",
    # Hyperparameter optimization
    "HyperoptManager",
    "SearchSpace",
    "MultiObjective",
    "PruningStrategy",
    "TrialResult",
    # Walk-forward training
    "WalkForwardTrainer",
    "WindowStrategy",
    "WalkForwardResult",
    "RetrainingTrigger",
    "WindowMetrics",
    # Distributed training
    "DistributedTrainer",
    "DistributedConfig",
    "NodeInfo",
    "ResourceMonitor",
]

__version__ = "1.0.0"
