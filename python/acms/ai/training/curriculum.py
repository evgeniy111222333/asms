"""
Curriculum Learning for ACMS
==============================

Progressive training strategies that increase difficulty as the model improves.
Designed specifically for cryptocurrency trading, with curricula for market
regimes, time horizons, and asset counts.

Features
--------
- CurriculumScheduler that orchestrates progression across stages
- MarketRegimeCurriculum: calm → trending → volatile → crisis
- TimeHorizonCurriculum: short-term → medium-term → long-term
- AssetCountCurriculum: single asset → multi-asset → full portfolio
- Adaptive progression based on model performance metrics
- Difficulty metrics and stage transition criteria
- Seamless integration with the Trainer class
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class DifficultyLevel(IntEnum):
    """Numeric difficulty levels for curriculum stages."""

    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class CurriculumStage:
    """A single stage in a curriculum.

    Attributes
    ----------
    name : str
        Human-readable stage name.
    difficulty : DifficultyLevel
        Numeric difficulty level.
    description : str
        What this stage entails.
    data_config : Dict[str, Any]
        Configuration for data selection/filtering at this stage.
    training_config_overrides : Dict[str, Any]
        Optional overrides for the TrainingConfig at this stage.
    min_epochs : int
        Minimum epochs before allowing stage transition.
    performance_threshold : float
        Metric threshold required for transition (e.g. val_loss < threshold).
    performance_metric : str
        Which metric to check for transition.
    performance_mode : str
        'min' or 'max' — whether lower or higher is better.
    """

    name: str
    difficulty: DifficultyLevel
    description: str = ""
    data_config: Dict[str, Any] = field(default_factory=dict)
    training_config_overrides: Dict[str, Any] = field(default_factory=dict)
    min_epochs: int = 3
    performance_threshold: float = 0.5
    performance_metric: str = "val_loss"
    performance_mode: str = "min"

    def is_performance_met(self, metrics: Dict[str, float]) -> bool:
        """Check whether current metrics meet the transition threshold."""
        value = metrics.get(self.performance_metric)
        if value is None:
            return False
        if self.performance_mode == "min":
            return value <= self.performance_threshold
        else:
            return value >= self.performance_threshold


@dataclass
class ProgressionCriteria:
    """Criteria that must all be satisfied for a stage transition.

    Attributes
    ----------
    min_epochs_in_stage : int
        Minimum epochs spent in current stage.
    performance_threshold : Optional[float]
        Metric threshold that must be met.
    performance_metric : str
        Metric name to evaluate.
    performance_mode : str
        'min' or 'max'.
    consecutive_epochs_met : int
        Number of consecutive epochs the threshold must be met.
    max_epochs_in_stage : int
        Hard cap — force transition after this many epochs.
    """

    min_epochs_in_stage: int = 3
    performance_threshold: Optional[float] = None
    performance_metric: str = "val_loss"
    performance_mode: str = "min"
    consecutive_epochs_met: int = 2
    max_epochs_in_stage: int = 50

    def should_advance(
        self,
        epochs_in_stage: int,
        recent_metrics: List[Dict[str, float]],
    ) -> bool:
        """Decide whether the model should advance to the next stage.

        Parameters
        ----------
        epochs_in_stage : int
            Number of epochs already spent in the current stage.
        recent_metrics : List[Dict[str, float]]
            Metrics from recent epochs (most recent last).

        Returns
        -------
        bool
            True if the model should advance.
        """
        # Force advance after max epochs
        if epochs_in_stage >= self.max_epochs_in_stage:
            logger.info("Max epochs in stage reached — forcing advancement.")
            return True

        # Must complete minimum epochs
        if epochs_in_stage < self.min_epochs_in_stage:
            return False

        # If no threshold, advance after min epochs
        if self.performance_threshold is None:
            return True

        # Check consecutive metric satisfaction
        if len(recent_metrics) < self.consecutive_epochs_met:
            return False

        recent = recent_metrics[-self.consecutive_epochs_met :]
        all_met = True
        for m in recent:
            value = m.get(self.performance_metric)
            if value is None:
                all_met = False
                break
            if self.performance_mode == "min":
                if value > self.performance_threshold:
                    all_met = False
                    break
            else:
                if value < self.performance_threshold:
                    all_met = False
                    break

        return all_met


# ---------------------------------------------------------------------------
# Base Curriculum
# ---------------------------------------------------------------------------


class BaseCurriculum(ABC):
    """Abstract base class for all curriculum strategies.

    Subclasses must define the stages and provide data filtering logic
    for each stage.
    """

    def __init__(self) -> None:
        self._stages: List[CurriculumStage] = []
        self._current_stage_idx: int = 0
        self._epochs_in_stage: int = 0
        self._recent_metrics: List[Dict[str, float]] = []
        self._progression_criteria = self._build_progression_criteria()
        self._build_stages()

    @property
    def current_stage(self) -> CurriculumStage:
        """The current curriculum stage."""
        return self._stages[self._current_stage_idx]

    @property
    def current_stage_index(self) -> int:
        """Zero-based index of the current stage."""
        return self._current_stage_idx

    @property
    def num_stages(self) -> int:
        """Total number of curriculum stages."""
        return len(self._stages)

    @property
    def is_final_stage(self) -> bool:
        """Whether the model is at the last curriculum stage."""
        return self._current_stage_idx >= len(self._stages) - 1

    @property
    def difficulty(self) -> DifficultyLevel:
        """Current difficulty level."""
        return self.current_stage.difficulty

    @abstractmethod
    def _build_stages(self) -> None:
        """Define the curriculum stages. Must populate self._stages."""

    @abstractmethod
    def _build_progression_criteria(self) -> ProgressionCriteria:
        """Define the default progression criteria."""

    @abstractmethod
    def filter_data(self, dataset: Dataset, stage: CurriculumStage) -> Dataset:
        """Filter or modify the dataset for the given stage.

        Parameters
        ----------
        dataset : Dataset
            The full training dataset.
        stage : CurriculumStage
            The stage to filter for.

        Returns
        -------
        Dataset
            The filtered/subset dataset for this stage.
        """

    def on_epoch_end(self, metrics: Dict[str, float]) -> bool:
        """Called at the end of each epoch; checks for stage transition.

        Parameters
        ----------
        metrics : Dict[str, float]
            Epoch metrics.

        Returns
        -------
        bool
            True if the stage was advanced.
        """
        self._epochs_in_stage += 1
        self._recent_metrics.append(metrics)

        if self.is_final_stage:
            return False

        should_advance = self._progression_criteria.should_advance(
            self._epochs_in_stage, self._recent_metrics
        )

        if should_advance:
            self._advance_stage()
            return True

        return False

    def _advance_stage(self) -> None:
        """Move to the next curriculum stage."""
        old_stage = self.current_stage.name
        self._current_stage_idx += 1
        self._epochs_in_stage = 0
        self._recent_metrics.clear()
        new_stage = self.current_stage.name
        logger.info(f"Curriculum advanced: '{old_stage}' → '{new_stage}'")

    def get_data_loader(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        **loader_kwargs: Any,
    ) -> DataLoader:
        """Get a DataLoader for the current curriculum stage.

        Parameters
        ----------
        dataset : Dataset
            The full training dataset.
        batch_size : int
            Batch size for the loader.
        shuffle : bool
            Whether to shuffle.
        **loader_kwargs
            Extra DataLoader arguments.

        Returns
        -------
        DataLoader
            Filtered data loader for the current stage.
        """
        filtered = self.filter_data(dataset, self.current_stage)
        return DataLoader(filtered, batch_size=batch_size, shuffle=shuffle, **loader_kwargs)

    def get_training_config_overrides(self) -> Dict[str, Any]:
        """Get TrainingConfig overrides for the current stage."""
        return self.current_stage.training_config_overrides

    def reset(self) -> None:
        """Reset the curriculum back to the first stage."""
        self._current_stage_idx = 0
        self._epochs_in_stage = 0
        self._recent_metrics.clear()

    def get_progress(self) -> Dict[str, Any]:
        """Get a summary of curriculum progress."""
        return {
            "current_stage": self.current_stage.name,
            "current_difficulty": self.current_stage.difficulty.name,
            "stage_index": self._current_stage_idx,
            "total_stages": self.num_stages,
            "epochs_in_stage": self._epochs_in_stage,
            "is_final_stage": self.is_final_stage,
            "progress_pct": (self._current_stage_idx / max(1, self.num_stages - 1)) * 100,
        }


# ---------------------------------------------------------------------------
# Market Regime Curriculum
# ---------------------------------------------------------------------------


class MarketRegimeCurriculum(BaseCurriculum):
    """Curriculum that progresses through market regimes.

    Stages progress from calm, low-volatility markets through trending
    and volatile markets, up to crisis scenarios. This mirrors how human
    traders typically learn — first understanding calm markets, then
    progressively handling more chaotic conditions.

    Expected data_config keys
    -------------------------
    volatility_max : float
        Maximum allowed volatility (std of returns) for the stage.
    volatility_min : float
        Minimum volatility for the stage.
    trend_strength_max : float
        Maximum absolute trend strength.
    drawdown_max : float
        Maximum drawdown allowed.
    """

    def _build_stages(self) -> None:
        self._stages = [
            CurriculumStage(
                name="calm_markets",
                difficulty=DifficultyLevel.BEGINNER,
                description="Low-volatility, mean-reverting markets",
                data_config={
                    "volatility_max": 0.01,
                    "trend_strength_max": 0.1,
                    "drawdown_max": 0.05,
                },
                training_config_overrides={"learning_rate": 1e-3},
                min_epochs=3,
                performance_threshold=0.3,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="trending_markets",
                difficulty=DifficultyLevel.EASY,
                description="Markets with clear directional trends",
                data_config={
                    "volatility_max": 0.03,
                    "trend_strength_max": 0.5,
                    "drawdown_max": 0.10,
                },
                training_config_overrides={"learning_rate": 8e-4},
                min_epochs=5,
                performance_threshold=0.4,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="volatile_markets",
                difficulty=DifficultyLevel.MEDIUM,
                description="High-volatility with frequent regime changes",
                data_config={
                    "volatility_max": 0.08,
                    "trend_strength_max": 0.8,
                    "drawdown_max": 0.20,
                },
                training_config_overrides={"learning_rate": 5e-4},
                min_epochs=5,
                performance_threshold=0.5,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="crisis_markets",
                difficulty=DifficultyLevel.HARD,
                description="Extreme volatility, flash crashes, black swans",
                data_config={
                    "volatility_max": 0.20,
                    "trend_strength_max": 1.0,
                    "drawdown_max": 0.50,
                },
                training_config_overrides={"learning_rate": 3e-4},
                min_epochs=5,
                performance_threshold=0.7,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="all_regimes",
                difficulty=DifficultyLevel.EXPERT,
                description="Full market regime spectrum",
                data_config={
                    "volatility_max": float("inf"),
                    "trend_strength_max": float("inf"),
                    "drawdown_max": float("inf"),
                },
                training_config_overrides={"learning_rate": 1e-4},
                min_epochs=10,
                performance_threshold=0.6,
                performance_metric="val_loss",
                performance_mode="min",
            ),
        ]

    def _build_progression_criteria(self) -> ProgressionCriteria:
        return ProgressionCriteria(
            min_epochs_in_stage=3,
            performance_threshold=None,  # Set per-stage
            performance_metric="val_loss",
            performance_mode="min",
            consecutive_epochs_met=2,
            max_epochs_in_stage=50,
        )

    def filter_data(self, dataset: Dataset, stage: CurriculumStage) -> Dataset:
        """Filter dataset to only include data matching the regime criteria.

        Assumes the dataset items have a 'metadata' attribute or are tuples
        where volatility/trend/drawdown info can be extracted. Falls back to
        index-based filtering if metadata is unavailable.
        """
        config = stage.data_config
        vol_max = config.get("volatility_max", float("inf"))

        # Try metadata-based filtering
        try:
            indices: List[int] = []
            for i in range(len(dataset)):
                item = dataset[i]
                # Attempt to extract metadata
                meta = item[2] if isinstance(item, (tuple, list)) and len(item) > 2 else None
                if isinstance(item, dict) and "metadata" in item:
                    meta = item["metadata"]

                if meta is not None and isinstance(meta, dict):
                    vol = meta.get("volatility", 0)
                    if vol <= vol_max:
                        indices.append(i)
                else:
                    # No metadata — include all
                    indices.append(i)

            if indices:
                return Subset(dataset, indices)
            else:
                logger.warning(
                    f"No data matched regime filter for stage '{stage.name}'; "
                    f"using full dataset."
                )
                return dataset

        except Exception as e:
            logger.warning(f"Regime filtering failed ({e}); using full dataset.")
            return dataset


# ---------------------------------------------------------------------------
# Time Horizon Curriculum
# ---------------------------------------------------------------------------


class TimeHorizonCurriculum(BaseCurriculum):
    """Curriculum that extends the prediction time horizon.

    Starts with short-term predictions (minutes) and progressively extends
    to longer horizons (hours, days, weeks). Short-term patterns are easier
    to learn and provide a foundation for longer-range predictions.

    Expected data_config keys
    -------------------------
    horizon_minutes : int
        Prediction horizon in minutes.
    lookback_minutes : int
        Lookback window in minutes.
    """

    def _build_stages(self) -> None:
        self._stages = [
            CurriculumStage(
                name="5min_horizon",
                difficulty=DifficultyLevel.BEGINNER,
                description="5-minute prediction horizon",
                data_config={
                    "horizon_minutes": 5,
                    "lookback_minutes": 60,
                },
                training_config_overrides={
                    "learning_rate": 1e-3,
                    "gradient_accumulation_steps": 1,
                },
                min_epochs=3,
                performance_threshold=0.2,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="15min_horizon",
                difficulty=DifficultyLevel.EASY,
                description="15-minute prediction horizon",
                data_config={
                    "horizon_minutes": 15,
                    "lookback_minutes": 120,
                },
                training_config_overrides={"learning_rate": 8e-4},
                min_epochs=4,
                performance_threshold=0.3,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="1h_horizon",
                difficulty=DifficultyLevel.MEDIUM,
                description="1-hour prediction horizon",
                data_config={
                    "horizon_minutes": 60,
                    "lookback_minutes": 480,
                },
                training_config_overrides={"learning_rate": 5e-4},
                min_epochs=5,
                performance_threshold=0.4,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="4h_horizon",
                difficulty=DifficultyLevel.HARD,
                description="4-hour prediction horizon",
                data_config={
                    "horizon_minutes": 240,
                    "lookback_minutes": 1440,
                },
                training_config_overrides={"learning_rate": 3e-4},
                min_epochs=5,
                performance_threshold=0.5,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="1d_horizon",
                difficulty=DifficultyLevel.EXPERT,
                description="1-day prediction horizon",
                data_config={
                    "horizon_minutes": 1440,
                    "lookback_minutes": 4320,
                },
                training_config_overrides={"learning_rate": 1e-4},
                min_epochs=8,
                performance_threshold=0.6,
                performance_metric="val_loss",
                performance_mode="min",
            ),
        ]

    def _build_progression_criteria(self) -> ProgressionCriteria:
        return ProgressionCriteria(
            min_epochs_in_stage=3,
            performance_threshold=None,
            performance_metric="val_loss",
            performance_mode="min",
            consecutive_epochs_met=2,
            max_epochs_in_stage=40,
        )

    def filter_data(self, dataset: Dataset, stage: CurriculumStage) -> Dataset:
        """Filter dataset for the appropriate time horizon.

        For time-series datasets, this adjusts the target offset to match
        the desired prediction horizon.
        """
        config = stage.data_config
        horizon = config.get("horizon_minutes", 5)
        lookback = config.get("lookback_minutes", 60)

        # If the dataset supports horizon configuration, apply it
        if hasattr(dataset, "set_horizon"):
            dataset.set_horizon(horizon_minutes=horizon, lookback_minutes=lookback)
            return dataset

        # For generic datasets, filter by sample count proportional to horizon
        total = len(dataset)
        # Shorter horizons use fewer samples initially
        ratio = min(1.0, (horizon / 1440.0) + 0.2)
        n_samples = max(100, int(total * ratio))

        # Use the most recent data for longer horizons
        start_idx = max(0, total - n_samples)
        indices = list(range(start_idx, total))
        return Subset(dataset, indices)


# ---------------------------------------------------------------------------
# Asset Count Curriculum
# ---------------------------------------------------------------------------


class AssetCountCurriculum(BaseCurriculum):
    """Curriculum that increases the number of trading assets.

    Starts with a single well-known asset (e.g., BTC), then adds more
    assets progressively. Single-asset models learn fundamental patterns
    before dealing with cross-asset correlations and portfolio effects.

    Expected data_config keys
    -------------------------
    assets : List[str]
        List of asset symbols for this stage.
    use_correlations : bool
        Whether cross-asset correlations are included.
    """

    def _build_stages(self) -> None:
        self._stages = [
            CurriculumStage(
                name="single_asset_btc",
                difficulty=DifficultyLevel.BEGINNER,
                description="Single asset: BTC only",
                data_config={
                    "assets": ["BTC"],
                    "use_correlations": False,
                },
                training_config_overrides={"learning_rate": 1e-3},
                min_epochs=5,
                performance_threshold=0.3,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="major_pairs",
                difficulty=DifficultyLevel.EASY,
                description="Major pairs: BTC, ETH",
                data_config={
                    "assets": ["BTC", "ETH"],
                    "use_correlations": False,
                },
                training_config_overrides={"learning_rate": 8e-4},
                min_epochs=5,
                performance_threshold=0.35,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="top_cap",
                difficulty=DifficultyLevel.MEDIUM,
                description="Top 5 assets with correlations",
                data_config={
                    "assets": ["BTC", "ETH", "BNB", "SOL", "ADA"],
                    "use_correlations": True,
                },
                training_config_overrides={"learning_rate": 5e-4},
                min_epochs=5,
                performance_threshold=0.4,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="mid_cap",
                difficulty=DifficultyLevel.HARD,
                description="Top 15 assets including mid-cap",
                data_config={
                    "assets": [
                        "BTC", "ETH", "BNB", "SOL", "ADA",
                        "AVAX", "DOT", "LINK", "MATIC", "UNI",
                        "AAVE", "ATOM", "FTM", "NEAR", "ALGO",
                    ],
                    "use_correlations": True,
                },
                training_config_overrides={"learning_rate": 3e-4},
                min_epochs=5,
                performance_threshold=0.5,
                performance_metric="val_loss",
                performance_mode="min",
            ),
            CurriculumStage(
                name="full_portfolio",
                difficulty=DifficultyLevel.EXPERT,
                description="Full portfolio: 30+ assets",
                data_config={
                    "assets": "all",
                    "use_correlations": True,
                },
                training_config_overrides={"learning_rate": 1e-4},
                min_epochs=10,
                performance_threshold=0.55,
                performance_metric="val_loss",
                performance_mode="min",
            ),
        ]

    def _build_progression_criteria(self) -> ProgressionCriteria:
        return ProgressionCriteria(
            min_epochs_in_stage=5,
            performance_threshold=None,
            performance_metric="val_loss",
            performance_mode="min",
            consecutive_epochs_met=3,
            max_epochs_in_stage=60,
        )

    def filter_data(self, dataset: Dataset, stage: CurriculumStage) -> Dataset:
        """Filter dataset to include only the assets for this stage."""
        config = stage.data_config
        assets = config.get("assets", ["BTC"])

        # If dataset has asset filtering capability
        if hasattr(dataset, "filter_assets"):
            return dataset.filter_assets(assets)

        # Generic: include all data (asset filtering must be handled by dataset)
        logger.info(
            f"Asset curriculum stage '{stage.name}': requesting assets {assets}. "
            f"Dataset does not support asset filtering; using all data."
        )
        return dataset


# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------


class CurriculumScheduler:
    """Orchestrates curriculum learning with multiple curriculum dimensions.

    Combines one or more curricula (e.g., market regime + time horizon) and
    manages their independent progression. Each curriculum advances based on
    its own criteria, providing a multi-dimensional difficulty ramp.

    Parameters
    ----------
    curricula : List[BaseCurriculum]
        List of curricula to coordinate.
    combination_strategy : str
        How to combine stage progressions:
        - 'independent': each curriculum advances independently
        - 'synchronized': all must be ready before any advances
        - 'sequential': advance one at a time in order
    """

    def __init__(
        self,
        curricula: List[BaseCurriculum],
        combination_strategy: str = "independent",
    ) -> None:
        self.curricula = curricula
        self.combination_strategy = combination_strategy
        self._epoch_metrics: List[Dict[str, float]] = []

    @property
    def overall_difficulty(self) -> float:
        """Compute a weighted average difficulty across all curricula.

        Returns
        -------
        float
            Difficulty score in [0, 1].
        """
        total = sum(c.difficulty for c in self.curricula)
        max_total = sum(DifficultyLevel.EXPERT for _ in self.curricula)
        return total / max(1, max_total)

    @property
    def all_final(self) -> bool:
        """Whether all curricula are at their final stage."""
        return all(c.is_final_stage for c in self.curricula)

    def on_epoch_end(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Notify all curricula of epoch end and check for advancement.

        Parameters
        ----------
        metrics : Dict[str, float]
            Epoch metrics.

        Returns
        -------
        Dict[str, bool]
            Mapping of curriculum name → whether it advanced.
        """
        self._epoch_metrics.append(metrics)
        advancements: Dict[str, bool] = {}

        if self.combination_strategy == "independent":
            for curriculum in self.curricula:
                name = curriculum.current_stage.name
                advanced = curriculum.on_epoch_end(metrics)
                advancements[name] = advanced

        elif self.combination_strategy == "synchronized":
            # Only advance if ALL curricula are ready
            ready = []
            for curriculum in self.curricula:
                criteria = curriculum._progression_criteria
                ready.append(
                    criteria.should_advance(
                        curriculum._epochs_in_stage, curriculum._recent_metrics
                    )
                )
            if all(ready) and not self.all_final:
                for curriculum in self.curricula:
                    name = curriculum.current_stage.name
                    curriculum._advance_stage()
                    advancements[name] = True
            else:
                for curriculum in self.curricula:
                    name = curriculum.current_stage.name
                    advancements[name] = False
                    curriculum._epochs_in_stage += 1
                    curriculum._recent_metrics.append(metrics)

        elif self.combination_strategy == "sequential":
            # Advance only the first non-final curriculum
            for curriculum in self.curricula:
                if not curriculum.is_final_stage:
                    name = curriculum.current_stage.name
                    advanced = curriculum.on_epoch_end(metrics)
                    advancements[name] = advanced
                    break

        return advancements

    def get_data_loaders(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        **loader_kwargs: Any,
    ) -> List[DataLoader]:
        """Get DataLoaders from all curricula for the current stage.

        Returns a list of DataLoaders — one per curriculum. The caller
        should combine them (e.g., concatenate batches) during training.
        """
        loaders = []
        for curriculum in self.curricula:
            loader = curriculum.get_data_loader(dataset, batch_size, **loader_kwargs)
            loaders.append(loader)
        return loaders

    def get_combined_training_overrides(self) -> Dict[str, Any]:
        """Merge training config overrides from all curricula.

        Later curricula take precedence when keys overlap.
        """
        overrides: Dict[str, Any] = {}
        for curriculum in self.curricula:
            overrides.update(curriculum.get_training_config_overrides())
        return overrides

    def get_progress(self) -> Dict[str, Any]:
        """Get comprehensive progress information from all curricula."""
        return {
            "overall_difficulty": self.overall_difficulty,
            "all_final": self.all_final,
            "strategy": self.combination_strategy,
            "curricula": {
                type(c).__name__: c.get_progress() for c in self.curricula
            },
        }

    def reset(self) -> None:
        """Reset all curricula to their initial stages."""
        for curriculum in self.curricula:
            curriculum.reset()
        self._epoch_metrics.clear()


# ---------------------------------------------------------------------------
# Adaptive Curriculum
# ---------------------------------------------------------------------------


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """Curriculum scheduler that adapts progression speed based on
    model performance trends.

    If the model is learning quickly, stages advance faster. If the model
    is struggling, it spends more time at each difficulty level.

    Parameters
    ----------
    curricula : List[BaseCurriculum]
        Curricula to coordinate.
    acceleration_factor : float
        Multiply min_epochs by this when model is improving rapidly.
    deceleration_factor : float
        Multiply min_epochs by this when model is stagnating.
    improvement_window : int
        Number of epochs to look back for trend detection.
    """

    def __init__(
        self,
        curricula: List[BaseCurriculum],
        combination_strategy: str = "independent",
        acceleration_factor: float = 0.5,
        deceleration_factor: float = 2.0,
        improvement_window: int = 5,
    ) -> None:
        super().__init__(curricula, combination_strategy)
        self.acceleration_factor = acceleration_factor
        self.deceleration_factor = deceleration_factor
        self.improvement_window = improvement_window

    def _is_improving(self, metric_name: str = "val_loss") -> bool:
        """Detect whether the model is improving based on recent metrics."""
        if len(self._epoch_metrics) < self.improvement_window:
            return False

        recent = self._epoch_metrics[-self.improvement_window:]
        values = [m.get(metric_name, float("inf")) for m in recent]

        # Simple trend: compare first half average to second half
        mid = len(values) // 2
        first_half_avg = sum(values[:mid]) / max(mid, 1)
        second_half_avg = sum(values[mid:]) / max(len(values) - mid, 1)

        return second_half_avg < first_half_avg

    def on_epoch_end(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Adaptively adjust curriculum progression before checking."""
        improving = self._is_improving()

        for curriculum in self.curricula:
            criteria = curriculum._progression_criteria
            base_min = criteria.min_epochs_in_stage

            if improving and curriculum._epochs_in_stage >= int(
                base_min * self.acceleration_factor
            ):
                # Model is learning well — relax the minimum
                criteria.min_epochs_in_stage = max(1, int(base_min * self.acceleration_factor))
            elif not improving:
                # Model is struggling — increase patience
                criteria.min_epochs_in_stage = int(base_min * self.deceleration_factor)

        return super().on_epoch_end(metrics)
