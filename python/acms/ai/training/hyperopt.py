"""
Hyperparameter Optimization for ACMS
======================================

Automated hyperparameter search using Optuna with support for multi-objective
optimization, pruning strategies, distributed search, and search space
visualization.

Features
--------
- HyperoptManager for managing optimization studies
- Predefined search spaces for all ACMS model types
- Multi-objective optimization (accuracy + latency + model size)
- Pruning strategies (median, successive halving, hyperband)
- Distributed optimization across multiple processes
- Warm starting from previous trial results
- Search space visualization (parameter importance, contour plots)
- Best hyperparameter tracking and persistence
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try importing Optuna — gracefully degrade if unavailable
try:
    import optuna
    from optuna.pruners import (
        HyperbandPruner,
        MedianPruner,
        SuccessiveHalvingPruner,
    )
    from optuna.samplers import TPESampler, NSGAIISampler
    from optuna.storages import InMemoryStorage, RDBStorage

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed. Hyperparameter optimization will not be available.")


# ---------------------------------------------------------------------------
# Enums and Data Classes
# ---------------------------------------------------------------------------


class MultiObjective(str, Enum):
    """Objectives for multi-objective optimization."""

    ACCURACY = "accuracy"
    LATENCY = "latency"
    MODEL_SIZE = "model_size"
    VAL_LOSS = "val_loss"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"


class PruningStrategy(str, Enum):
    """Pruning strategies for early stopping of bad trials."""

    MEDIAN = "median"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"
    NONE = "none"


@dataclass
class TrialResult:
    """Result of a single hyperparameter optimization trial.

    Attributes
    ----------
    trial_number : int
        Trial index.
    params : Dict[str, Any]
        Hyperparameter configuration tested.
    values : Dict[str, float]
        Objective values achieved.
    state : str
        Trial state ('COMPLETE', 'PRUNED', 'FAIL').
    duration_s : float
        Duration of the trial in seconds.
    best_epoch : int
        Epoch at which best metric was achieved.
    """

    trial_number: int
    params: Dict[str, Any] = field(default_factory=dict)
    values: Dict[str, float] = field(default_factory=dict)
    state: str = "COMPLETE"
    duration_s: float = 0.0
    best_epoch: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trial_number": self.trial_number,
            "params": self.params,
            "values": self.values,
            "state": self.state,
            "duration_s": self.duration_s,
            "best_epoch": self.best_epoch,
        }


# ---------------------------------------------------------------------------
# Search Space Definitions
# ---------------------------------------------------------------------------


class SearchSpace:
    """Defines the hyperparameter search space for ACMS models.

    Provides pre-configured search spaces for different model types and
    a fluent API for custom search space definition.

    Example
    -------
    >>> space = SearchSpace.for_model("transformer")
    >>> space.add_float("dropout", 0.1, 0.5)
    >>> space.add_int("num_layers", 2, 8)
    """

    def __init__(self) -> None:
        self._definitions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def for_model(cls, model_type: str) -> "SearchSpace":
        """Create a pre-configured search space for a model type.

        Parameters
        ----------
        model_type : str
            One of: 'transformer', 'lstm', 'cnn', 'mlp', 'ensemble'.

        Returns
        -------
        SearchSpace
            Configured search space.
        """
        space = cls()
        builders = {
            "transformer": space._transformer_space,
            "lstm": space._lstm_space,
            "cnn": space._cnn_space,
            "mlp": space._mlp_space,
            "ensemble": space._ensemble_space,
        }
        builder = builders.get(model_type)
        if builder:
            builder()
        else:
            logger.warning(f"No predefined space for model type '{model_type}'; using MLP default.")
            space._mlp_space()
        return space

    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None,
    ) -> "SearchSpace":
        """Add a float hyperparameter.

        Parameters
        ----------
        name : str
            Parameter name.
        low : float
            Minimum value.
        high : float
            Maximum value.
        log : bool
            Whether to sample on a log scale.
        step : Optional[float]
            Step size for discretization.
        """
        self._definitions[name] = {
            "type": "float",
            "low": low,
            "high": high,
            "log": log,
            "step": step,
        }
        return self

    def add_int(
        self,
        name: str,
        low: int,
        high: int,
        step: int = 1,
        log: bool = False,
    ) -> "SearchSpace":
        """Add an integer hyperparameter."""
        self._definitions[name] = {
            "type": "int",
            "low": low,
            "high": high,
            "step": step,
            "log": log,
        }
        return self

    def add_categorical(
        self,
        name: str,
        choices: List[Any],
    ) -> "SearchSpace":
        """Add a categorical hyperparameter."""
        self._definitions[name] = {
            "type": "categorical",
            "choices": choices,
        }
        return self

    def add_fixed(self, name: str, value: Any) -> "SearchSpace":
        """Add a fixed (non-searchable) parameter."""
        self._definitions[name] = {
            "type": "fixed",
            "value": value,
        }
        return self

    def sample(self, trial: Any) -> Dict[str, Any]:
        """Sample hyperparameters from the space using an Optuna trial.

        Parameters
        ----------
        trial : optuna.trial.Trial
            The Optuna trial to sample from.

        Returns
        -------
        Dict[str, Any]
            Sampled hyperparameter values.
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna is required for sampling")

        params = {}
        for name, defn in self._definitions.items():
            ptype = defn["type"]
            if ptype == "float":
                params[name] = trial.suggest_float(
                    name,
                    defn["low"],
                    defn["high"],
                    log=defn.get("log", False),
                    step=defn.get("step"),
                )
            elif ptype == "int":
                params[name] = trial.suggest_int(
                    name,
                    defn["low"],
                    defn["high"],
                    step=defn.get("step", 1),
                    log=defn.get("log", False),
                )
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, defn["choices"])
            elif ptype == "fixed":
                params[name] = defn["value"]
        return params

    def get_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Return the raw parameter definitions."""
        return deepcopy(self._definitions)

    @property
    def parameter_names(self) -> List[str]:
        """List of all parameter names."""
        return list(self._definitions.keys())

    @property
    def searchable_parameters(self) -> List[str]:
        """List of non-fixed parameter names."""
        return [
            k for k, v in self._definitions.items() if v["type"] != "fixed"
        ]

    # --- Pre-configured spaces ---

    def _transformer_space(self) -> None:
        """Search space for Transformer-based models."""
        self.add_int("d_model", 64, 512, step=64)
        self.add_int("nhead", 4, 16, step=2)
        self.add_int("num_encoder_layers", 1, 8)
        self.add_int("num_decoder_layers", 1, 8)
        self.add_float("dropout", 0.05, 0.5)
        self.add_float("learning_rate", 1e-5, 1e-2, log=True)
        self.add_float("weight_decay", 1e-6, 1e-2, log=True)
        self.add_int("batch_size", 16, 256, step=16)
        self.add_categorical("optimizer", ["adam", "adamw", "sgd"])
        self.add_float("warmup_ratio", 0.0, 0.3)
        self.add_int("gradient_accumulation_steps", 1, 8)

    def _lstm_space(self) -> None:
        """Search space for LSTM-based models."""
        self.add_int("hidden_size", 32, 512, step=32)
        self.add_int("num_layers", 1, 6)
        self.add_float("dropout", 0.0, 0.5)
        self.add_float("learning_rate", 1e-5, 1e-2, log=True)
        self.add_float("weight_decay", 1e-6, 1e-2, log=True)
        self.add_int("batch_size", 16, 256, step=16)
        self.add_categorical("optimizer", ["adam", "adamw", "rmsprop"])
        self.add_bool("bidirectional")
        self.add_int("gradient_accumulation_steps", 1, 4)

    def _cnn_space(self) -> None:
        """Search space for CNN-based models."""
        self.add_int("num_channels", 16, 128, step=16)
        self.add_int("num_layers", 2, 8)
        self.add_int("kernel_size", 3, 7, step=2)
        self.add_float("dropout", 0.05, 0.5)
        self.add_float("learning_rate", 1e-5, 1e-2, log=True)
        self.add_float("weight_decay", 1e-6, 1e-2, log=True)
        self.add_int("batch_size", 16, 256, step=16)
        self.add_categorical("optimizer", ["adam", "adamw", "sgd"])

    def _mlp_space(self) -> None:
        """Search space for MLP models."""
        self.add_int("hidden_dim", 64, 1024, step=64)
        self.add_int("num_layers", 1, 8)
        self.add_float("dropout", 0.0, 0.5)
        self.add_float("learning_rate", 1e-5, 1e-2, log=True)
        self.add_float("weight_decay", 1e-6, 1e-2, log=True)
        self.add_int("batch_size", 16, 512, step=16)
        self.add_categorical("optimizer", ["adam", "adamw", "sgd"])
        self.add_categorical("activation", ["relu", "gelu", "elu", "swish"])

    def _ensemble_space(self) -> None:
        """Search space for ensemble models."""
        self.add_int("num_models", 3, 10)
        self.add_float("learning_rate", 1e-5, 1e-2, log=True)
        self.add_float("dropout", 0.0, 0.5)
        self.add_int("batch_size", 16, 256, step=16)
        self.add_categorical("aggregation", ["mean", "median", "weighted"])
        self.add_float("diversity_weight", 0.0, 1.0)

    # Helper for bool via categorical
    def add_bool(self, name: str) -> "SearchSpace":
        """Add a boolean hyperparameter."""
        return self.add_categorical(name, [True, False])


# ---------------------------------------------------------------------------
# Hyperopt Manager
# ---------------------------------------------------------------------------


class HyperoptManager:
    """Manages hyperparameter optimization studies using Optuna.

    Provides a high-level interface for creating studies, running trials,
    and analyzing results. Supports single-objective and multi-objective
    optimization, various pruning strategies, and warm starting.

    Parameters
    ----------
    study_name : str
        Name of the optimization study.
    storage : Optional[str]
        Database URL for persistent storage (e.g., 'sqlite:///optuna.db').
        None uses in-memory storage.
    objectives : List[MultiObjective]
        Objectives to optimize.
    directions : List[str]
        'minimize' or 'maximize' for each objective.
    pruning_strategy : PruningStrategy
        Pruning strategy for early stopping of unpromising trials.
    n_trials : int
        Maximum number of trials.
    timeout : Optional[float]
        Maximum optimization time in seconds.
    n_jobs : int
        Number of parallel jobs.
    seed : int
        Random seed for reproducibility.
    warm_start_trials : Optional[List[TrialResult]]
        Previous trial results to warm start from.
    """

    def __init__(
        self,
        study_name: str = "acms_hyperopt",
        storage: Optional[str] = None,
        objectives: Optional[List[MultiObjective]] = None,
        directions: Optional[List[str]] = None,
        pruning_strategy: PruningStrategy = PruningStrategy.HYPERBAND,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        seed: int = 42,
        warm_start_trials: Optional[List[TrialResult]] = None,
    ) -> None:
        if not OPTUNA_AVAILABLE:
            raise RuntimeError(
                "Optuna is required for HyperoptManager. "
                "Install it with: pip install optuna"
            )

        self.study_name = study_name
        self.objectives = objectives or [MultiObjective.VAL_LOSS]
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.seed = seed
        self.pruning_strategy = pruning_strategy

        # Default directions: minimize loss, maximize accuracy/sharpe
        if directions is None:
            self.directions = []
            for obj in self.objectives:
                if obj in (MultiObjective.VAL_LOSS, MultiObjective.LATENCY,
                           MultiObjective.MODEL_SIZE, MultiObjective.MAX_DRAWDOWN):
                    self.directions.append("minimize")
                else:
                    self.directions.append("maximize")
        else:
            self.directions = directions

        # Create storage
        self._storage = storage
        if storage:
            self._optuna_storage = RDBStorage(storage)
        else:
            self._optuna_storage = InMemoryStorage()

        # Create pruner
        self._pruner = self._create_pruner(pruning_strategy)

        # Create sampler
        is_multi = len(self.objectives) > 1
        if is_multi:
            self._sampler = NSGAIISampler(seed=seed)
        else:
            self._sampler = TPESampler(seed=seed)

        # Create study
        self._study = optuna.create_study(
            study_name=study_name,
            storage=self._optuna_storage,
            sampler=self._sampler,
            pruner=self._pruner,
            directions=self.directions,
        )

        # Results tracking
        self._trial_results: List[TrialResult] = []
        self._best_params: Optional[Dict[str, Any]] = None
        self._best_values: Optional[Dict[str, float]] = None

        # Warm start
        if warm_start_trials:
            self._warm_start(warm_start_trials)

    def _create_pruner(self, strategy: PruningStrategy) -> Any:
        """Create the Optuna pruner based on strategy."""
        if strategy == PruningStrategy.MEDIAN:
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1,
            )
        elif strategy == PruningStrategy.SUCCESSIVE_HALVING:
            return SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=4,
                min_early_stopping_rate=0,
            )
        elif strategy == PruningStrategy.HYPERBAND:
            return HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3,
            )
        elif strategy == PruningStrategy.NONE:
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

    def _warm_start(self, trials: List[TrialResult]) -> None:
        """Add previous trial results to warm start the study.

        Parameters
        ----------
        trials : List[TrialResult]
            Previous trial results to inject.
        """
        for trial_result in trials:
            self._study.add_trial(
                optuna.trial.create_trial(
                    params=trial_result.params,
                    distributions={
                        k: self._make_distribution(k, v)
                        for k, v in trial_result.params.items()
                    },
                    values=list(trial_result.values.values()),
                    state=optuna.trial.TrialState.COMPLETE,
                )
            )
        logger.info(f"Warm started with {len(trials)} previous trials")

    @staticmethod
    def _make_distribution(name: str, value: Any) -> Any:
        """Create an Optuna distribution that's compatible with a value."""
        if isinstance(value, bool):
            return optuna.distributions.CategoricalDistribution([True, False])
        elif isinstance(value, int):
            return optuna.distributions.IntDistribution(value, value)
        elif isinstance(value, float):
            return optuna.distributions.FloatDistribution(value, value)
        else:
            return optuna.distributions.CategoricalDistribution([value])

    def optimize(
        self,
        objective_fn: Callable,
        search_space: SearchSpace,
        n_trials: Optional[int] = None,
        timeout: Optional[float] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> List[TrialResult]:
        """Run the hyperparameter optimization.

        Parameters
        ----------
        objective_fn : Callable
            A function that takes (trial, params) and returns a dict of
            objective values, or a float for single-objective optimization.
            For step-wise evaluation, use trial.report() for pruning.
        search_space : SearchSpace
            The search space to explore.
        n_trials : Optional[int]
            Override number of trials.
        timeout : Optional[float]
            Override timeout in seconds.
        callbacks : Optional[List[Callable]]
            Optuna study callbacks.

        Returns
        -------
        List[TrialResult]
            Results of all completed trials.
        """
        n_trials = n_trials or self.n_trials
        timeout = timeout or self.timeout

        def _optuna_objective(trial: optuna.trial.Trial) -> Any:
            """Wrapper that samples params and calls the user objective."""
            start_time = time.time()
            params = search_space.sample(trial)

            try:
                result = objective_fn(trial, params)

                # Handle both single-objective and multi-objective returns
                if isinstance(result, dict):
                    values = list(result.values())
                elif isinstance(result, (int, float)):
                    values = [float(result)]
                else:
                    values = [float(result)]

                duration = time.time() - start_time

                trial_result = TrialResult(
                    trial_number=trial.number,
                    params=params,
                    values=dict(zip([o.value for o in self.objectives], values)),
                    state="COMPLETE",
                    duration_s=duration,
                )
                self._trial_results.append(trial_result)
                return values[0] if len(values) == 1 else values

            except optuna.exceptions.TrialPruned:
                trial_result = TrialResult(
                    trial_number=trial.number,
                    params=params,
                    state="PRUNED",
                    duration_s=time.time() - start_time,
                )
                self._trial_results.append(trial_result)
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                trial_result = TrialResult(
                    trial_number=trial.number,
                    params=params,
                    state="FAIL",
                    duration_s=time.time() - start_time,
                )
                self._trial_results.append(trial_result)
                return float("inf") if len(self.directions) == 1 else [float("inf")]

        logger.info(
            f"Starting optimization: {n_trials} trials, "
            f"{len(self.objectives)} objectives, "
            f"pruner={self.pruning_strategy.value}"
        )

        self._study.optimize(
            _optuna_objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
        )

        self._update_best()
        return self._trial_results

    def _update_best(self) -> None:
        """Update best params and values from the study."""
        try:
            if len(self.objectives) == 1:
                best_trial = self._study.best_trial
                self._best_params = best_trial.params
                self._best_values = {
                    self.objectives[0].value: best_trial.value
                }
            else:
                # Multi-objective: use the first Pareto-optimal trial
                pareto = self._study.best_trials
                if pareto:
                    best = pareto[0]
                    self._best_params = best.params
                    self._best_values = dict(zip(
                        [o.value for o in self.objectives], best.values
                    ))
        except ValueError:
            # No completed trials yet
            logger.debug("No completed trials yet for best params")

    @property
    def best_params(self) -> Optional[Dict[str, Any]]:
        """Best hyperparameters found so far."""
        return self._best_params

    @property
    def best_values(self) -> Optional[Dict[str, float]]:
        """Best objective values found so far."""
        return self._best_values

    @property
    def trial_results(self) -> List[TrialResult]:
        """All trial results."""
        return self._trial_results

    def get_param_importances(self) -> Dict[str, float]:
        """Compute parameter importance using fANOVA.

        Returns
        -------
        Dict[str, float]
            Parameter name → importance score.
        """
        if not OPTUNA_AVAILABLE:
            return {}

        try:
            from optuna.importance import get_param_importances

            if len(self.objectives) > 1:
                logger.warning("Param importance is for single-objective; using first objective.")

            importances = get_param_importances(self._study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not compute param importances: {e}")
            return {}

    def visualize_parallel_coordinate(self) -> Any:
        """Generate a parallel coordinate plot of the study.

        Returns
        -------
        optuna.visualization plot object
        """
        if not OPTUNA_AVAILABLE:
            return None
        try:
            import optuna.visualization as vis
            return vis.plot_parallel_coordinate(self._study)
        except Exception as e:
            logger.warning(f"Could not generate parallel coordinate plot: {e}")
            return None

    def visualize_param_importances(self) -> Any:
        """Generate a parameter importance plot.

        Returns
        -------
        optuna.visualization plot object
        """
        if not OPTUNA_AVAILABLE:
            return None
        try:
            import optuna.visualization as vis
            return vis.plot_param_importances(self._study)
        except Exception as e:
            logger.warning(f"Could not generate importance plot: {e}")
            return None

    def visualize_optimization_history(self) -> Any:
        """Generate an optimization history plot.

        Returns
        -------
        optuna.visualization plot object
        """
        if not OPTUNA_AVAILABLE:
            return None
        try:
            import optuna.visualization as vis
            return vis.plot_optimization_history(self._study)
        except Exception as e:
            logger.warning(f"Could not generate history plot: {e}")
            return None

    def visualize_pareto_front(self) -> Any:
        """Generate a Pareto front plot (for multi-objective studies).

        Returns
        -------
        optuna.visualization plot object
        """
        if not OPTUNA_AVAILABLE or len(self.objectives) < 2:
            return None
        try:
            import optuna.visualization as vis
            return vis.plot_pareto_front(self._study)
        except Exception as e:
            logger.warning(f"Could not generate Pareto front plot: {e}")
            return None

    def save_results(self, path: str) -> None:
        """Save optimization results to a JSON file.

        Parameters
        ----------
        path : str
            Output file path.
        """
        data = {
            "study_name": self.study_name,
            "objectives": [o.value for o in self.objectives],
            "directions": self.directions,
            "best_params": self._best_params,
            "best_values": self._best_values,
            "n_trials": len(self._trial_results),
            "trials": [t.to_dict() for t in self._trial_results],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")

    @classmethod
    def load_results(cls, path: str) -> List[TrialResult]:
        """Load optimization results from a JSON file.

        Parameters
        ----------
        path : str
            Input file path.

        Returns
        -------
        List[TrialResult]
            Loaded trial results.
        """
        with open(path) as f:
            data = json.load(f)

        results = []
        for t in data.get("trials", []):
            results.append(TrialResult(
                trial_number=t["trial_number"],
                params=t.get("params", {}),
                values=t.get("values", {}),
                state=t.get("state", "UNKNOWN"),
                duration_s=t.get("duration_s", 0.0),
                best_epoch=t.get("best_epoch", 0),
            ))
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization study."""
        completed = [t for t in self._trial_results if t.state == "COMPLETE"]
        pruned = [t for t in self._trial_results if t.state == "PRUNED"]
        failed = [t for t in self._trial_results if t.state == "FAIL"]

        return {
            "study_name": self.study_name,
            "n_trials": len(self._trial_results),
            "n_completed": len(completed),
            "n_pruned": len(pruned),
            "n_failed": len(failed),
            "best_params": self._best_params,
            "best_values": self._best_values,
            "pruning_strategy": self.pruning_strategy.value,
            "objectives": [o.value for o in self.objectives],
            "prune_rate": len(pruned) / max(1, len(self._trial_results)),
        }
