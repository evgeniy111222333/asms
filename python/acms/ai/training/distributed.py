"""
Distributed Training for ACMS
===============================

Distributed training support using PyTorch Distributed Data Parallel (DDP)
for scaling model training across multiple GPUs and nodes.

Features
--------
- DistributedTrainer wrapping PyTorch DDP
- Process group initialization and cleanup
- Gradient synchronization across workers
- Distributed data sampling for proper data partitioning
- Multi-node coordination
- Fault tolerance and recovery
- Resource monitoring across nodes
"""

from __future__ import annotations

import logging
import os
import socket
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    Attributes
    ----------
    backend : str
        Distributed backend ('nccl' for GPU, 'gloo' for CPU).
    init_method : str
        URL for process group initialization (e.g., 'tcp://host:port').
    world_size : int
        Total number of distributed processes.
    rank : int
        Rank of the current process.
    local_rank : int
        Local rank on the current node.
    master_addr : str
        Address of the master node.
    master_port : int
        Port of the master node.
    timeout_minutes : float
        Timeout for process group operations.
    find_unused_parameters : bool
        Whether to find unused parameters in DDP.
    bucket_cap_mb : int
        Bucket size in MB for gradient all-reduce.
    gradient_as_bucket_view : bool
        Whether to use gradient bucket view for memory savings.
    sync_batch_norm : bool
        Whether to use SyncBatchNorm.
    """

    backend: str = "nccl"
    init_method: Optional[str] = None
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: int = 29500
    timeout_minutes: float = 30.0
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    gradient_as_bucket_view: bool = False
    sync_batch_norm: bool = False

    @property
    def is_distributed(self) -> bool:
        """Whether distributed training is active."""
        return self.world_size > 1

    @property
    def is_master(self) -> bool:
        """Whether this is the master (rank 0) process."""
        return self.rank == 0

    @property
    def init_url(self) -> str:
        """Construct the init URL from address and port."""
        if self.init_method:
            return self.init_method
        return f"tcp://{self.master_addr}:{self.master_port}"


@dataclass
class NodeInfo:
    """Information about a compute node in the distributed cluster.

    Attributes
    ----------
    hostname : str
        Node hostname.
    rank : int
        Global rank of this node.
    local_rank : int
        Local rank within the node.
    world_size : int
        Total number of processes.
    n_gpus : int
        Number of GPUs available.
    gpu_names : List[str]
        Names of available GPUs.
    gpu_memory_total_mb : List[float]
        Total GPU memory in MB for each GPU.
    gpu_memory_used_mb : List[float]
        Used GPU memory in MB for each GPU.
    cpu_count : int
        Number of CPU cores.
    cpu_percent : float
        CPU utilization percentage.
    memory_total_gb : float
        Total system RAM in GB.
    memory_available_gb : float
        Available system RAM in GB.
    """

    hostname: str = ""
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    n_gpus: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_total_mb: List[float] = field(default_factory=list)
    gpu_memory_used_mb: List[float] = field(default_factory=list)
    cpu_count: int = 0
    cpu_percent: float = 0.0
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0

    @classmethod
    def collect(cls, rank: int = 0, local_rank: int = 0, world_size: int = 1) -> "NodeInfo":
        """Collect information about the current node.

        Parameters
        ----------
        rank : int
            Global rank.
        local_rank : int
            Local rank.
        world_size : int
            Total processes.

        Returns
        -------
        NodeInfo
            Current node information.
        """
        info = cls(
            hostname=socket.gethostname(),
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        )

        # GPU info
        if torch.cuda.is_available():
            info.n_gpus = torch.cuda.device_count()
            for i in range(info.n_gpus):
                info.gpu_names.append(torch.cuda.get_device_name(i))
                info.gpu_memory_total_mb.append(
                    torch.cuda.get_device_properties(i).total_mem / (1024 * 1024)
                )
                info.gpu_memory_used_mb.append(
                    torch.cuda.memory_allocated(i) / (1024 * 1024)
                )

        # CPU info
        try:
            import multiprocessing
            info.cpu_count = multiprocessing.cpu_count()
        except ImportError:
            info.cpu_count = 0

        # Memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            info.memory_total_gb = mem.total / (1024 ** 3)
            info.memory_available_gb = mem.available / (1024 ** 3)
            info.cpu_percent = psutil.cpu_percent(interval=0.1)
        except ImportError:
            logger.debug("psutil not available for system info")

        return info

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hostname": self.hostname,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "n_gpus": self.n_gpus,
            "gpu_names": self.gpu_names,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "cpu_count": self.cpu_count,
            "cpu_percent": self.cpu_percent,
            "memory_total_gb": self.memory_total_gb,
            "memory_available_gb": self.memory_available_gb,
        }


# ---------------------------------------------------------------------------
# Resource Monitor
# ---------------------------------------------------------------------------


class ResourceMonitor:
    """Monitors resource usage across distributed nodes.

    Periodically collects GPU memory, CPU usage, and training throughput
    metrics from all nodes and aggregates them.

    Parameters
    ----------
    config : DistributedConfig
        Distributed configuration.
    monitor_interval_s : float
        Seconds between monitoring snapshots.
    """

    def __init__(
        self,
        config: DistributedConfig,
        monitor_interval_s: float = 5.0,
    ) -> None:
        self.config = config
        self.monitor_interval_s = monitor_interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshots: List[Dict[str, Any]] = []
        self._gpu_memory_history: List[List[float]] = []

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"Resource monitor started (rank {self.config.rank})")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Resource monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            snapshot = self._collect_snapshot()
            self._snapshots.append(snapshot)
            time.sleep(self.monitor_interval_s)

    def _collect_snapshot(self) -> Dict[str, Any]:
        """Collect a resource usage snapshot."""
        snapshot: Dict[str, Any] = {
            "timestamp": time.time(),
            "rank": self.config.rank,
        }

        if torch.cuda.is_available():
            gpu_idx = self.config.local_rank
            if gpu_idx < torch.cuda.device_count():
                snapshot["gpu_memory_allocated_mb"] = (
                    torch.cuda.memory_allocated(gpu_idx) / (1024 * 1024)
                )
                snapshot["gpu_memory_reserved_mb"] = (
                    torch.cuda.memory_reserved(gpu_idx) / (1024 * 1024)
                )
                snapshot["gpu_max_memory_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(gpu_idx) / (1024 * 1024)
                )

        return snapshot

    def get_current_usage(self) -> Dict[str, float]:
        """Get the most recent resource usage."""
        if not self._snapshots:
            return self._collect_snapshot()
        return self._snapshots[-1]

    def aggregate_across_nodes(self) -> Dict[str, Any]:
        """Aggregate resource usage across all distributed nodes.

        Uses all-gather to collect metrics from all ranks.

        Returns
        -------
        Dict[str, Any]
            Aggregated resource metrics.
        """
        if not self.config.is_distributed or not dist.is_initialized():
            return self.get_current_usage()

        local_snapshot = self._collect_snapshot()

        # Gather all snapshots
        snapshots = [None] * self.config.world_size
        dist.all_gather_object(snapshots, local_snapshot)

        aggregated: Dict[str, Any] = {
            "n_nodes": self.config.world_size,
            "snapshots": snapshots,
        }

        # Compute aggregate GPU memory
        gpu_mems = [
            s.get("gpu_memory_allocated_mb", 0)
            for s in snapshots
            if s is not None
        ]
        if gpu_mems:
            aggregated["total_gpu_memory_mb"] = sum(gpu_mems)
            aggregated["avg_gpu_memory_mb"] = sum(gpu_mems) / len(gpu_mems)
            aggregated["max_gpu_memory_mb"] = max(gpu_mems)

        return aggregated

    @property
    def snapshot_count(self) -> int:
        """Number of monitoring snapshots collected."""
        return len(self._snapshots)


# ---------------------------------------------------------------------------
# Process Group Manager
# ---------------------------------------------------------------------------


class ProcessGroupManager:
    """Manages PyTorch distributed process group lifecycle.

    Handles initialization, barrier synchronization, and cleanup
    for the distributed process group.
    """

    @staticmethod
    def initialize(config: DistributedConfig) -> None:
        """Initialize the distributed process group.

        Parameters
        ----------
        config : DistributedConfig
            Distributed configuration.
        """
        if dist.is_initialized():
            logger.warning("Process group already initialized")
            return

        if not config.is_distributed:
            logger.info("Single-process mode; skipping process group init")
            return

        backend = config.backend
        if backend == "nccl" and not torch.cuda.is_available():
            logger.warning("NCCL backend requires CUDA; falling back to gloo")
            backend = "gloo"

        import datetime

        timeout = datetime.timedelta(minutes=config.timeout_minutes)

        dist.init_process_group(
            backend=backend,
            init_method=config.init_url,
            world_size=config.world_size,
            rank=config.rank,
            timeout=timeout,
        )

        # Set device for current process
        if torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)

        logger.info(
            f"Process group initialized: rank={config.rank}, "
            f"world_size={config.world_size}, backend={backend}"
        )

    @staticmethod
    def cleanup() -> None:
        """Clean up the distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group cleaned up")

    @staticmethod
    def barrier() -> None:
        """Synchronize all processes."""
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def is_initialized() -> bool:
        """Check if the process group is initialized."""
        return dist.is_initialized()


# ---------------------------------------------------------------------------
# Distributed Trainer
# ---------------------------------------------------------------------------


class DistributedTrainer:
    """Distributed trainer using PyTorch Distributed Data Parallel.

    Wraps a model with DDP, handles distributed data sampling, gradient
    synchronization, and provides fault tolerance.

    Parameters
    ----------
    model : nn.Module
        The model to train in distributed mode.
    config : DistributedConfig
        Distributed training configuration.
    loss_fn : Callable
        Loss function.
    optimizer_class : Type[torch.optim.Optimizer]
        Optimizer class.
    optimizer_kwargs : Dict[str, Any]
        Optimizer keyword arguments.
    lr_scheduler_fn : Optional[Callable]
        Factory function for LR scheduler (takes optimizer, returns scheduler).
    gradient_accumulation_steps : int
        Steps to accumulate gradients before sync.
    amp : bool
        Enable automatic mixed precision.
    max_grad_norm : float
        Maximum gradient norm for clipping.
    checkpoint_dir : Optional[str]
        Directory for saving distributed checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        loss_fn: Optional[Callable] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler_fn: Optional[Callable] = None,
        gradient_accumulation_steps: int = 1,
        amp: bool = True,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.loss_fn = loss_fn or nn.MSELoss()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.amp = amp and torch.cuda.is_available()
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir

        # Device setup
        self.device = torch.device(
            f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Initialize process group
        ProcessGroupManager.initialize(config)

        # Move model to device
        model = model.to(self.device)

        # SyncBatchNorm
        if config.sync_batch_norm and torch.cuda.is_available():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap with DDP
        if config.is_distributed:
            self.model = DDP(
                model,
                device_ids=[config.local_rank] if torch.cuda.is_available() else None,
                output_device=config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=config.find_unused_parameters,
                bucket_cap_mb=config.bucket_cap_mb,
                gradient_as_bucket_view=config.gradient_as_bucket_view,
            )
            logger.info(f"Model wrapped with DDP (rank {config.rank})")
        else:
            self.model = model

        # Optimizer (created after DDP wrapping)
        self.optimizer_kwargs = optimizer_kwargs or {"lr": 1e-3}
        self.optimizer = optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

        # LR scheduler
        self.scheduler = lr_scheduler_fn(self.optimizer) if lr_scheduler_fn else None

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.amp else None

        # Resource monitor
        self.resource_monitor = ResourceMonitor(config)

        # Node info
        self.node_info = NodeInfo.collect(
            rank=config.rank,
            local_rank=config.local_rank,
            world_size=config.world_size,
        )

        if config.is_master:
            logger.info(f"Distributed trainer initialized: {self.node_info.to_dict()}")

    def create_distributed_sampler(
        self, dataset: torch.utils.data.Dataset, shuffle: bool = True
    ) -> DistributedSampler:
        """Create a DistributedSampler for the dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to sample from.
        shuffle : bool
            Whether to shuffle each epoch.

        Returns
        -------
        DistributedSampler
            Distributed-aware sampler.
        """
        return DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle,
        )

    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """Create a DataLoader with distributed sampling.

        Parameters
        ----------
        dataset : Dataset
            The dataset.
        batch_size : int
            Per-process batch size.
        shuffle : bool
            Whether to shuffle (handled by sampler in distributed mode).
        num_workers : int
            DataLoader workers.
        pin_memory : bool
            Pin memory for faster GPU transfer.

        Returns
        -------
        DataLoader
            Distributed-aware DataLoader.
        """
        if self.config.is_distributed:
            sampler = self.create_distributed_sampler(dataset, shuffle=shuffle)
            # In distributed mode, shuffle must be False (sampler handles it)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory and self.device.type == "cuda",
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory and self.device.type == "cuda",
            )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train one epoch in distributed mode.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader (with DistributedSampler).
        epoch : int
            Current epoch number.

        Returns
        -------
        Dict[str, float]
            Training metrics.
        """
        self.model.train()

        # Set epoch for proper shuffling in DistributedSampler
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_samples = 0
        nan_count = 0

        for batch_idx, batch in enumerate(train_loader):
            x, y = self._unpack_batch(batch)
            batch_size = x.shape[0]

            with torch.cuda.amp.autocast(enabled=self.amp):
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss = loss / self.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                # Check for NaN
                has_nan = False
                for p in self.model.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        has_nan = True
                        break

                if has_nan:
                    nan_count += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * self.gradient_accumulation_steps * batch_size
            total_samples += batch_size

        # Average loss across all processes
        avg_loss = total_loss / max(total_samples, 1)
        if self.config.is_distributed:
            avg_loss = self._average_metric(avg_loss)

        # Step scheduler
        if self.scheduler:
            self.scheduler.step()

        return {
            "train_loss": avg_loss,
            "nan_count": float(nan_count),
        }

    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Run validation in distributed mode.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader.

        Returns
        -------
        Dict[str, float]
            Validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = self._unpack_batch(batch)
                with torch.cuda.amp.autocast(enabled=self.amp):
                    y_pred = self.model(x)
                    loss = self.loss_fn(y_pred, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]

        avg_loss = total_loss / max(total_samples, 1)
        if self.config.is_distributed:
            avg_loss = self._average_metric(avg_loss)

        return {"val_loss": avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> Dict[str, List[float]]:
        """Run the full distributed training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : Optional[DataLoader]
            Validation data loader.
        epochs : int
            Number of training epochs.

        Returns
        -------
        Dict[str, List[float]]
            Training history (losses per epoch).
        """
        self.resource_monitor.start()
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        if self.config.is_master:
            logger.info(
                f"Distributed training started: {epochs} epochs, "
                f"world_size={self.config.world_size}"
            )

        try:
            for epoch in range(epochs):
                train_metrics = self.train_epoch(train_loader, epoch)
                history["train_loss"].append(train_metrics["train_loss"])

                val_metrics = {}
                if val_loader:
                    val_metrics = self.validate(val_loader)
                    history["val_loss"].append(val_metrics["val_loss"])

                if self.config.is_master:
                    val_str = f", val_loss={val_metrics['val_loss']:.6f}" if val_metrics else ""
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"train_loss={train_metrics['train_loss']:.6f}"
                        f"{val_str}"
                    )

                # Periodic checkpoint
                if self.config.is_master and self.checkpoint_dir:
                    if (epoch + 1) % 5 == 0:
                        self._save_checkpoint(epoch)

        except Exception as e:
            logger.error(f"Distributed training failed at epoch {epoch}: {e}")
            raise
        finally:
            self.resource_monitor.stop()
            if self.config.is_master and self.checkpoint_dir:
                self._save_checkpoint(epochs - 1, filename="final_model.pt")

        if self.config.is_master:
            logger.info("Distributed training complete")

        return history

    def _average_metric(self, value: float) -> float:
        """Average a metric across all distributed processes.

        Parameters
        ----------
        value : float
            Local metric value.

        Returns
        -------
        float
            Averaged metric value.
        """
        if not self.config.is_distributed or not dist.is_initialized():
            return value

        tensor = torch.tensor([value], device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return (tensor / self.config.world_size).item()

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack a batch and move to the correct device."""
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x, y = batch["input"], batch["target"]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")
        return (
            x.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )

    def _save_checkpoint(self, epoch: int, filename: Optional[str] = None) -> None:
        """Save a distributed checkpoint.

        Only the master process saves. The model state dict is extracted
        from the DDP wrapper.
        """
        from pathlib import Path

        if not self.checkpoint_dir:
            return

        filename = filename or f"checkpoint_epoch_{epoch:04d}.pt"
        path = Path(self.checkpoint_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        # Extract state dict from DDP wrapper
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "world_size": self.config.world_size,
                "rank": self.config.rank,
            },
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> None:
        """Load a checkpoint for distributed training.

        Maps the checkpoint to the correct GPU for each process.

        Parameters
        ----------
        path : str
            Path to the checkpoint file.
        map_location : Optional[str]
            Device mapping string.
        """
        if map_location is None and torch.cuda.is_available():
            map_location = f"cuda:{self.config.local_rank}"

        checkpoint = torch.load(path, map_location=map_location)

        # Load into the underlying model (unwrap DDP)
        model = self.model.module if isinstance(self.model, DDP) else self.model
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded from {path} (rank {self.config.rank})")

    @staticmethod
    def cleanup() -> None:
        """Clean up the distributed process group."""
        ProcessGroupManager.cleanup()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def setup_distributed_from_env() -> DistributedConfig:
    """Create a DistributedConfig from environment variables.

    Reads the standard PyTorch distributed environment variables:
    - WORLD_SIZE
    - RANK
    - LOCAL_RANK
    - MASTER_ADDR
    - MASTER_PORT

    Returns
    -------
    DistributedConfig
        Configuration derived from environment.
    """
    return DistributedConfig(
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        master_addr=os.environ.get("MASTER_ADDR", "localhost"),
        master_port=int(os.environ.get("MASTER_PORT", "29500")),
        backend=os.environ.get("DIST_BACKEND", "nccl"),
    )


def launch_distributed(
    model_fn: Callable[..., nn.Module],
    train_dataset: torch.utils.data.Dataset,
    val_dataset: Optional[torch.utils.data.Dataset] = None,
    loss_fn: Optional[Callable] = None,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    **kwargs: Any,
) -> Dict[str, List[float]]:
    """Convenience function for launching distributed training.

    Reads distributed configuration from environment variables,
    creates the trainer, and runs training.

    Parameters
    ----------
    model_fn : Callable
        Factory function that returns a new model.
    train_dataset : Dataset
        Training dataset.
    val_dataset : Optional[Dataset]
        Validation dataset.
    loss_fn : Optional[Callable]
        Loss function.
    epochs : int
        Number of training epochs.
    batch_size : int
        Per-process batch size.
    learning_rate : float
        Learning rate.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    Dict[str, List[float]]
        Training history.
    """
    config = setup_distributed_from_env()

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)

    # Create model
    model = model_fn()

    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        config=config,
        loss_fn=loss_fn,
        optimizer_kwargs={"lr": learning_rate},
        **kwargs,
    )

    # Create data loaders
    train_loader = trainer.create_dataloader(train_dataset, batch_size=batch_size)
    val_loader = None
    if val_dataset:
        val_loader = trainer.create_dataloader(
            val_dataset, batch_size=batch_size * 2, shuffle=False
        )

    # Train
    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    # Cleanup
    trainer.cleanup()

    return history
