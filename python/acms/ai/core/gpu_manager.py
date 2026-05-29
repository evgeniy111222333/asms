"""GPU device management for the ACMS AI module.

Provides a singleton GPUManager that handles:
- Device selection and auto-detection
- CUDA context management
- Memory tracking and OOM prevention
- Mixed precision (AMP) context managers
- GPU memory pool and caching
- Multi-GPU orchestration
- Automatic CPU fallback when no GPU is available
- Device placement utilities for PyTorch tensors
- Gradient scaling for mixed precision training
- Memory usage monitoring and reporting

All operations gracefully fall back to CPU when CUDA is not available
or when PyTorch is not installed, with appropriate logging.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

from .config import GPUConfig

logger = logging.getLogger(__name__)

# Lazy torch import flag
_TORCH_AVAILABLE: Optional[bool] = None


def _is_torch_available() -> bool:
    """Check if PyTorch is available (cached result)."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
            logger.warning(
                "PyTorch not installed. GPU features unavailable. "
                "Install with: pip install torch"
            )
    return _TORCH_AVAILABLE


def _is_cuda_available() -> bool:
    """Check if CUDA is available."""
    if not _is_torch_available():
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ============================================================================
# GPU Memory Info
# ============================================================================


@dataclass
class GPUMemoryInfo:
    """Snapshot of GPU memory usage.

    Attributes:
        device_id: GPU device index.
        total_mb: Total GPU memory in MB.
        allocated_mb: Currently allocated memory in MB.
        reserved_mb: Memory reserved by the caching allocator in MB.
        free_mb: Free memory (total - reserved) in MB.
        utilization: Memory utilization ratio [0, 1].
    """

    device_id: int = 0
    total_mb: float = 0.0
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0
    utilization: float = 0.0

    def __str__(self) -> str:
        return (
            f"GPU {self.device_id}: "
            f"{self.allocated_mb:.0f}/{self.total_mb:.0f} MB "
            f"({self.utilization:.1%} utilized, "
            f"{self.free_mb:.0f} MB free)"
        )


@dataclass
class GPUDeviceInfo:
    """Information about a GPU device.

    Attributes:
        device_id: GPU device index.
        name: Device name (e.g., 'NVIDIA A100-SXM4-80GB').
        compute_capability: Compute capability tuple (major, minor).
        total_memory_mb: Total device memory in MB.
        multiprocessor_count: Number of multiprocessors.
        is_available: Whether the device is currently available.
    """

    device_id: int = 0
    name: str = "CPU"
    compute_capability: Tuple[int, int] = (0, 0)
    total_memory_mb: float = 0.0
    multiprocessor_count: int = 0
    is_available: bool = False

    def __str__(self) -> str:
        return (
            f"GPU {self.device_id}: {self.name} "
            f"({self.total_memory_mb:.0f} MB, "
            f"SM {self.compute_capability[0]}.{self.compute_capability[1]})"
        )


# ============================================================================
# GPU Manager Singleton
# ============================================================================


class GPUManager:
    """Singleton GPU device manager for the ACMS AI subsystem.

    Provides centralized GPU resource management including device
    selection, memory tracking, AMP context, and multi-GPU orchestration.

    Usage:
        manager = GPUManager()
        manager.initialize(GPUConfig(device="auto"))

        # Move tensor to device
        tensor = manager.to_device(numpy_array)

        # AMP context for training
        with manager.amp_context():
            output = model(input)
            loss = criterion(output, target)
            manager.scale_and_step(loss, optimizer)

        # Memory info
        info = manager.get_memory_info()
        print(info)
    """

    _instance: Optional["GPUManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "GPUManager":
        """Create or return the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize the GPU manager (only runs once due to singleton)."""
        if self._initialized:
            return

        self._config: GPUConfig = GPUConfig()
        self._device: str = "cpu"
        self._device_ids: List[int] = []
        self._scaler: Any = None  # torch.cuda.amp.GradScaler
        self._amp_enabled: bool = False
        self._devices_info: List[GPUDeviceInfo] = []
        self._memory_history: List[Dict[str, GPUMemoryInfo]] = []
        self._oom_count: int = 0
        self._initialized_flag = True
        self._init_lock = threading.Lock()

    # ----------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------

    def initialize(self, config: Optional[GPUConfig] = None) -> None:
        """Initialize the GPU manager with configuration.

        Args:
            config: GPU configuration. Uses defaults if not provided.
        """
        with self._init_lock:
            if self._initialized and config is not None:
                logger.info("Re-initializing GPUManager with new config")

            self._config = config or GPUConfig()
            self._detect_devices()
            self._device = self._config.resolve_device()
            self._device_ids = self._config.get_device_ids()
            self._setup_amp()
            self._setup_cudnn()
            self._initialized = True

            logger.info(
                "GPUManager initialized: device=%s, device_ids=%s, "
                "amp=%s, n_gpus=%d",
                self._device,
                self._device_ids,
                self._amp_enabled,
                len(self._device_ids),
            )

    def _detect_devices(self) -> None:
        """Detect and catalog available GPU devices."""
        self._devices_info = []

        if not _is_cuda_available():
            self._devices_info.append(GPUDeviceInfo(
                device_id=-1, name="CPU", is_available=True
            ))
            return

        try:
            import torch
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self._devices_info.append(GPUDeviceInfo(
                    device_id=i,
                    name=props.name,
                    compute_capability=(props.major, props.minor),
                    total_memory_mb=props.total_memory / (1024 * 1024),
                    multiprocessor_count=props.multi_processor_count,
                    is_available=True,
                ))
                logger.debug("Detected GPU %d: %s", i, props.name)
        except Exception as e:
            logger.warning("Error detecting GPUs: %s", e)
            self._devices_info.append(GPUDeviceInfo(
                device_id=-1, name="CPU", is_available=True
            ))

    def _setup_amp(self) -> None:
        """Set up Automatic Mixed Precision if configured and available."""
        self._amp_enabled = False
        self._scaler = None

        if not self._config.enable_mixed_precision:
            return

        if self._device == "cpu":
            logger.info("AMP disabled: running on CPU")
            return

        if not _is_torch_available():
            return

        try:
            import torch
            if self._config.amp_dtype == "bfloat16":
                # BF16 doesn't need a GradScaler
                self._amp_enabled = True
                logger.info("AMP enabled with bfloat16 (no GradScaler needed)")
            else:
                self._scaler = torch.cuda.amp.GradScaler(
                    init_scale=2**16,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                    enabled=True,
                )
                self._amp_enabled = True
                logger.info("AMP enabled with float16 + GradScaler")
        except Exception as e:
            logger.warning("Failed to set up AMP: %s", e)
            self._amp_enabled = False

    def _setup_cudnn(self) -> None:
        """Configure cuDNN settings."""
        if not _is_torch_available() or self._device == "cpu":
            return

        try:
            import torch
            if self._config.enable_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.debug("cuDNN benchmark enabled")
            if self._config.enable_cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                logger.debug("cuDNN deterministic mode enabled")
        except Exception as e:
            logger.warning("Failed to configure cuDNN: %s", e)

    # ----------------------------------------------------------------
    # Device Properties
    # ----------------------------------------------------------------

    @property
    def device(self) -> str:
        """Current compute device string."""
        return self._device

    @property
    def is_gpu(self) -> bool:
        """Whether currently running on GPU."""
        return self._device != "cpu"

    @property
    def is_multi_gpu(self) -> bool:
        """Whether multiple GPUs are available."""
        return len(self._device_ids) > 1

    @property
    def num_gpus(self) -> int:
        """Number of available GPUs."""
        return len(self._device_ids)

    @property
    def amp_enabled(self) -> bool:
        """Whether AMP is currently enabled."""
        return self._amp_enabled

    @property
    def scaler(self) -> Any:
        """The AMP GradScaler instance (or None)."""
        return self._scaler

    @property
    def devices_info(self) -> List[GPUDeviceInfo]:
        """Information about all detected devices."""
        return list(self._devices_info)

    def get_primary_device_id(self) -> int:
        """Get the primary GPU device ID.

        Returns:
            Primary device ID, or -1 if running on CPU.
        """
        return self._device_ids[0] if self._device_ids else -1

    # ----------------------------------------------------------------
    # Memory Management
    # ----------------------------------------------------------------

    def get_memory_info(self, device_id: Optional[int] = None) -> GPUMemoryInfo:
        """Get current GPU memory usage information.

        Args:
            device_id: GPU device ID. Uses primary device if None.

        Returns:
            GPUMemoryInfo snapshot.
        """
        if self._device == "cpu":
            return GPUMemoryInfo(device_id=-1, name="CPU")

        if not _is_torch_available():
            return GPUMemoryInfo(device_id=-1)

        try:
            import torch
            dev_id = device_id if device_id is not None else self.get_primary_device_id()
            if dev_id < 0:
                return GPUMemoryInfo(device_id=-1)

            total = torch.cuda.get_device_properties(dev_id).total_memory / (1024**2)
            allocated = torch.cuda.memory_allocated(dev_id) / (1024**2)
            reserved = torch.cuda.memory_reserved(dev_id) / (1024**2)
            free = total - reserved
            utilization = allocated / total if total > 0 else 0.0

            return GPUMemoryInfo(
                device_id=dev_id,
                total_mb=total,
                allocated_mb=allocated,
                reserved_mb=reserved,
                free_mb=free,
                utilization=utilization,
            )
        except Exception as e:
            logger.warning("Error getting memory info: %s", e)
            return GPUMemoryInfo(device_id=device_id or -1)

    def get_all_memory_info(self) -> Dict[int, GPUMemoryInfo]:
        """Get memory info for all available GPUs.

        Returns:
            Dictionary mapping device_id -> GPUMemoryInfo.
        """
        result = {}
        for dev_id in self._device_ids:
            result[dev_id] = self.get_memory_info(dev_id)
        return result

    def memory_summary(self) -> str:
        """Generate a human-readable memory summary.

        Returns:
            Formatted string with memory usage for all devices.
        """
        lines = ["=" * 60, "GPU Memory Summary", "=" * 60]
        if self._device == "cpu":
            lines.append("Running on CPU - no GPU memory to report")
        else:
            for dev_id in self._device_ids:
                info = self.get_memory_info(dev_id)
                lines.append(f"  {info}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def empty_cache(self) -> None:
        """Release all unoccupied cached GPU memory."""
        if self._device == "cpu" or not _is_torch_available():
            return

        try:
            import torch
            torch.cuda.empty_cache()
            logger.debug("GPU cache emptied")
        except Exception as e:
            logger.warning("Error emptying GPU cache: %s", e)

    def reset_peak_memory(self, device_id: Optional[int] = None) -> None:
        """Reset peak memory tracking for a device.

        Args:
            device_id: GPU device ID. Uses primary device if None.
        """
        if self._device == "cpu" or not _is_torch_available():
            return

        try:
            import torch
            dev_id = device_id if device_id is not None else self.get_primary_device_id()
            if dev_id >= 0:
                torch.cuda.reset_peak_memory_stats(dev_id)
                logger.debug("Peak memory stats reset for device %d", dev_id)
        except Exception as e:
            logger.warning("Error resetting peak memory: %s", e)

    def check_memory_available(self, required_mb: float, device_id: Optional[int] = None) -> bool:
        """Check if sufficient GPU memory is available.

        Args:
            required_mb: Required memory in MB.
            device_id: GPU device ID. Uses primary device if None.

        Returns:
            True if sufficient memory is available.
        """
        info = self.get_memory_info(device_id)
        available = info.free_mb * self._config.memory_fraction
        if available >= required_mb:
            return True
        logger.warning(
            "Insufficient GPU memory: need %.0f MB, have %.0f MB available",
            required_mb, available,
        )
        return False

    def record_memory_snapshot(self) -> None:
        """Record a memory usage snapshot for monitoring."""
        snapshot = self.get_all_memory_info()
        if snapshot:
            self._memory_history.append(snapshot)
            # Keep only last 1000 snapshots
            if len(self._memory_history) > 1000:
                self._memory_history = self._memory_history[-1000:]

    # ----------------------------------------------------------------
    # Device Placement
    # ----------------------------------------------------------------

    def to_device(
        self,
        data: Union[np.ndarray, Any, List, Dict, Tuple],
        device: Optional[str] = None,
    ) -> Any:
        """Move data to the specified device.

        Handles numpy arrays, PyTorch tensors, and nested structures
        of tensors (lists, dicts, tuples).

        Args:
            data: Data to move. Can be numpy array, torch tensor,
                or nested structure of tensors.
            device: Target device string. Uses current device if None.

        Returns:
            Data on the target device.
        """
        target_device = device or self._device

        if isinstance(data, np.ndarray):
            return self._numpy_to_device(data, target_device)

        if _is_torch_available():
            import torch
            if isinstance(data, torch.Tensor):
                return data.to(target_device)
            if isinstance(data, dict):
                return {k: self.to_device(v, target_device) for k, v in data.items()}
            if isinstance(data, (list, tuple)):
                moved = [self.to_device(item, target_device) for item in data]
                return type(data)(moved) if isinstance(data, tuple) else moved

        # Fallback: return as-is for CPU
        return data

    def _numpy_to_device(self, array: np.ndarray, device: str) -> Any:
        """Convert a numpy array to a device tensor.

        Args:
            array: Numpy array to convert.
            device: Target device string.

        Returns:
            PyTorch tensor on the target device, or the original numpy
            array if PyTorch is not available.
        """
        if not _is_torch_available():
            return array

        try:
            import torch
            tensor = torch.from_numpy(array)
            return tensor.to(device)
        except Exception as e:
            logger.warning("Failed to move numpy array to %s: %s", device, e)
            return array

    def to_numpy(self, data: Any) -> Union[np.ndarray, Any]:
        """Move data from device to numpy array.

        Args:
            data: PyTorch tensor or numpy array.

        Returns:
            Numpy array on CPU.
        """
        if isinstance(data, np.ndarray):
            return data

        if _is_torch_available():
            import torch
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()

        return data

    # ----------------------------------------------------------------
    # AMP Context Managers
    # ----------------------------------------------------------------

    @contextmanager
    def amp_context(self) -> Generator[None, None, None]:
        """Context manager for Automatic Mixed Precision inference/training.

        Wraps the forward pass in torch.cuda.amp.autocast when AMP
        is enabled. No-op when AMP is disabled or on CPU.

        Yields:
            None - this is a context manager.

        Example:
            with manager.amp_context():
                output = model(input)
        """
        if not self._amp_enabled or self._device == "cpu":
            yield
            return

        if not _is_torch_available():
            yield
            return

        try:
            import torch
            dtype = torch.bfloat16 if self._config.amp_dtype == "bfloat16" else torch.float16
            with torch.cuda.amp.autocast(dtype=dtype):
                yield
        except RuntimeError as e:
            if "autocast" in str(e):
                logger.warning("AMP autocast failed, falling back to FP32: %s", e)
                yield
            else:
                raise

    @contextmanager
    def memory_aware_context(
        self,
        required_mb: float = 0.0,
        device_id: Optional[int] = None,
    ) -> Generator[None, None, None]:
        """Context manager that monitors memory and handles OOM.

        Checks available memory before entering, and handles OOM
        gracefully by clearing cache and retrying once.

        Args:
            required_mb: Estimated memory required for the operation.
            device_id: GPU device ID to monitor.

        Yields:
            None - this is a context manager.
        """
        # Pre-check
        if required_mb > 0 and not self.check_memory_available(required_mb, device_id):
            self.empty_cache()
            if not self.check_memory_available(required_mb, device_id):
                raise MemoryError(
                    f"Insufficient GPU memory: need {required_mb:.0f} MB"
                )

        try:
            yield
        except MemoryError:
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self._oom_count += 1
                logger.error(
                    "GPU OOM (total OOM events: %d). Clearing cache...",
                    self._oom_count,
                )
                self.empty_cache()
                raise MemoryError(
                    f"GPU out of memory. OOM count: {self._oom_count}"
                ) from e
            raise

    # ----------------------------------------------------------------
    # Gradient Scaling (for AMP training)
    # ----------------------------------------------------------------

    def scale_loss(self, loss: Any) -> Any:
        """Scale loss for mixed precision training.

        Args:
            loss: Loss tensor.

        Returns:
            Scaled loss tensor, or original if AMP is disabled.
        """
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def scale_and_step(
        self,
        loss: Any,
        optimizer: Any,
        clip_norm: float = 0.0,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Perform scaled backward pass + optimizer step.

        Handles gradient unscaling, optional gradient clipping,
        and scaler update. Returns gradient statistics.

        Args:
            loss: Loss tensor.
            optimizer: PyTorch optimizer.
            clip_norm: Maximum gradient norm for clipping (0 = no clipping).
            optimizer_kwargs: Additional kwargs for optimizer.step().

        Returns:
            Dictionary with gradient statistics.
        """
        stats: Dict[str, float] = {}
        if not _is_torch_available():
            return stats

        import torch

        if self._scaler is not None:
            # Scaled backward
            self._scaler.scale(loss).backward()

            # Unscale before clipping
            if clip_norm > 0:
                self._scaler.unscale_(optimizer)

            # Gradient clipping
            if clip_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    clip_norm,
                )
                stats["grad_norm"] = float(total_norm)

            # Step with scale check
            scale_before = self._scaler.get_scale()
            kwargs = optimizer_kwargs or {}
            self._scaler.step(optimizer, **kwargs)
            self._scaler.update()
            scale_after = self._scaler.get_scale()

            stats["scale_before"] = float(scale_before)
            stats["scale_after"] = float(scale_after)
            stats["skipped_step"] = 1.0 if scale_after < scale_before else 0.0
        else:
            # Standard backward
            loss.backward()

            if clip_norm > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    clip_norm,
                )
                stats["grad_norm"] = float(total_norm)

            kwargs = optimizer_kwargs or {}
            optimizer.step(**kwargs)

        return stats

    # ----------------------------------------------------------------
    # Multi-GPU Orchestration
    # ----------------------------------------------------------------

    def get_device_for_rank(self, rank: int) -> str:
        """Get the device string for a given distributed rank.

        Maps rank to GPU device ID when multiple GPUs are available.

        Args:
            rank: Global rank of the process.

        Returns:
            Device string (e.g., 'cuda:0', 'cuda:1', 'cpu').
        """
        if not self._device_ids:
            return "cpu"

        device_id = self._device_ids[rank % len(self._device_ids)]
        return f"cuda:{device_id}"

    def balance_tensors_across_gpus(
        self,
        tensors: List[Any],
        strategy: str = "round_robin",
    ) -> Dict[int, List[Any]]:
        """Distribute tensors across available GPUs.

        Args:
            tensors: List of PyTorch tensors to distribute.
            strategy: Distribution strategy ('round_robin' or 'size_based').

        Returns:
            Dictionary mapping device_id -> list of tensors on that device.
        """
        if not self._device_ids or not _is_torch_available():
            return {-1: tensors}

        distribution: Dict[int, List[Any]] = {
            dev_id: [] for dev_id in self._device_ids
        }

        if strategy == "round_robin":
            for i, tensor in enumerate(tensors):
                dev_id = self._device_ids[i % len(self._device_ids)]
                distribution[dev_id].append(tensor.to(f"cuda:{dev_id}"))
        elif strategy == "size_based":
            # Sort by tensor size descending, then greedily assign to least loaded GPU
            import torch
            indexed = [(t.element_size() * t.nelement(), idx, t) for idx, t in enumerate(tensors)]
            indexed.sort(key=lambda x: x[0], reverse=True)
            gpu_loads = {dev_id: 0 for dev_id in self._device_ids}
            for size, _, tensor in indexed:
                least_loaded = min(gpu_loads, key=gpu_loads.get)
                distribution[least_loaded].append(tensor.to(f"cuda:{least_loaded}"))
                gpu_loads[least_loaded] += size
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return distribution

    # ----------------------------------------------------------------
    # OOM Prevention
    # ----------------------------------------------------------------

    def try_on_device(
        self,
        fn: Any,
        *args: Any,
        fallback_to_cpu: bool = True,
        retry_after_clear: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with OOM prevention and CPU fallback.

        Tries to run the function on GPU. If OOM occurs:
        1. Clears cache and retries (if retry_after_clear is True)
        2. Falls back to CPU (if fallback_to_cpu is True)

        Args:
            fn: Callable to execute.
            *args: Positional arguments for fn.
            fallback_to_cpu: Whether to fall back to CPU on OOM.
            retry_after_clear: Whether to retry after clearing cache.
            **kwargs: Keyword arguments for fn.

        Returns:
            Result of fn(*args, **kwargs).

        Raises:
            MemoryError: If all fallback strategies are exhausted.
        """
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise

            self._oom_count += 1
            logger.warning(
                "GPU OOM (#%d) in %s", self._oom_count, fn.__name__ if hasattr(fn, '__name__') else 'function'
            )

            if retry_after_clear:
                self.empty_cache()
                try:
                    return fn(*args, **kwargs)
                except RuntimeError as e2:
                    if "out of memory" not in str(e2).lower():
                        raise

            if fallback_to_cpu and self._device != "cpu":
                logger.warning("Falling back to CPU for %s",
                               fn.__name__ if hasattr(fn, '__name__') else 'function')
                # Move all tensor arguments to CPU
                cpu_args = tuple(self._move_to_cpu(a) for a in args)
                cpu_kwargs = {k: self._move_to_cpu(v) for k, v in kwargs.items()}
                return fn(*cpu_args, **cpu_kwargs)

            raise MemoryError(
                f"GPU OOM and no fallback available. OOM count: {self._oom_count}"
            ) from e

    def _move_to_cpu(self, obj: Any) -> Any:
        """Move an object to CPU if it's a PyTorch tensor."""
        if not _is_torch_available():
            return obj
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: self._move_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved = [self._move_to_cpu(item) for item in obj]
            return type(obj)(moved) if isinstance(obj, tuple) else moved
        return obj

    # ----------------------------------------------------------------
    # Monitoring
    # ----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU status for monitoring dashboards.

        Returns:
            Dictionary with device info, memory, and configuration.
        """
        memory_info = self.get_all_memory_info()
        return {
            "device": self._device,
            "is_gpu": self.is_gpu,
            "num_gpus": self.num_gpus,
            "amp_enabled": self._amp_enabled,
            "oom_count": self._oom_count,
            "devices": [
                {
                    "device_id": info.device_id,
                    "name": info.name,
                    "total_memory_mb": info.total_memory_mb,
                    "compute_capability": list(info.compute_capability),
                    "is_available": info.is_available,
                }
                for info in self._devices_info
            ],
            "memory": {
                str(dev_id): {
                    "total_mb": mi.total_mb,
                    "allocated_mb": mi.allocated_mb,
                    "free_mb": mi.free_mb,
                    "utilization": mi.utilization,
                }
                for dev_id, mi in memory_info.items()
            },
            "config": {
                "mixed_precision": self._config.enable_mixed_precision,
                "amp_dtype": self._config.amp_dtype,
                "memory_fraction": self._config.memory_fraction,
                "gradient_checkpointing": self._config.gradient_checkpointing,
            },
        }

    def reset(self) -> None:
        """Reset the GPU manager to its initial state.

        Clears all caches, resets OOM counter, and removes singleton.
        Useful for testing.
        """
        self.empty_cache()
        self._oom_count = 0
        self._memory_history.clear()
        self._scaler = None
        self._amp_enabled = False
        logger.info("GPUManager reset")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance entirely.

        WARNING: Only use in testing. Not thread-safe.
        """
        if cls._instance is not None:
            cls._instance.reset()
        cls._instance = None
        global _TORCH_AVAILABLE
        _TORCH_AVAILABLE = None


# ============================================================================
# Module-Level Convenience Functions
# ============================================================================


def get_gpu_manager() -> GPUManager:
    """Get the singleton GPUManager instance.

    Auto-initializes with default config if not yet initialized.

    Returns:
        GPUManager singleton instance.
    """
    manager = GPUManager()
    if not manager._initialized:
        manager.initialize()
    return manager


def device() -> str:
    """Get the current compute device string.

    Returns:
        Device string (e.g., 'cuda', 'cuda:0', 'cpu').
    """
    return get_gpu_manager().device


def is_gpu_available() -> bool:
    """Check if GPU is available for computation.

    Returns:
        True if CUDA is available.
    """
    return _is_cuda_available()


def to_device(data: Any, target_device: Optional[str] = None) -> Any:
    """Move data to the specified device.

    Convenience wrapper around GPUManager.to_device().

    Args:
        data: Data to move.
        target_device: Target device. Uses current device if None.

    Returns:
        Data on the target device.
    """
    return get_gpu_manager().to_device(data, target_device)


def empty_cache() -> None:
    """Empty the GPU cache."""
    get_gpu_manager().empty_cache()


def memory_info(device_id: Optional[int] = None) -> GPUMemoryInfo:
    """Get GPU memory information.

    Args:
        device_id: GPU device ID. Uses primary device if None.

    Returns:
        GPUMemoryInfo snapshot.
    """
    return get_gpu_manager().get_memory_info(device_id)
