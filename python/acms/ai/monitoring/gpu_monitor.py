"""GPU Resource Monitoring for ACMS AI training and inference.

Implements:
- GPUMonitor: Real-time GPU tracking with memory, utilization, temperature
- GPUInfo: GPU device information dataclass
- TrainingJobQueue: Training job queue management with priority
- Resource allocation optimization
- GPU health alerts and cost tracking
- Automatic scaling recommendations
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class GPUHealth(str, Enum):
    """GPU health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


class JobPriority(str, Enum):
    """Training job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobStatus(str, Enum):
    """Training job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GPUInfo:
    """GPU device information and current state.

    Attributes:
        device_id: GPU device index.
        name: GPU model name.
        total_memory_mb: Total GPU memory in megabytes.
        used_memory_mb: Currently used memory in megabytes.
        utilization_pct: GPU compute utilization percentage.
        temperature_c: GPU temperature in Celsius.
        power_usage_w: Current power draw in watts.
        power_limit_w: Maximum power limit in watts.
        health: Current health status.
        last_updated: Timestamp of last status update.
    """
    device_id: int = 0
    name: str = "unknown"
    total_memory_mb: int = 0
    used_memory_mb: int = 0
    utilization_pct: float = 0.0
    temperature_c: float = 0.0
    power_usage_w: float = 0.0
    power_limit_w: float = 0.0
    health: GPUHealth = GPUHealth.OFFLINE
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def free_memory_mb(self) -> int:
        """Available GPU memory in megabytes."""
        return max(0, self.total_memory_mb - self.used_memory_mb)

    @property
    def memory_utilization_pct(self) -> float:
        """Memory utilization as percentage."""
        return (self.used_memory_mb / self.total_memory_mb * 100) if self.total_memory_mb > 0 else 0.0

    @property
    def power_utilization_pct(self) -> float:
        """Power utilization as percentage of limit."""
        return (self.power_usage_w / self.power_limit_w * 100) if self.power_limit_w > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "total_memory_mb": self.total_memory_mb,
            "used_memory_mb": self.used_memory_mb,
            "free_memory_mb": self.free_memory_mb,
            "memory_utilization_pct": round(self.memory_utilization_pct, 1),
            "utilization_pct": round(self.utilization_pct, 1),
            "temperature_c": round(self.temperature_c, 1),
            "power_usage_w": round(self.power_usage_w, 1),
            "power_limit_w": round(self.power_limit_w, 1),
            "power_utilization_pct": round(self.power_utilization_pct, 1),
            "health": self.health.value,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class TrainingJob:
    """A training job in the queue.

    Attributes:
        job_id: Unique job identifier.
        model_id: Model to train.
        config: Training configuration dict.
        priority: Job priority.
        status: Current job status.
        gpu_device_id: Assigned GPU device (-1 if unassigned).
        required_memory_mb: Required GPU memory.
        submitted_at: When the job was submitted.
        started_at: When execution started.
        completed_at: When execution completed.
        error_message: Error message if job failed.
        progress_pct: Training progress percentage.
    """
    job_id: str
    model_id: str
    config: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.QUEUED
    gpu_device_id: int = -1
    required_memory_mb: int = 4096
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_pct: float = 0.0

    @property
    def wait_time_seconds(self) -> float:
        """Time spent waiting in queue."""
        start = self.started_at or datetime.utcnow()
        return (start - self.submitted_at).total_seconds()

    @property
    def runtime_seconds(self) -> Optional[float]:
        """Job runtime if started."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()

    @property
    def priority_value(self) -> int:
        """Numeric priority for sorting (higher = more urgent)."""
        return {"urgent": 4, "high": 3, "normal": 2, "low": 1}.get(self.priority.value, 2)


# ============================================================================
# Training Job Queue
# ============================================================================

class TrainingJobQueue:
    """Priority-based training job queue with GPU allocation.

    Manages training jobs, assigns them to available GPUs,
    and tracks job lifecycle from submission to completion.
    """

    def __init__(self, max_concurrent_per_gpu: int = 1):
        """Initialize the training job queue.

        Args:
            max_concurrent_per_gpu: Maximum concurrent jobs per GPU.
        """
        self.max_concurrent_per_gpu = max_concurrent_per_gpu
        self._jobs: Dict[str, TrainingJob] = {}
        self._gpu_assignments: Dict[int, List[str]] = defaultdict(list)
        self._job_counter = 0

    def submit(self, model_id: str, config: Dict[str, Any],
               priority: JobPriority = JobPriority.NORMAL,
               required_memory_mb: int = 4096) -> str:
        """Submit a new training job.

        Args:
            model_id: Model to train.
            config: Training configuration.
            priority: Job priority.
            required_memory_mb: Required GPU memory.

        Returns:
            Job identifier.
        """
        self._job_counter += 1
        job_id = f"train_{self._job_counter:06d}"
        job = TrainingJob(
            job_id=job_id,
            model_id=model_id,
            config=config,
            priority=priority,
            required_memory_mb=required_memory_mb,
        )
        self._jobs[job_id] = job
        logger.info("Training job '%s' submitted for model '%s' with priority '%s'",
                     job_id, model_id, priority.value)
        return job_id

    def get_next_job(self, available_gpus: Dict[int, GPUInfo]) -> Optional[Tuple[str, int]]:
        """Get the next job to execute and assign it to a GPU.

        Args:
            available_gpus: Dict of GPU device ID to GPUInfo.

        Returns:
            Tuple of (job_id, gpu_device_id) or None if no job can be scheduled.
        """
        # Get queued jobs sorted by priority (highest first), then by submission time
        queued_jobs = [
            j for j in self._jobs.values() if j.status == JobStatus.QUEUED
        ]
        queued_jobs.sort(key=lambda j: (-j.priority_value, j.submitted_at))

        for job in queued_jobs:
            # Find an available GPU with enough memory
            for device_id, gpu_info in available_gpus.items():
                current_jobs = self._gpu_assignments.get(device_id, [])
                running_count = sum(
                    1 for jid in current_jobs
                    if self._jobs.get(jid, TrainingJob(job_id="", model_id="")).status == JobStatus.RUNNING
                )
                if (running_count < self.max_concurrent_per_gpu and
                        gpu_info.free_memory_mb >= job.required_memory_mb and
                        gpu_info.health in (GPUHealth.HEALTHY, GPUHealth.WARNING)):
                    # Assign job to this GPU
                    job.gpu_device_id = device_id
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.utcnow()
                    self._gpu_assignments[device_id].append(job.job_id)
                    logger.info("Job '%s' assigned to GPU %d", job.job_id, device_id)
                    return job.job_id, device_id

        return None

    def complete_job(self, job_id: str, error_message: Optional[str] = None) -> bool:
        """Mark a job as completed or failed.

        Args:
            job_id: Job identifier.
            error_message: Error message if the job failed.

        Returns:
            True if the job was found and updated.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return False

        job.completed_at = datetime.utcnow()
        job.progress_pct = 100.0 if error_message is None else job.progress_pct

        if error_message:
            job.status = JobStatus.FAILED
            job.error_message = error_message
            logger.error("Job '%s' failed: %s", job_id, error_message)
        else:
            job.status = JobStatus.COMPLETED
            logger.info("Job '%s' completed successfully", job_id)

        # Remove from GPU assignments
        if job.gpu_device_id >= 0:
            assignments = self._gpu_assignments.get(job.gpu_device_id, [])
            if job_id in assignments:
                assignments.remove(job_id)

        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        Args:
            job_id: Job identifier.

        Returns:
            True if the job was cancelled.
        """
        job = self._jobs.get(job_id)
        if job is None or job.status not in (JobStatus.QUEUED, JobStatus.RUNNING):
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()
        if job.gpu_device_id >= 0:
            assignments = self._gpu_assignments.get(job.gpu_device_id, [])
            if job_id in assignments:
                assignments.remove(job_id)
        logger.info("Job '%s' cancelled", job_id)
        return True

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status summary.

        Returns:
            Dict with job counts by status and queue depth.
        """
        status_counts = defaultdict(int)
        for job in self._jobs.values():
            status_counts[job.status.value] += 1

        queued = [j for j in self._jobs.values() if j.status == JobStatus.QUEUED]
        avg_wait = 0.0
        if queued:
            now = datetime.utcnow()
            waits = [(now - j.submitted_at).total_seconds() for j in queued]
            avg_wait = float(np.mean(waits))

        return {
            "total_jobs": len(self._jobs),
            "status_counts": dict(status_counts),
            "queue_depth": status_counts.get("queued", 0),
            "running_jobs": status_counts.get("running", 0),
            "average_wait_seconds": round(avg_wait, 1),
        }


# ============================================================================
# GPU Monitor
# ============================================================================

class GPUMonitor:
    """Real-time GPU resource monitoring with health alerts and cost tracking.

    Tracks GPU utilization, memory usage, temperature, and power consumption.
    Provides health alerts, cost estimation for cloud GPU usage, and
    automatic scaling recommendations.
    """

    def __init__(self, temp_warning_threshold: float = 80.0,
                 temp_critical_threshold: float = 90.0,
                 memory_warning_pct: float = 90.0,
                 utilization_low_pct: float = 10.0,
                 gpu_hourly_cost: float = 2.48,
                 history_window: int = 3600,
                 check_interval_seconds: float = 10.0):
        """Initialize the GPU monitor.

        Args:
            temp_warning_threshold: Temperature for warning alert (Celsius).
            temp_critical_threshold: Temperature for critical alert (Celsius).
            memory_warning_pct: Memory usage percentage for warning.
            utilization_low_pct: GPU utilization below this is considered underused.
            gpu_hourly_cost: Hourly cost per GPU (for cloud cost tracking).
            history_window: Number of historical data points to retain.
            check_interval_seconds: How often to poll GPU stats.
        """
        self.temp_warning_threshold = temp_warning_threshold
        self.temp_critical_threshold = temp_critical_threshold
        self.memory_warning_pct = memory_warning_pct
        self.utilization_low_pct = utilization_low_pct
        self.gpu_hourly_cost = gpu_hourly_cost
        self.history_window = history_window
        self.check_interval_seconds = check_interval_seconds

        self._gpus: Dict[int, GPUInfo] = {}
        self._history: Dict[int, Deque[Dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=history_window)
        )
        self._alerts: List[Dict[str, Any]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[datetime] = None
        self._total_cost: float = 0.0
        self.job_queue = TrainingJobQueue()

    async def start(self) -> None:
        """Start GPU monitoring loop."""
        self._running = True
        self._start_time = datetime.utcnow()
        self._detect_gpus()
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("GPU monitor started, tracking %d GPU(s)", len(self._gpus))

    async def stop(self) -> None:
        """Stop GPU monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.debug("GPU monitor task cancelled during stop")
        logger.info("GPU monitor stopped")

    def _detect_gpus(self) -> None:
        """Detect available GPUs using pynvml or torch."""
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self._gpus[i] = GPUInfo(
                    device_id=i,
                    name=name,
                    total_memory_mb=int(mem_info.total / (1024 * 1024)),
                    used_memory_mb=int(mem_info.used / (1024 * 1024)),
                    health=GPUHealth.HEALTHY,
                )
            pynvml.nvmlShutdown()
            logger.info("Detected %d NVIDIA GPU(s)", device_count)
        except ImportError:
            self._detect_gpus_torch()

    def _detect_gpus_torch(self) -> None:
        """Fallback GPU detection using PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    total_mem = torch.cuda.get_device_properties(i).total_mem
                    self._gpus[i] = GPUInfo(
                        device_id=i,
                        name=name,
                        total_memory_mb=int(total_mem / (1024 * 1024)),
                        health=GPUHealth.HEALTHY,
                    )
                logger.info("Detected %d GPU(s) via PyTorch", device_count)
            else:
                logger.info("No CUDA GPUs available")
        except ImportError:
            logger.warning("Neither pynvml nor torch available; GPU monitoring disabled")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop that periodically polls GPU stats."""
        while self._running:
            try:
                await self._update_stats()
                self._check_health()
                self._update_cost()
                self._schedule_jobs()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("GPU monitor loop error: %s", e)
            await asyncio.sleep(self.check_interval_seconds)

    async def _update_stats(self) -> None:
        """Update GPU statistics from drivers."""
        try:
            import pynvml
            pynvml.nvmlInit()
            for device_id, gpu_info in self._gpus.items():
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0

                    gpu_info.used_memory_mb = int(mem_info.used / (1024 * 1024))
                    gpu_info.utilization_pct = float(util.gpu)
                    gpu_info.temperature_c = float(temp)
                    gpu_info.power_usage_w = power
                    gpu_info.power_limit_w = power_limit
                    gpu_info.last_updated = datetime.utcnow()

                    # Store history
                    self._history[device_id].append({
                        "timestamp": time.time(),
                        "memory_pct": gpu_info.memory_utilization_pct,
                        "utilization_pct": gpu_info.utilization_pct,
                        "temperature_c": gpu_info.temperature_c,
                        "power_w": gpu_info.power_usage_w,
                    })
                except Exception as e:
                    logger.warning("Failed to update GPU %d stats: %s", device_id, e)
                    gpu_info.health = GPUHealth.OFFLINE
            pynvml.nvmlShutdown()
        except ImportError:
            logger.debug("pynvml not available for GPU stats update")

    def _check_health(self) -> None:
        """Check GPU health and generate alerts."""
        for device_id, gpu in self._gpus.items():
            old_health = gpu.health

            # Temperature checks
            if gpu.temperature_c >= self.temp_critical_threshold:
                gpu.health = GPUHealth.CRITICAL
                self._add_alert(device_id, "temperature_critical",
                                f"GPU {device_id} temperature {gpu.temperature_c:.1f}°C "
                                f"exceeds critical threshold {self.temp_critical_threshold}°C")
            elif gpu.temperature_c >= self.temp_warning_threshold:
                gpu.health = GPUHealth.WARNING
                self._add_alert(device_id, "temperature_warning",
                                f"GPU {device_id} temperature {gpu.temperature_c:.1f}°C "
                                f"exceeds warning threshold {self.temp_warning_threshold}°C")

            # Memory checks
            if gpu.memory_utilization_pct >= self.memory_warning_pct:
                if gpu.health == GPUHealth.HEALTHY:
                    gpu.health = GPUHealth.WARNING
                self._add_alert(device_id, "memory_high",
                                f"GPU {device_id} memory usage {gpu.memory_utilization_pct:.1f}% "
                                f"exceeds {self.memory_warning_pct}%")

            # Reset health if conditions improve
            if old_health in (GPUHealth.WARNING, GPUHealth.CRITICAL):
                if (gpu.temperature_c < self.temp_warning_threshold and
                        gpu.memory_utilization_pct < self.memory_warning_pct):
                    gpu.health = GPUHealth.HEALTHY

    def _add_alert(self, device_id: int, alert_type: str, message: str) -> None:
        """Add a GPU health alert.

        Args:
            device_id: GPU device index.
            alert_type: Type of alert.
            message: Alert message.
        """
        # Deduplicate: don't add same alert type for same GPU within 5 minutes
        now = time.time()
        for existing in self._alerts[-20:]:
            if (existing.get("device_id") == device_id and
                    existing.get("alert_type") == alert_type and
                    now - existing.get("timestamp", 0) < 300):
                return

        self._alerts.append({
            "device_id": device_id,
            "alert_type": alert_type,
            "message": message,
            "timestamp": now,
            "gpu_name": self._gpus[device_id].name if device_id in self._gpus else "unknown",
        })

    def _update_cost(self) -> None:
        """Update cumulative cloud GPU cost."""
        if self._start_time is None:
            return
        elapsed_hours = (datetime.utcnow() - self._start_time).total_seconds() / 3600
        active_gpus = sum(1 for g in self._gpus.values()
                          if g.health != GPUHealth.OFFLINE)
        self._total_cost = elapsed_hours * active_gpus * self.gpu_hourly_cost

    def _schedule_jobs(self) -> None:
        """Attempt to schedule queued training jobs to available GPUs."""
        while True:
            result = self.job_queue.get_next_job(self._gpus)
            if result is None:
                break
            job_id, device_id = result
            logger.info("Scheduled job '%s' on GPU %d", job_id, device_id)

    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Generate GPU scaling recommendations based on utilization.

        Analyzes historical utilization and queue depth to recommend
        scaling up or down.

        Returns:
            Dict with scaling recommendation and reasoning.
        """
        if not self._gpus:
            return {"recommendation": "no_gpus", "reason": "No GPUs detected"}

        # Average utilization across all GPUs
        avg_utils = []
        for device_id in self._gpus:
            history = list(self._history.get(device_id, []))
            if history:
                recent_utils = [h["utilization_pct"] for h in history[-60:]]  # Last 10 min
                avg_utils.append(float(np.mean(recent_utils)))

        avg_util = float(np.mean(avg_utils)) if avg_utils else 0.0
        queue_status = self.job_queue.get_queue_status()
        queue_depth = queue_status.get("queue_depth", 0)

        recommendation = "maintain"
        reason = f"Average utilization: {avg_util:.1f}%"
        target_gpu_count = len(self._gpus)

        if avg_util > 85 and queue_depth > 0:
            recommendation = "scale_up"
            target_gpu_count = len(self._gpus) + max(1, queue_depth // 2)
            reason = f"High utilization ({avg_util:.1f}%) with {queue_depth} queued jobs"
        elif avg_util < self.utilization_low_pct and queue_depth == 0:
            recommendation = "scale_down"
            target_gpu_count = max(1, len(self._gpus) - 1)
            reason = f"Low utilization ({avg_util:.1f}%) with no queued jobs"

        return {
            "recommendation": recommendation,
            "current_gpu_count": len(self._gpus),
            "target_gpu_count": target_gpu_count,
            "average_utilization_pct": round(avg_util, 1),
            "queue_depth": queue_depth,
            "reason": reason,
            "estimated_cost_per_hour": round(
                target_gpu_count * self.gpu_hourly_cost, 2
            ),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive GPU monitoring status.

        Returns:
            Dict with GPU info, alerts, cost, and scaling recommendation.
        """
        gpu_list = [g.to_dict() for g in self._gpus.values()]
        scaling = self.get_scaling_recommendation()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "gpu_count": len(self._gpus),
            "gpus": gpu_list,
            "alerts": self._alerts[-20:],
            "total_cost_usd": round(self._total_cost, 2),
            "job_queue": self.job_queue.get_queue_status(),
            "scaling_recommendation": scaling,
        }

    def get_gpu_history(self, device_id: int, limit: int = 360) -> List[Dict[str, float]]:
        """Get historical GPU metrics for a device.

        Args:
            device_id: GPU device index.
            limit: Maximum number of data points.

        Returns:
            List of metric dicts.
        """
        history = list(self._history.get(device_id, []))
        return history[-limit:]
