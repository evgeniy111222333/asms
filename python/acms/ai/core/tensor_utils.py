"""Tensor operations and data utilities for the ACMS AI module.

Provides a comprehensive suite of tensor manipulation utilities
for the AI pipeline:

- TensorDataset: Lazy loading dataset from Parquet files
- StreamingDataLoader: Memory-efficient streaming for large datasets
- TensorCache: LRU tensor cache with configurable eviction
- Device-aware tensor creation and movement utilities
- Normalization / standardization utilities
- Sequence padding and masking for variable-length series
- Attention mask generation
- Temporal split utilities for time series
- Sliding window dataset generator
- Data augmentation for time series (jitter, scaling, masking)
- Efficient batch collation functions

All utilities handle the case where PyTorch is not installed by
falling back to numpy operations where possible.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Lazy Tensor Import
# ============================================================================

_TORCH_AVAILABLE: Optional[bool] = None


def _check_torch() -> bool:
    """Check if PyTorch is available (cached)."""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def _import_torch():
    """Import and return torch module.

    Returns:
        torch module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    import torch
    return torch


# ============================================================================
# Tensor Dataset with Lazy Parquet Loading
# ============================================================================


class TensorDataset:
    """Lazy-loading dataset from Parquet files.

    Supports loading features and targets from Parquet files with
    memory-mapped access and optional feature selection.

    Attributes:
        features_path: Path to Parquet file containing features.
        targets_path: Path to Parquet file containing targets.
        feature_columns: Columns to load as features (None = all).
        target_columns: Columns to load as targets (None = all).
        cache_in_memory: Whether to cache loaded data in memory.
    """

    def __init__(
        self,
        features_path: Union[str, Path],
        targets_path: Optional[Union[str, Path]] = None,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        cache_in_memory: bool = False,
    ) -> None:
        """Initialize the tensor dataset.

        Args:
            features_path: Path to Parquet file with features.
            targets_path: Path to Parquet file with targets.
            feature_columns: Specific feature columns to load.
            target_columns: Specific target columns to load.
            cache_in_memory: Cache loaded data in memory.
        """
        self.features_path = Path(features_path)
        self.targets_path = Path(targets_path) if targets_path else None
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.cache_in_memory = cache_in_memory

        self._features: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self._length: Optional[int] = None

    @property
    def length(self) -> int:
        """Number of samples in the dataset."""
        if self._length is None:
            self._length = self._peek_length()
        return self._length

    def _peek_length(self) -> int:
        """Peek at the dataset length without full load."""
        try:
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(self.features_path)
            return metadata.num_rows
        except ImportError:
            logger.debug("pyarrow not available for Parquet metadata peek")
        except Exception as e:
            logger.warning("Could not peek at Parquet metadata: %s", e)

        # Fallback: load and count
        features = self._load_features()
        return len(features)

    def _load_features(self) -> np.ndarray:
        """Load features from Parquet file."""
        if self._features is not None:
            return self._features

        try:
            import pandas as pd
            df = pd.read_parquet(self.features_path, columns=self.feature_columns)
            features = df.values.astype(np.float32)
        except ImportError:
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(self.features_path, columns=self.feature_columns)
                features = table.to_pandas().values.astype(np.float32)
            except ImportError:
                raise ImportError(
                    "pandas or pyarrow is required for Parquet loading. "
                    "Install with: pip install pandas pyarrow"
                )

        if self.cache_in_memory:
            self._features = features

        return features

    def _load_targets(self) -> Optional[np.ndarray]:
        """Load targets from Parquet file."""
        if self._targets is not None:
            return self._targets

        if self.targets_path is None:
            return None

        try:
            import pandas as pd
            df = pd.read_parquet(self.targets_path, columns=self.target_columns)
            targets = df.values.astype(np.float32)
        except ImportError:
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(self.targets_path, columns=self.target_columns)
                targets = table.to_pandas().values.astype(np.float32)
            except ImportError:
                raise ImportError(
                    "pandas or pyarrow is required for Parquet loading."
                )

        if self.cache_in_memory:
            self._targets = targets

        return targets

    def get_features(self) -> np.ndarray:
        """Get the full feature matrix.

        Returns:
            Feature array of shape (n_samples, n_features).
        """
        return self._load_features()

    def get_targets(self) -> Optional[np.ndarray]:
        """Get the full target array.

        Returns:
            Target array, or None if no targets.
        """
        return self._load_targets()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Get a single sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features, target) arrays.
        """
        features = self._load_features()
        targets = self._load_targets()
        x = features[idx]
        y = targets[idx] if targets is not None else None
        return x, y

    def to_torch_dataset(self):
        """Convert to a PyTorch TensorDataset.

        Returns:
            torch.utils.data.TensorDataset.

        Raises:
            ImportError: If PyTorch is not installed.
        """
        if not _check_torch():
            raise ImportError("PyTorch is required for to_torch_dataset()")

        torch = _import_torch()
        from torch.utils.data import TensorDataset as TorchTensorDataset

        features = torch.from_numpy(self._load_features())
        targets = self._load_targets()

        if targets is not None:
            targets_tensor = torch.from_numpy(targets)
            return TorchTensorDataset(features, targets_tensor)
        return TorchTensorDataset(features)


# ============================================================================
# Streaming DataLoader
# ============================================================================


class StreamingDataLoader:
    """Memory-efficient streaming data loader for large datasets.

    Loads data in chunks from Parquet files, enabling processing
    of datasets that don't fit in memory.

    Args:
        features_path: Path to Parquet file with features.
        targets_path: Path to Parquet file with targets.
        chunk_size: Number of rows per chunk.
        feature_columns: Specific feature columns to load.
        target_columns: Specific target columns to load.
        shuffle_chunks: Whether to shuffle chunk order.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        features_path: Union[str, Path],
        targets_path: Optional[Union[str, Path]] = None,
        chunk_size: int = 10000,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        shuffle_chunks: bool = False,
        seed: int = 42,
    ) -> None:
        self.features_path = Path(features_path)
        self.targets_path = Path(targets_path) if targets_path else None
        self.chunk_size = chunk_size
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed

        self._total_rows = self._count_rows()
        self._n_chunks = math.ceil(self._total_rows / chunk_size)

    def _count_rows(self) -> int:
        """Count total rows in the dataset."""
        try:
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(self.features_path)
            return metadata.num_rows
        except Exception as e:
            logger.warning("Could not count rows via Parquet metadata: %s", e)

        import pandas as pd
        df = pd.read_parquet(self.features_path, columns=[])
        return len(df)

    @property
    def total_rows(self) -> int:
        """Total number of rows in the dataset."""
        return self._total_rows

    @property
    def n_chunks(self) -> int:
        """Number of chunks."""
        return self._n_chunks

    def iter_chunks(self) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
        """Iterate over data chunks.

        Yields:
            Tuple of (features_chunk, targets_chunk) numpy arrays.
        """
        import pandas as pd

        chunk_indices = list(range(self._n_chunks))
        if self.shuffle_chunks:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(chunk_indices)

        for chunk_idx in chunk_indices:
            skip = chunk_idx * self.chunk_size

            features_df = pd.read_parquet(
                self.features_path,
                columns=self.feature_columns,
                skiprows=lambda i: i < skip and i >= skip + self.chunk_size,
            )
            # More reliable chunk reading
            features_df = pd.read_parquet(
                self.features_path,
                columns=self.feature_columns,
            ).iloc[skip:skip + self.chunk_size]

            features = features_df.values.astype(np.float32)

            targets = None
            if self.targets_path is not None:
                targets_df = pd.read_parquet(
                    self.targets_path,
                    columns=self.target_columns,
                ).iloc[skip:skip + self.chunk_size]
                targets = targets_df.values.astype(np.float32)

            yield features, targets

    def iter_batches(
        self,
        batch_size: int = 64,
    ) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
        """Iterate over mini-batches across all chunks.

        Args:
            batch_size: Mini-batch size.

        Yields:
            Tuple of (features_batch, targets_batch) numpy arrays.
        """
        buffer_features: List[np.ndarray] = []
        buffer_targets: List[np.ndarray] = []
        buffer_size = 0

        for chunk_features, chunk_targets in self.iter_chunks():
            buffer_features.append(chunk_features)
            if chunk_targets is not None:
                buffer_targets.append(chunk_targets)
            buffer_size += len(chunk_features)

            while buffer_size >= batch_size:
                # Concatenate buffer
                all_features = np.concatenate(buffer_features, axis=0)
                all_targets = (
                    np.concatenate(buffer_targets, axis=0)
                    if buffer_targets
                    else None
                )

                yield all_features[:batch_size], (
                    all_targets[:batch_size] if all_targets is not None else None
                )

                # Keep remainder in buffer
                remainder_features = all_features[batch_size:]
                buffer_features = [remainder_features]
                buffer_targets = [all_targets[batch_size:]] if all_targets is not None else []
                buffer_size = len(remainder_features)

        # Yield remaining buffer
        if buffer_size > 0:
            all_features = np.concatenate(buffer_features, axis=0)
            all_targets = (
                np.concatenate(buffer_targets, axis=0)
                if buffer_targets
                else None
            )
            yield all_features, all_targets


# ============================================================================
# Tensor Cache with LRU Eviction
# ============================================================================


class TensorCache:
    """LRU cache for tensors with configurable size limits.

    Provides thread-safe caching of computed tensors (e.g., features,
    embeddings) with least-recently-used eviction when the cache
    exceeds its size limit.

    Args:
        max_size: Maximum number of items in the cache.
        max_memory_mb: Maximum total memory usage in MB (0 = unlimited).
        ttl_seconds: Time-to-live for cached items (0 = no expiry).
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 4096,
        ttl_seconds: float = 0,
    ) -> None:
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict[str, Tuple[np.ndarray, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve a cached tensor.

        Args:
            key: Cache key.

        Returns:
            Cached tensor, or None if not found or expired.
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            tensor, timestamp = self._cache[key]

            # Check TTL
            if self.ttl_seconds > 0:
                age = time.time() - timestamp
                if age > self.ttl_seconds:
                    del self._cache[key]
                    self._misses += 1
                    return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return tensor

    def put(self, key: str, tensor: np.ndarray) -> None:
        """Store a tensor in the cache.

        Args:
            key: Cache key.
            tensor: Tensor to cache.
        """
        with self._lock:
            if key in self._cache:
                # Remove old entry to update
                del self._cache[key]

            self._cache[key] = (tensor, time.time())
            self._enforce_limits()

    def has(self, key: str) -> bool:
        """Check if a key exists in the cache (and is not expired).

        Args:
            key: Cache key.

        Returns:
            True if the key exists and is valid.
        """
        return self.get(key) is not None

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from the cache.

        Args:
            key: Cache key.

        Returns:
            True if the key was found and removed.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def _enforce_limits(self) -> None:
        """Enforce size and memory limits by evicting LRU items."""
        # Enforce max_size
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # Remove oldest

        # Enforce max_memory_mb
        if self.max_memory_mb > 0:
            total_mb = self._estimate_memory_mb()
            while total_mb > self.max_memory_mb and self._cache:
                self._cache.popitem(last=False)
                total_mb = self._estimate_memory_mb()

    def _estimate_memory_mb(self) -> float:
        """Estimate total memory usage of cached items.

        Returns:
            Estimated memory usage in MB.
        """
        total_bytes = 0
        for tensor, _ in self._cache.values():
            total_bytes += tensor.nbytes
        return total_bytes / (1024 * 1024)

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "memory_mb": self._estimate_memory_mb(),
        }


# ============================================================================
# Device-Aware Tensor Creation
# ============================================================================


def create_tensor(
    data: Union[List, np.ndarray, float, int],
    dtype: Optional[str] = None,
    device: Optional[str] = None,
) -> Union[np.ndarray, Any]:
    """Create a tensor with optional device placement.

    Creates a numpy array by default, or a PyTorch tensor if
    a GPU device is specified.

    Args:
        data: Input data.
        dtype: Data type string ('float32', 'float16', 'int64', etc.).
        device: Target device ('cpu', 'cuda', 'cuda:0', None for numpy).

    Returns:
        Numpy array or PyTorch tensor.
    """
    dtype = dtype or "float32"
    np_dtype = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
        "bool": np.bool_,
    }.get(dtype, np.float32)

    arr = np.asarray(data, dtype=np_dtype)

    if device is not None and device != "cpu" and _check_torch():
        torch = _import_torch()
        torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }.get(dtype, torch.float32)
        return torch.from_numpy(arr).to(device=device, dtype=torch_dtype)

    return arr


def move_to_device(
    data: Any,
    device: str,
) -> Any:
    """Move data to a specific device.

    Handles numpy arrays, PyTorch tensors, and nested structures.

    Args:
        data: Data to move.
        device: Target device string.

    Returns:
        Data on the target device.
    """
    if isinstance(data, np.ndarray):
        if device == "cpu" or not _check_torch():
            return data
        torch = _import_torch()
        return torch.from_numpy(data).to(device)

    if _check_torch():
        torch = _import_torch()
        if isinstance(data, torch.Tensor):
            return data.to(device)

    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        moved = [move_to_device(item, device) for item in data]
        return type(data)(moved) if isinstance(data, tuple) else moved

    return data


# ============================================================================
# Normalization / Standardization
# ============================================================================


class StandardScaler:
    """Standard score normalization (z-score).

    Computes mean and standard deviation from training data and
    applies z-score normalization: (x - mean) / std.

    Attributes:
        mean: Feature means.
        std: Feature standard deviations.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Fit the scaler on training data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            Self for chaining.
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + self.epsilon
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.

        Args:
            X: Data to normalize.

        Returns:
            Normalized data.

        Raises:
            RuntimeError: If scaler has not been fitted.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Training data.

        Returns:
            Normalized data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the normalization.

        Args:
            X: Normalized data.

        Returns:
            Data in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return X * self.std + self.mean


class MinMaxScaler:
    """Min-max normalization to [0, 1] range.

    Attributes:
        min_: Feature minimums.
        range_: Feature ranges (max - min).
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.min_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Fit the scaler on training data.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            Self for chaining.
        """
        self.min_ = np.min(X, axis=0)
        self.range_ = np.max(X, axis=0) - self.min_ + self.epsilon
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply min-max normalization.

        Args:
            X: Data to normalize.

        Returns:
            Normalized data in [0, 1].
        """
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (X - self.min_) / self.range_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the normalization."""
        if self.min_ is None or self.range_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return X * self.range_ + self.min_


class RobustScaler:
    """Robust scaling using median and IQR.

    Centers data using median and scales using interquartile range,
    making it robust to outliers.

    Attributes:
        center_: Feature medians.
        scale_: Feature IQRs.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = epsilon
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        """Fit using median and IQR.

        Args:
            X: Training data.

        Returns:
            Self for chaining.
        """
        self.center_ = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        self.scale_ = (q75 - q25) + self.epsilon
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply robust scaling."""
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the scaling."""
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return X * self.scale_ + self.center_


# ============================================================================
# Sequence Padding and Masking
# ============================================================================


def pad_sequences(
    sequences: List[np.ndarray],
    max_length: Optional[int] = None,
    padding_value: float = 0.0,
    padding_side: str = "right",
    truncation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad variable-length sequences to equal length.

    Args:
        sequences: List of 1-D or 2-D numpy arrays.
        max_length: Maximum sequence length. Auto-detected if None.
        padding_value: Value used for padding.
        padding_side: 'right' or 'left' padding.
        truncation: Whether to truncate sequences exceeding max_length.

    Returns:
        Tuple of (padded_sequences, attention_mask) where:
        - padded_sequences: Array of shape (n_seqs, max_length, ...) or (n_seqs, max_length)
        - attention_mask: Binary array of shape (n_seqs, max_length), 1=valid, 0=padding
    """
    if not sequences:
        return np.array([]), np.array([])

    # Determine max length
    lengths = [len(seq) for seq in sequences]
    if max_length is None:
        max_length = max(lengths)

    n_seqs = len(sequences)
    is_2d = sequences[0].ndim >= 2

    if is_2d:
        feature_dim = sequences[0].shape[1]
        padded = np.full(
            (n_seqs, max_length, feature_dim), padding_value, dtype=np.float32
        )
    else:
        padded = np.full(
            (n_seqs, max_length), padding_value, dtype=np.float32
        )

    mask = np.zeros((n_seqs, max_length), dtype=np.float32)

    for i, seq in enumerate(sequences):
        seq_len = min(len(seq), max_length) if truncation else len(seq)
        if truncation and len(seq) > max_length:
            seq = seq[:max_length]

        if padding_side == "right":
            if is_2d:
                padded[i, :seq_len, :] = seq[:seq_len]
            else:
                padded[i, :seq_len] = seq[:seq_len]
            mask[i, :seq_len] = 1.0
        else:  # left padding
            start = max_length - seq_len
            if is_2d:
                padded[i, start:, :] = seq[:seq_len]
            else:
                padded[i, start:] = seq[:seq_len]
            mask[i, start:] = 1.0

    return padded, mask


def create_attention_mask(
    lengths: List[int],
    max_length: Optional[int] = None,
    causal: bool = False,
) -> np.ndarray:
    """Create attention masks for variable-length sequences.

    Args:
        lengths: List of actual sequence lengths.
        max_length: Maximum sequence length. Auto-detected if None.
        causal: Whether to create a causal (lower triangular) mask.

    Returns:
        Attention mask of shape (n_seqs, max_length) for padding mask,
        or (n_seqs, max_length, max_length) for causal mask.
        1.0 = attend, 0.0 = masked.
    """
    if max_length is None:
        max_length = max(lengths) if lengths else 0

    n_seqs = len(lengths)
    mask = np.zeros((n_seqs, max_length), dtype=np.float32)

    for i, length in enumerate(lengths):
        mask[i, :length] = 1.0

    if causal:
        # Create causal mask: (n_seqs, max_length, max_length)
        causal_mask = np.tril(
            np.ones((max_length, max_length), dtype=np.float32)
        )
        # Combine with padding mask
        # (n_seqs, 1, max_length) * (1, max_length, max_length)
        causal_mask = mask[:, np.newaxis, :] * causal_mask[np.newaxis, :, :]
        return causal_mask

    return mask


# ============================================================================
# Temporal Split Utilities
# ============================================================================


@dataclass
class TemporalSplit:
    """Result of a temporal train/val/test split.

    Attributes:
        train_indices: Indices for the training set.
        val_indices: Indices for the validation set.
        test_indices: Indices for the test set.
        train_ratio: Fraction of data used for training.
        val_ratio: Fraction of data used for validation.
        test_ratio: Fraction of data used for testing.
        gap: Number of samples between train and val, and val and test.
    """

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    gap: int = 0


def temporal_split(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    gap: int = 0,
) -> TemporalSplit:
    """Split data temporally while preserving order.

    Creates a train/validation/test split that respects temporal
    ordering, with optional gaps between splits to prevent
    information leakage.

    Args:
        n_samples: Total number of samples.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        gap: Number of samples to skip between splits.

    Returns:
        TemporalSplit with index arrays.

    Raises:
        ValueError: If ratios don't sum to 1.0 or data is insufficient.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    total_gap = gap * 2
    effective_samples = n_samples - total_gap
    if effective_samples <= 0:
        raise ValueError(
            f"Not enough samples ({n_samples}) for gaps ({total_gap})"
        )

    train_end = int(effective_samples * train_ratio)
    val_end = train_end + int(effective_samples * val_ratio)

    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end + gap, train_end + gap + (val_end - train_end))
    test_start = val_end + gap
    test_indices = np.arange(test_start, n_samples)

    return TemporalSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        gap=gap,
    )


def walk_forward_splits(
    n_samples: int,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    gap: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward cross-validation splits.

    Properly handles temporal ordering: each split uses all past
    data for training and future data for testing, expanding the
    training window over time.

    Args:
        n_samples: Total number of samples.
        n_splits: Number of cross-validation splits.
        train_ratio: Fraction of each split's data for training.
        gap: Gap between train and test sets.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    test_size = n_samples // (n_splits + 1)
    if test_size < 2:
        return [(np.arange(n_samples - 1), np.array([n_samples - 1]))]

    splits = []
    for i in range(n_splits):
        test_end = n_samples - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - gap

        if train_end <= 0:
            continue

        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, test_end)
        splits.append((train_indices, test_indices))

    return splits


# ============================================================================
# Sliding Window Dataset Generator
# ============================================================================


class SlidingWindowDataset:
    """Generate sliding window samples from time series data.

    Creates overlapping windows from a time series for supervised
    learning, where each window becomes an input sequence and
    the subsequent values become targets.

    Args:
        data: Input time series array of shape (n_timesteps, n_features).
        window_size: Size of the input window.
        horizon: Prediction horizon (number of future steps).
        stride: Step size between consecutive windows.
        target_column: Column index for target (0 = first column).
    """

    def __init__(
        self,
        data: np.ndarray,
        window_size: int = 60,
        horizon: int = 1,
        stride: int = 1,
        target_column: int = 0,
    ) -> None:
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.stride = stride
        self.target_column = target_column

        self._n_samples = max(
            0, (len(data) - window_size - horizon + 1) // stride
        )

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single window sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (window, target) arrays.
        """
        start = idx * self.stride
        end = start + self.window_size

        window = self.data[start:end]
        target = self.data[end:end + self.horizon, self.target_column]

        return window.astype(np.float32), target.astype(np.float32)

    def get_all_windows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate all windows at once.

        Returns:
            Tuple of (windows, targets) arrays.
            windows: shape (n_samples, window_size, n_features)
            targets: shape (n_samples, horizon)
        """
        if self._n_samples == 0:
            return np.array([]), np.array([])

        windows = np.empty(
            (self._n_samples, self.window_size, self.data.shape[1]),
            dtype=np.float32,
        )
        targets = np.empty(
            (self._n_samples, self.horizon),
            dtype=np.float32,
        )

        for i in range(self._n_samples):
            w, t = self[i]
            windows[i] = w
            targets[i] = t

        return windows, targets

    def iter_batches(
        self,
        batch_size: int = 64,
        shuffle: bool = False,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Iterate over mini-batches of windows.

        Args:
            batch_size: Mini-batch size.
            shuffle: Whether to shuffle sample order.

        Yields:
            Tuple of (batch_windows, batch_targets) arrays.
        """
        indices = np.arange(self._n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, self._n_samples, batch_size):
            batch_indices = indices[start:start + batch_size]
            windows = np.empty(
                (len(batch_indices), self.window_size, self.data.shape[1]),
                dtype=np.float32,
            )
            targets = np.empty(
                (len(batch_indices), self.horizon),
                dtype=np.float32,
            )
            for j, idx in enumerate(batch_indices):
                windows[j], targets[j] = self[idx]
            yield windows, targets


# ============================================================================
# Time Series Data Augmentation
# ============================================================================


class TimeSeriesAugmentor:
    """Data augmentation for time series.

    Provides several augmentation strategies suitable for
    financial time series data.

    Strategies:
    - jitter: Add Gaussian noise
    - scaling: Multiply by a random scale factor
    - masking: Randomly mask out time steps
    - time_warp: Non-linear time warping
    - window_slice: Random sub-window extraction
    - magnitude_warp: Smooth magnitude perturbation
    """

    def __init__(
        self,
        jitter_sigma: float = 0.03,
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        masking_ratio: float = 0.05,
        time_warp_sigma: float = 0.2,
        magnitude_warp_sigma: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.jitter_sigma = jitter_sigma
        self.scaling_range = scaling_range
        self.masking_ratio = masking_ratio
        self.time_warp_sigma = time_warp_sigma
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self._rng = np.random.RandomState(seed)

    def jitter(self, X: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the time series.

        Args:
            X: Input array of shape (seq_len, n_features) or (seq_len,).

        Returns:
            Augmented array with noise added.
        """
        noise = self._rng.normal(0, self.jitter_sigma, size=X.shape).astype(X.dtype)
        return X + noise

    def scaling(self, X: np.ndarray) -> np.ndarray:
        """Scale the time series by a random factor.

        Args:
            X: Input array.

        Returns:
            Scaled array.
        """
        scale = self._rng.uniform(*self.scaling_range)
        return X * scale

    def masking(self, X: np.ndarray, mask_value: float = 0.0) -> np.ndarray:
        """Randomly mask out time steps.

        Args:
            X: Input array of shape (seq_len, n_features).
            mask_value: Value to use for masked positions.

        Returns:
            Array with some time steps masked.
        """
        result = X.copy()
        if X.ndim == 1:
            mask = self._rng.random(len(X)) < self.masking_ratio
            result[mask] = mask_value
        else:
            mask = self._rng.random(X.shape[0]) < self.masking_ratio
            result[mask, :] = mask_value
        return result

    def magnitude_warp(self, X: np.ndarray, n_knots: int = 4) -> np.ndarray:
        """Apply smooth magnitude warping.

        Multiplies the time series by a smoothly varying random curve,
        generated by interpolating random knots.

        Args:
            X: Input array of shape (seq_len, n_features).
            n_knots: Number of control points for the warping curve.

        Returns:
            Magnitude-warped array.
        """
        seq_len = X.shape[0]
        knot_xs = np.linspace(0, seq_len - 1, n_knots)
        knot_ys = self._rng.normal(1.0, self.magnitude_warp_sigma, size=n_knots)

        from scipy.interpolate import CubicSpline
        cs = CubicSpline(knot_xs, knot_ys)
        warp_curve = cs(np.arange(seq_len))

        if X.ndim == 1:
            return X * warp_curve
        return X * warp_curve[:, np.newaxis]

    def time_warp(self, X: np.ndarray, n_knots: int = 4) -> np.ndarray:
        """Apply time warping (non-uniform time scaling).

        Stretches and compresses different parts of the time series
        using a smooth warping function.

        Args:
            X: Input array of shape (seq_len, n_features).
            n_knots: Number of control points.

        Returns:
            Time-warped array of the same shape.
        """
        seq_len = X.shape[0]
        orig_steps = np.arange(seq_len)

        # Random warping: perturb time indices
        knot_xs = np.linspace(0, seq_len - 1, n_knots)
        knot_ys = self._rng.normal(0, self.time_warp_sigma, size=n_knots)
        knot_ys[0] = 0
        knot_ys[-1] = 0  # Fix endpoints

        from scipy.interpolate import CubicSpline
        cs = CubicSpline(knot_xs, knot_ys)
        warp = cs(orig_steps)

        # New time indices
        new_indices = np.clip(orig_steps + warp * seq_len, 0, seq_len - 1).astype(int)

        if X.ndim == 1:
            return X[new_indices]
        return X[new_indices, :]

    def window_slice(self, X: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:
        """Extract a random contiguous sub-window and resize to original length.

        Args:
            X: Input array of shape (seq_len, n_features).
            reduce_ratio: Ratio of sub-window length to original.

        Returns:
            Sub-window resized to original length.
        """
        seq_len = X.shape[0]
        sub_len = int(seq_len * reduce_ratio)
        start = self._rng.randint(0, seq_len - sub_len + 1)

        if X.ndim == 1:
            sub_window = X[start:start + sub_len]
            # Resize to original length using linear interpolation
            indices = np.linspace(0, sub_len - 1, seq_len)
            return np.interp(indices, np.arange(sub_len), sub_window).astype(X.dtype)
        else:
            sub_window = X[start:start + sub_len, :]
            result = np.empty_like(X)
            indices = np.linspace(0, sub_len - 1, seq_len)
            for j in range(X.shape[1]):
                result[:, j] = np.interp(
                    indices, np.arange(sub_len), sub_window[:, j]
                ).astype(X.dtype)
            return result

    def augment(
        self,
        X: np.ndarray,
        methods: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Apply a sequence of augmentations.

        Args:
            X: Input array.
            methods: List of augmentation method names to apply.
                If None, applies jitter + scaling.

        Returns:
            Augmented array.
        """
        if methods is None:
            methods = ["jitter", "scaling"]

        result = X.copy()
        for method_name in methods:
            method = getattr(self, method_name, None)
            if method is not None:
                result = method(result)
            else:
                logger.warning("Unknown augmentation method: %s", method_name)

        return result


# ============================================================================
# Efficient Batch Collation
# ============================================================================


def collate_batch(
    batch: List[Tuple[np.ndarray, ...]],
    pad: bool = False,
    padding_value: float = 0.0,
) -> Tuple[np.ndarray, ...]:
    """Collate a list of sample tuples into batched arrays.

    Handles variable-length sequences with optional padding.

    Args:
        batch: List of (features, target, ...) tuples.
        pad: Whether to pad variable-length sequences.
        padding_value: Value for padding.

    Returns:
        Tuple of batched arrays.
    """
    if not batch:
        return ()

    n_fields = len(batch[0])
    result = []

    for field_idx in range(n_fields):
        arrays = [sample[field_idx] for sample in batch]

        if arrays[0] is None:
            result.append(None)
            continue

        if pad and any(a.shape != arrays[0].shape for a in arrays):
            # Pad to max length in batch
            padded, _ = pad_sequences(arrays, padding_value=padding_value)
            result.append(padded)
        else:
            try:
                result.append(np.stack(arrays, axis=0))
            except ValueError as e:
                logger.warning(
                    "Could not stack arrays: %s. Falling back to padding.", e
                )
                padded, _ = pad_sequences(arrays, padding_value=padding_value)
                result.append(padded)

    return tuple(result)


def collate_to_torch(
    batch: List[Tuple[np.ndarray, ...]],
    pad: bool = False,
    padding_value: float = 0.0,
    device: Optional[str] = None,
) -> Tuple[Any, ...]:
    """Collate and convert to PyTorch tensors.

    Args:
        batch: List of sample tuples.
        pad: Whether to pad variable-length sequences.
        padding_value: Value for padding.
        device: Target device for tensors.

    Returns:
        Tuple of PyTorch tensors.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not _check_torch():
        raise ImportError("PyTorch is required for collate_to_torch()")

    torch = _import_torch()
    numpy_batch = collate_batch(batch, pad=pad, padding_value=padding_value)

    tensors = []
    for arr in numpy_batch:
        if arr is None:
            tensors.append(None)
        elif isinstance(arr, np.ndarray):
            tensor = torch.from_numpy(arr)
            if device is not None:
                tensor = tensor.to(device)
            tensors.append(tensor)
        else:
            tensors.append(arr)

    return tuple(tensors)
