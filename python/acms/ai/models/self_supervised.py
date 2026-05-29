"""
Self-Supervised Pretraining for Market Representations
========================================================

Implements self-supervised learning methods for pretraining on unlabeled
market data, producing rich representations that transfer to downstream tasks:

- ContrastiveLearning: SimCLR-style contrastive learning for market data
- MaskedAutoEncoder: MAE-style masked reconstruction for temporal data
- TemporalContrastiveLoss: Contrastive loss with temporal awareness
- MarketDataAugmenter: Domain-specific data augmentation for crypto
- SelfSupervisedPretrainer: Unified pretraining pipeline

All models support GPU training with graceful CPU fallback.

Typical usage:
    >>> encoder = nn.LSTM(64, 128, batch_first=True)
    >>> cl = ContrastiveLearning(encoder, projection_dim=64)
    >>> loss = cl.train_step(batch_data)
    >>> features = cl.encode(batch_data)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Market Data Augmenter
# ---------------------------------------------------------------------------

class MarketDataAugmenter:
    """Domain-specific data augmentation for cryptocurrency market data.

    Generates positive pairs for contrastive learning by applying
    financially meaningful transformations:
    - Gaussian noise injection (simulates microstructure noise)
    - Time masking (simulates missing data)
    - Feature masking (simulates feature unavailability)
    - Temporal jittering (small shifts in time alignment)
    - Magnitude scaling (simulates different market magnitudes)
    - Mixup augmentation (combines two samples)

    Args:
        noise_std: Standard deviation of Gaussian noise.
        time_mask_ratio: Fraction of timesteps to mask.
        feature_mask_ratio: Fraction of features to mask.
        jitter_range: Max temporal shift in timesteps.
        scale_range: (min, max) scaling factor range.
        mixup_alpha: Dirichlet concentration for mixup.
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        time_mask_ratio: float = 0.1,
        feature_mask_ratio: float = 0.1,
        jitter_range: int = 3,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        mixup_alpha: float = 0.2,
    ) -> None:
        self.noise_std = noise_std
        self.time_mask_ratio = time_mask_ratio
        self.feature_mask_ratio = feature_mask_ratio
        self.jitter_range = jitter_range
        self.scale_range = scale_range
        self.mixup_alpha = mixup_alpha

    def augment(self, x: Tensor) -> Tensor:
        """Apply a random combination of augmentations.

        Args:
            x: Input tensor of shape (batch, seq_len, feature_dim).

        Returns:
            Augmented tensor of the same shape.
        """
        x_aug = x.clone()

        # 1. Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise

        # 2. Magnitude scaling
        scale = torch.empty(1, device=x.device).uniform_(*self.scale_range)
        x_aug = x_aug * scale

        # 3. Time masking
        if self.time_mask_ratio > 0:
            seq_len = x.shape[1]
            mask_len = max(1, int(seq_len * self.time_mask_ratio))
            mask_start = torch.randint(0, seq_len - mask_len + 1, (1,)).item()
            x_aug[:, mask_start : mask_start + mask_len, :] = 0

        # 4. Feature masking
        if self.feature_mask_ratio > 0:
            feature_dim = x.shape[2]
            mask = torch.rand(feature_dim, device=x.device) < self.feature_mask_ratio
            x_aug[:, :, mask] = 0

        return x_aug

    def augment_pair(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate a positive pair (two different augmentations of x).

        Args:
            x: Input tensor (batch, seq_len, feature_dim).

        Returns:
            Tuple of (x_aug1, x_aug2).
        """
        return self.augment(x), self.augment(x)

    def temporal_jitter(self, x: Tensor) -> Tensor:
        """Apply temporal jittering by shifting the sequence.

        Args:
            x: (batch, seq_len, feature_dim)

        Returns:
            Jittered tensor.
        """
        shift = torch.randint(
            -self.jitter_range, self.jitter_range + 1, (1,)
        ).item()
        if shift == 0:
            return x
        x_shifted = torch.roll(x, shifts=shift, dims=1)
        # Zero out the wrapped portion
        if shift > 0:
            x_shifted[:, :shift, :] = 0
        else:
            x_shifted[:, shift:, :] = 0
        return x_shifted

    def mixup(
        self, x1: Tensor, x2: Tensor
    ) -> Tensor:
        """Apply mixup between two samples.

        Args:
            x1: First sample (batch, seq_len, feature_dim).
            x2: Second sample (batch, seq_len, feature_dim).

        Returns:
            Mixed sample.
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        return lam * x1 + (1 - lam) * x2


# ---------------------------------------------------------------------------
# Temporal Contrastive Loss
# ---------------------------------------------------------------------------

class TemporalContrastiveLoss(nn.Module):
    """Contrastive loss with temporal awareness for market data.

    Extends NT-Xent loss with:
    - Temperature scaling
    - Temporal hardness weighting (closer-in-time negatives are harder)
    - Optional supervised contrastive mode

    Args:
        temperature: Temperature parameter for softmax.
        temporal_weight: Weight for temporal hardness (0 = uniform).
        max_time_diff: Maximum time difference for hardness weighting.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        temporal_weight: float = 0.1,
        max_time_diff: float = 86400.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.temporal_weight = temporal_weight
        self.max_time_diff = max_time_diff

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        timestamps: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute temporal contrastive loss.

        Args:
            z1: Projected representations from view 1 (batch, proj_dim).
            z2: Projected representations from view 2 (batch, proj_dim).
            timestamps: Optional (batch,) timestamps for temporal weighting.

        Returns:
            Scalar loss.
        """
        batch_size = z1.shape[0]

        # L2-normalise
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate: (2B, dim)
        z = torch.cat([z1, z2], dim=0)

        # Similarity matrix: (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature

        # Positive pairs: (i, i+B) and (i+B, i)
        positive_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=z.device)
        for i in range(batch_size):
            positive_mask[i, i + batch_size] = 1.0
            positive_mask[i + batch_size, i] = 1.0

        # Self-mask: exclude diagonal
        self_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)

        # Temporal hardness weighting
        if timestamps is not None and self.temporal_weight > 0:
            # Compute time differences between all pairs
            t1 = torch.cat([timestamps, timestamps], dim=0)
            time_diff = torch.abs(t1.unsqueeze(1) - t1.unsqueeze(0))
            # Closer in time = harder negative = higher weight
            hardness = torch.exp(-time_diff / self.max_time_diff) * self.temporal_weight
            sim = sim + hardness * (~positive_mask.bool() & self_mask).float()

        # Numerator: positive similarity
        pos_sim = (sim * positive_mask).sum(dim=-1)

        # Denominator: sum over all non-self entries
        denom = (sim * self_mask.float()).sum(dim=-1)

        # NT-Xent loss
        loss = -torch.log(pos_sim / (denom + 1e-8) + 1e-8)
        return loss.mean()


# ---------------------------------------------------------------------------
# Contrastive Learning Module
# ---------------------------------------------------------------------------

class ContrastiveLearning(nn.Module):
    """SimCLR-style contrastive learning for market data representations.

    Architecture:
        Encoder → Projection Head → Contrastive Loss

    The encoder produces representations for downstream tasks.
    The projection head is used only during pretraining.

    Args:
        encoder: Encoder module (e.g., LSTM, Transformer) that takes
                 (batch, seq_len, feature_dim) and returns (batch, hidden_dim).
        hidden_dim: Encoder output dimension.
        projection_dim: Projection head output dimension.
        temperature: NT-Xent temperature.
        device: Torch device.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device_ = device or _get_device()
        self.encoder = encoder.to(self.device_)
        self.hidden_dim = hidden_dim

        # Projection head: MLP with one hidden layer
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        ).to(self.device_)

        self.augmenter = MarketDataAugmenter()
        self.loss_fn = TemporalContrastiveLoss(temperature=temperature)

    def encode(self, x: Tensor) -> Tensor:
        """Produce representations using the encoder only.

        Handles encoders that return a single tensor or a tuple
        (e.g., LSTM which returns (output, (h_n, c_n))).

        Args:
            x: (batch, seq_len, feature_dim)

        Returns:
            Representations (batch, hidden_dim).
        """
        output = self.encoder(x)
        # Handle encoders that return tuples (e.g., LSTM)
        if isinstance(output, tuple):
            # LSTM returns (output, (h_n, c_n))
            _, hidden = output
            if isinstance(hidden, tuple):
                h_n, _ = hidden
                return h_n[-1]  # Last layer, last direction
            return hidden[-1]
        return output

    def project(self, h: Tensor) -> Tensor:
        """Project encoder output through the projection head.

        Args:
            h: Encoder output (batch, hidden_dim).

        Returns:
            Projected representations (batch, projection_dim).
        """
        return self.projection_head(h)

    def forward(
        self, x: Tensor, timestamps: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Full forward pass with augmentation and contrastive loss.

        Args:
            x: (batch, seq_len, feature_dim)
            timestamps: Optional (batch,) timestamps.

        Returns:
            Dict with 'loss', 'z1', 'z2', 'h1', 'h2'.
        """
        # Generate augmented views
        x1, x2 = self.augmenter.augment_pair(x)

        # Encode
        h1 = self.encode(x1)
        h2 = self.encode(x2)

        # Project
        z1 = self.project(h1)
        z2 = self.project(h2)

        # Contrastive loss
        loss = self.loss_fn(z1, z2, timestamps)

        return {"loss": loss, "z1": z1, "z2": z2, "h1": h1, "h2": h2}

    def train_step(
        self,
        x: Tensor,
        optimizer: torch.optim.Optimizer,
        timestamps: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            x: Input data (batch, seq_len, feature_dim).
            optimizer: PyTorch optimizer.
            timestamps: Optional timestamps for temporal weighting.

        Returns:
            Dict with loss value.
        """
        self.train()
        output = self.forward(x, timestamps)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def extract_features(self, x: Tensor) -> Tensor:
        """Extract encoder features for downstream use.

        Args:
            x: (batch, seq_len, feature_dim)

        Returns:
            Features (batch, hidden_dim).
        """
        self.eval()
        return self.encode(x)


# ---------------------------------------------------------------------------
# Masked Auto-Encoder
# ---------------------------------------------------------------------------

class MaskedAutoEncoder(nn.Module):
    """Masked Auto-Encoder for temporal market data.

    Randomly masks timesteps and trains the model to reconstruct them,
    learning rich temporal representations in the process.

    Architecture:
        Encoder (masked input) → Latent → Decoder → Reconstruct full input

    Args:
        input_dim: Feature dimension per timestep.
        hidden_dim: Latent dimension.
        num_layers: Number of encoder/decoder layers.
        mask_ratio: Fraction of timesteps to mask.
        num_heads: Attention heads (if using transformer).
        encoder_type: 'lstm' or 'transformer'.
        dropout: Dropout rate.
        device: Torch device.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        mask_ratio: float = 0.5,
        num_heads: int = 4,
        encoder_type: str = "lstm",
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device_ = device or _get_device()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mask_ratio = mask_ratio
        self.encoder_type = encoder_type

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Mask token (learnable)
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.decoder = nn.LSTM(
                hidden_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
            )
        elif encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            decoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.to(self.device_)

    def _create_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create a random boolean mask for timesteps.

        Args:
            seq_len: Sequence length.
            device: Torch device.

        Returns:
            Boolean mask of shape (seq_len,) where True = masked.
        """
        num_masked = max(1, int(seq_len * self.mask_ratio))
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        indices = torch.randperm(seq_len, device=device)[:num_masked]
        mask[indices] = True
        return mask

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Forward pass with masking and reconstruction.

        Args:
            x: (batch, seq_len, input_dim)
            mask: Optional (seq_len,) boolean mask (True = masked).
                  If None, randomly generated.

        Returns:
            Dict with 'reconstruction', 'latent', 'mask', 'loss'.
        """
        B, S, _ = x.shape

        if mask is None:
            mask = self._create_mask(S, x.device)

        # Project input
        h = self.input_proj(x)  # (B, S, hidden)

        # Replace masked positions with mask token
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1).expand_as(h)
        h = torch.where(mask_expanded, self.mask_token.unsqueeze(0).unsqueeze(0).expand_as(h), h)

        # Encode
        if self.encoder_type == "lstm":
            encoded, _ = self.encoder(h)
        else:
            encoded = self.encoder(h)
        encoded = self.layer_norm(encoded)

        # Decode
        if self.encoder_type == "lstm":
            decoded, _ = self.decoder(encoded)
        else:
            decoded = self.decoder(encoded)

        # Reconstruct
        reconstruction = self.output_proj(decoded)  # (B, S, input_dim)

        # Loss: only on masked positions
        mask_broad = mask.unsqueeze(0).unsqueeze(-1).expand_as(x)
        masked_original = x[mask_broad]
        masked_reconstruction = reconstruction[mask_broad]
        loss = F.mse_loss(masked_reconstruction, masked_original)

        return {
            "reconstruction": reconstruction,
            "latent": encoded,
            "mask": mask,
            "loss": loss,
        }

    def train_step(
        self,
        x: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            x: Input data (batch, seq_len, input_dim).
            optimizer: PyTorch optimizer.

        Returns:
            Dict with loss and reconstruction metrics.
        """
        self.train()
        output = self.forward(x)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute reconstruction quality on unmasked positions
        with torch.no_grad():
            mask = output["mask"]
            mask_broad = mask.unsqueeze(0).unsqueeze(-1).expand_as(x)
            unmasked_original = x[~mask_broad]
            unmasked_recon = output["reconstruction"][~mask_broad]
            unmasked_loss = F.mse_loss(unmasked_recon, unmasked_original)

        return {
            "masked_loss": loss.item(),
            "unmasked_loss": unmasked_loss.item(),
        }

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Extract latent representations for downstream tasks.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            Latent representation (batch, seq_len, hidden_dim).
        """
        self.eval()
        h = self.input_proj(x)
        if self.encoder_type == "lstm":
            encoded, _ = self.encoder(h)
        else:
            encoded = self.encoder(h)
        return self.layer_norm(encoded)

    @torch.no_grad()
    def reconstruct(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Reconstruct the input (for anomaly detection).

        Args:
            x: (batch, seq_len, input_dim)
            mask: Optional mask.

        Returns:
            Reconstruction (batch, seq_len, input_dim).
        """
        self.eval()
        output = self.forward(x, mask)
        return output["reconstruction"]


# ---------------------------------------------------------------------------
# Self-Supervised Pretrainer
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    """Configuration for self-supervised pretraining.

    Attributes:
        method: Pretraining method – 'contrastive', 'mae', or 'combined'.
        input_dim: Feature dimension per timestep.
        hidden_dim: Hidden dimension.
        num_layers: Encoder layers.
        projection_dim: Contrastive projection dimension.
        mask_ratio: MAE mask ratio.
        temperature: Contrastive temperature.
        learning_rate: Optimiser learning rate.
        weight_decay: L2 regularisation.
        batch_size: Batch size for training.
        num_epochs: Total pretraining epochs.
        warmup_epochs: Learning rate warmup epochs.
        device: Torch device.
    """

    method: str = "combined"
    input_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    projection_dim: int = 64
    mask_ratio: float = 0.5
    temperature: float = 0.07
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    warmup_epochs: int = 10
    device: Optional[torch.device] = None


class SelfSupervisedPretrainer:
    """Unified self-supervised pretraining pipeline.

    Supports:
    - Contrastive learning (SimCLR-style)
    - Masked auto-encoding (MAE-style)
    - Combined: alternating between both objectives

    After pretraining, the encoder can be transferred to downstream tasks
    such as price prediction, regime detection, or risk assessment.

    Args:
        config: PretrainConfig with all hyperparameters.
    """

    def __init__(self, config: Optional[PretrainConfig] = None) -> None:
        self.config = config or PretrainConfig()
        self.device = self.config.device or _get_device()

        # Build encoder backbone (shared)
        self.encoder = nn.LSTM(
            self.config.input_dim,
            self.config.hidden_dim,
            self.config.num_layers,
            batch_first=True,
        ).to(self.device)

        # Extract last hidden state from LSTM output
        self._encoder_wrapper = _LSTMEncoder(self.encoder).to(self.device)

        # Pretraining modules
        self.contrastive = ContrastiveLearning(
            encoder=self._encoder_wrapper,
            hidden_dim=self.config.hidden_dim,
            projection_dim=self.config.projection_dim,
            temperature=self.config.temperature,
            device=self.device,
        )

        self.mae = MaskedAutoEncoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            mask_ratio=self.config.mask_ratio,
            device=self.device,
        )

        self._metrics_history: List[Dict[str, float]] = []

    def pretrain(
        self,
        data_loader: Any,
        loss_fn: Optional[callable] = None,
    ) -> List[Dict[str, float]]:
        """Run the full pretraining loop.

        Args:
            data_loader: Iterable yielding (batch, seq_len, input_dim) tensors.
            loss_fn: Optional custom loss (not typically needed).

        Returns:
            List of metric dicts from each epoch.
        """
        method = self.config.method.lower()

        # Set up optimizers
        if method in ("contrastive", "combined"):
            cl_optimizer = torch.optim.AdamW(
                self.contrastive.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if method in ("mae", "combined"):
            mae_optimizer = torch.optim.AdamW(
                self.mae.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        for epoch in range(self.config.num_epochs):
            epoch_metrics: Dict[str, float] = {"epoch": epoch}

            for batch_idx, batch in enumerate(data_loader):
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device).float()

                if method == "contrastive":
                    metrics = self.contrastive.train_step(x, cl_optimizer)
                    epoch_metrics["cl_loss"] = epoch_metrics.get("cl_loss", 0) + metrics["loss"]

                elif method == "mae":
                    metrics = self.mae.train_step(x, mae_optimizer)
                    epoch_metrics["mae_loss"] = epoch_metrics.get("mae_loss", 0) + metrics["masked_loss"]

                elif method == "combined":
                    # Alternate between contrastive and MAE
                    if batch_idx % 2 == 0:
                        metrics = self.contrastive.train_step(x, cl_optimizer)
                        epoch_metrics["cl_loss"] = epoch_metrics.get("cl_loss", 0) + metrics["loss"]
                    else:
                        metrics = self.mae.train_step(x, mae_optimizer)
                        epoch_metrics["mae_loss"] = epoch_metrics.get("mae_loss", 0) + metrics["masked_loss"]

            # Average over batches
            num_batches = max(batch_idx + 1, 1)
            for key in list(epoch_metrics.keys()):
                if key != "epoch":
                    epoch_metrics[key] /= num_batches

            self._metrics_history.append(epoch_metrics)

        return self._metrics_history

    @torch.no_grad()
    def extract_features(self, x: Tensor) -> Tensor:
        """Extract pretrained features for downstream tasks.

        Uses the contrastive encoder (which shares the LSTM backbone).

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            Features (batch, hidden_dim).
        """
        self.eval_all()
        return self.contrastive.extract_features(x)

    @torch.no_grad()
    def extract_temporal_features(self, x: Tensor) -> Tensor:
        """Extract per-timestep features using the MAE encoder.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            Temporal features (batch, seq_len, hidden_dim).
        """
        self.eval_all()
        return self.mae.encode(x)

    def eval_all(self) -> None:
        """Set all sub-modules to eval mode."""
        self.contrastive.eval()
        self.mae.eval()

    def get_encoder(self) -> nn.Module:
        """Return the pretrained encoder for downstream use.

        Returns:
            The LSTM encoder module.
        """
        return self.encoder

    @property
    def metrics_history(self) -> List[Dict[str, float]]:
        """Return pretraining metrics history."""
        return self._metrics_history


class _LSTMEncoder(nn.Module):
    """Wrapper to extract the last hidden state from an LSTM.

    This makes the LSTM compatible with the ContrastiveLearning module
    which expects an encoder that returns (batch, hidden_dim).

    Args:
        lstm: LSTM module.
    """

    def __init__(self, lstm: nn.LSTM) -> None:
        super().__init__()
        self.lstm = lstm

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            Last hidden state (batch, hidden_dim).
        """
        output, (h_n, _) = self.lstm(x)
        return h_n[-1]  # Last layer, last direction
