"""
Temporal Fusion Transformer (TFT) for Multi-Horizon Crypto Price Forecasting
=============================================================================

Implements the TFT architecture (Lim et al., 2021) adapted for cryptocurrency
markets with support for:
- Variable selection across static, encoder, and decoder inputs
- Gated residual networks for skip connections and non-linear processing
- Interpretable multi-head attention with time-aware masking
- Quantile forecasts for uncertainty estimation
- Multi-horizon prediction (1min, 5min, 15min, 1h, 4h, 1d)
- GPU training with mixed precision (AMP)
- Attention weight extraction for model interpretability

Typical usage:
    >>> config = TFTConfig()
    >>> model = TemporalFusionTransformer(config)
    >>> output = model(batch)  # batch from TFTDataPreprocessor
    >>> print(output.quantiles.shape)  # (batch, forecast_horizon, num_quantiles)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TFTConfig:
    """Hyperparameter configuration for the Temporal Fusion Transformer.

    Attributes:
        static_input_dim: Number of static covariate features (e.g. asset ID embeddings).
        encoder_input_dim: Number of encoder-side temporal features (known past).
        decoder_input_dim: Number of decoder-side temporal features (known future).
        hidden_dim: Internal hidden dimension used throughout the model.
        num_heads: Number of attention heads in the interpretable multi-head attention.
        lstm_layers: Number of LSTM layers in the sequential encoder.
        lstm_dropout: Dropout probability between LSTM layers.
        num_grn_layers: Number of GRN residual layers per variable selection.
        output_dim: Number of output targets (e.g. 1 for log-return, 2 for price+volume).
        forecast_horizons: List of forecast horizons in timesteps.
        quantiles: Quantile levels for probabilistic forecasting.
        dropout: General dropout rate.
        max_encoder_length: Maximum lookback window length.
        max_decoder_length: Maximum forecast window length.
        use_mixed_precision: Whether to use AMP for GPU training.
        attention_dropout: Dropout within the attention mechanism.
    """

    static_input_dim: int = 16
    encoder_input_dim: int = 64
    decoder_input_dim: int = 32
    hidden_dim: int = 128
    num_heads: int = 4
    lstm_layers: int = 2
    lstm_dropout: float = 0.1
    num_grn_layers: int = 2
    output_dim: int = 1
    forecast_horizons: List[int] = field(
        default_factory=lambda: [1, 5, 15, 60, 240, 1440]
    )
    quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9]
    )
    dropout: float = 0.1
    max_encoder_length: int = 240
    max_decoder_length: int = 60
    use_mixed_precision: bool = True
    attention_dropout: float = 0.1

    @property
    def num_quantiles(self) -> int:
        """Return the number of quantile levels."""
        return len(self.quantiles)

    @property
    def horizon_dim(self) -> int:
        """Return the effective forecast dimension (horizons × output_dim)."""
        return len(self.forecast_horizons) * self.output_dim


# ---------------------------------------------------------------------------
# Gated Residual Network (GRN)
# ---------------------------------------------------------------------------

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network – core building block of the TFT.

    GRN(x) = LayerNorm(η₁ + GLU(W₂ · ELU(W₁ · LayerNorm(x))))
    where η₁ is a skip connection and GLU applies a gating mechanism.

    Optionally accepts a context vector that is concatenated to the input
    before the first linear layer.

    Args:
        input_dim: Dimensionality of the input tensor.
        hidden_dim: Internal hidden dimension (default: same as input_dim).
        output_dim: Dimensionality of the output tensor.
        context_dim: Optional context vector dimension (0 = no context).
        dropout: Dropout probability applied after the first linear layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim
        self.output_dim = output_dim or input_dim
        self.context_dim = context_dim

        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.fc1 = nn.Linear(
            self.input_dim + self.context_dim, self.hidden_dim
        )
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim * 2)  # *2 for GLU
        self.gate_fc = nn.Linear(self.output_dim * 2, self.output_dim)
        self.skip_linear: Optional[nn.Linear] = None
        if self.input_dim != self.output_dim:
            self.skip_linear = nn.Linear(self.input_dim, self.output_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        for module in [self.fc1, self.fc2, self.gate_fc]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        if self.skip_linear is not None:
            nn.init.xavier_uniform_(self.skip_linear.weight)
            nn.init.zeros_(self.skip_linear.bias)

    def forward(
        self, x: Tensor, context: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass through the GRN.

        Args:
            x: Input tensor of shape (..., input_dim).
            context: Optional context tensor of shape (..., context_dim).

        Returns:
            Output tensor of shape (..., output_dim).
        """
        residual = x
        x = self.layer_norm(x)

        if context is not None:
            x = torch.cat([x, context], dim=-1)

        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # GLU gating: split into value and gate, then multiply
        value, gate = x.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        x = value * gate

        # Skip connection
        if self.skip_linear is not None:
            residual = self.skip_linear(residual)
        x = x + residual
        return x


# ---------------------------------------------------------------------------
# Variable Selection Network
# ---------------------------------------------------------------------------

class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network that learns feature importance per timestep.

    For each input group (static, encoder, decoder) the VSN:
    1. Flattens all features into a single vector.
    2. Passes through a GRN (with optional context) to produce sparse weights.
    3. Applies softmax to produce selection probabilities.
    4. Returns the weighted sum of features along with the weights.

    Args:
        input_dim: Dimensionality of each individual feature.
        num_features: Number of input features in this group.
        hidden_dim: Internal hidden dimension.
        context_dim: Optional context vector dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        hidden_dim: int,
        context_dim: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.flattened_dim = input_dim * num_features
        self.grn = GatedResidualNetwork(
            input_dim=self.flattened_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            context_dim=context_dim,
            dropout=dropout,
        )
        self.feature_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_features)
            ]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, inputs: Tensor, context: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Select and weight features.

        Args:
            inputs: Tensor of shape (..., num_features, input_dim).
            context: Optional context of shape (..., context_dim).

        Returns:
            A tuple of:
              - weighted_features: shape (..., hidden_dim)
              - weights: shape (..., num_features) – feature importance
        """
        # Flatten features for the selection GRN
        flat = inputs.reshape(*inputs.shape[:-2], self.flattened_dim)
        weights = self.grn(flat, context=context)
        weights = self.softmax(weights)

        # Process each feature individually, then stack
        processed_features = torch.stack(
            [grn(inputs[..., i, :]) for i, grn in enumerate(self.feature_grns)],
            dim=-2,
        )

        # Weighted sum across features
        weighted = torch.sum(
            processed_features * weights.unsqueeze(-1), dim=-2
        )
        return weighted, weights


# ---------------------------------------------------------------------------
# Interpretable Multi-Head Attention
# ---------------------------------------------------------------------------

class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention with time-aware masking.

    Unlike standard multi-head attention, each head shares the same value
    projection matrix so that attention weights can be averaged across heads
    for a single interpretable attention map.

    Supports causal masking so the decoder cannot attend to future timesteps.

    Args:
        hidden_dim: Total hidden dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        attention_dropout: Dropout probability on attention weights.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Shared value projection for interpretability
        self.v_proj = nn.Linear(hidden_dim, self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.head_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(attention_dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Compute interpretable multi-head attention.

        Args:
            query: (batch, seq_q, hidden_dim)
            key:   (batch, seq_k, hidden_dim)
            value: (batch, seq_k, hidden_dim)
            mask:  Optional (batch, seq_q, seq_k) boolean mask (True = attend).

        Returns:
            A tuple of:
              - output: (batch, seq_q, hidden_dim)
              - attention: (batch, seq_q, seq_k) – averaged across heads
        """
        B, S_q, _ = query.shape
        S_k = key.shape[1]

        Q = self.q_proj(query)  # (B, S_q, hidden)
        K = self.k_proj(key)    # (B, S_k, hidden)
        V = self.v_proj(value)  # (B, S_k, head_dim)  -- shared!

        # Reshape Q, K for multi-head: (B, num_heads, S, head_dim)
        Q = Q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, S_q, S_k)

        if mask is not None:
            # Expand mask for heads
            scores = scores.masked_fill(
                mask.unsqueeze(1) == False,  # noqa: E712
                float("-inf"),
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Average attention weights across heads for interpretability
        avg_attention = attn_weights.mean(dim=1)  # (B, S_q, S_k)

        # Weighted sum: each head attends with shared V
        # (B, H, S_q, S_k) × (B, 1, S_k, head_dim) → (B, H, S_q, head_dim)
        V_expanded = V.unsqueeze(1)  # (B, 1, S_k, head_dim)
        context = torch.matmul(attn_weights, V_expanded.expand(-1, self.num_heads, -1, -1))
        # Average across heads (shared value makes this meaningful)
        context = context.mean(dim=1)  # (B, S_q, head_dim)

        output = self.out_proj(context)  # (B, S_q, hidden_dim)
        return output, avg_attention


# ---------------------------------------------------------------------------
# Static Covariate Encoder
# ---------------------------------------------------------------------------

class StaticCovariateEncoder(nn.Module):
    """Encodes static covariates into context vectors for the TFT.

    Produces four context vectors:
      - c_s:  for variable selection (encoder)
      - c_e:  for variable selection (decoder)
      - c_c:  for LSTM cell state initialisation
      - c_h:  for LSTM hidden state initialisation

    Args:
        input_dim: Dimensionality of static input features.
        hidden_dim: Internal hidden dimension.
        num_features: Number of static features (for VSN).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_features: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            input_dim=input_dim,
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.grn_s = GatedResidualNetwork(hidden_dim, output_dim=hidden_dim, dropout=dropout)
        self.grn_e = GatedResidualNetwork(hidden_dim, output_dim=hidden_dim, dropout=dropout)
        self.grn_c = GatedResidualNetwork(hidden_dim, output_dim=hidden_dim, dropout=dropout)
        self.grn_h = GatedResidualNetwork(hidden_dim, output_dim=hidden_dim, dropout=dropout)

    def forward(
        self, static_inputs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Encode static covariates.

        Args:
            static_inputs: (batch, num_features, input_dim)

        Returns:
            Tuple of:
              - c_s: selection context for encoder  (batch, hidden_dim)
              - c_e: selection context for decoder  (batch, hidden_dim)
              - c_c: LSTM cell state context        (batch, hidden_dim)
              - c_h: LSTM hidden state context      (batch, hidden_dim)
              - static_weights: feature importance   (batch, num_features)
        """
        selected, weights = self.vsn(static_inputs)
        c_s = self.grn_s(selected)
        c_e = self.grn_e(selected)
        c_c = self.grn_c(selected)
        c_h = self.grn_h(selected)
        return c_s, c_e, c_c, c_h, weights


# ---------------------------------------------------------------------------
# Temporal Covariate Encoder
# ---------------------------------------------------------------------------

class TemporalCovariateEncoder(nn.Module):
    """Encodes temporal (time-varying) covariates via LSTM + skip connections.

    The encoder processes known past features through a bi-directional LSTM,
    while the decoder processes known future features through a uni-directional LSTM.
    Skip connections are added via GRNs.

    Args:
        encoder_input_dim: Input dimension for encoder-side features.
        decoder_input_dim: Input dimension for decoder-side features.
        hidden_dim: Internal hidden dimension.
        num_encoder_features: Number of encoder features (for VSN).
        num_decoder_features: Number of decoder features (for VSN).
        lstm_layers: Number of LSTM layers.
        lstm_dropout: Dropout between LSTM layers.
        dropout: General dropout.
    """

    def __init__(
        self,
        encoder_input_dim: int,
        decoder_input_dim: int,
        hidden_dim: int,
        num_encoder_features: int,
        num_decoder_features: int,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_vsn = VariableSelectionNetwork(
            input_dim=encoder_input_dim,
            num_features=num_encoder_features,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,  # conditioned on static context
            dropout=dropout,
        )
        self.decoder_vsn = VariableSelectionNetwork(
            input_dim=decoder_input_dim,
            num_features=num_decoder_features,
            hidden_dim=hidden_dim,
            context_dim=hidden_dim,
            dropout=dropout,
        )

        # Bidirectional LSTM for encoder
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        # Unidirectional LSTM for decoder
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )

        # Project bidirectional output back to hidden_dim
        self.encoder_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        # Gate input: concatenation of projected output and original bidirectional output
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

        self.post_lstm_grn = GatedResidualNetwork(
            hidden_dim, output_dim=hidden_dim, dropout=dropout
        )

    def forward(
        self,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
        c_s: Tensor,
        c_e: Tensor,
        c_c: Optional[Tensor] = None,
        c_h: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Encode temporal covariates.

        Args:
            encoder_inputs: (batch, enc_len, num_enc_features, enc_input_dim)
            decoder_inputs: (batch, dec_len, num_dec_features, dec_input_dim)
            c_s: Static context for encoder variable selection (batch, hidden_dim)
            c_e: Static context for decoder variable selection (batch, hidden_dim)
            c_c: LSTM cell state initialisation (batch, hidden_dim) or None
            c_h: LSTM hidden state initialisation (batch, hidden_dim) or None

        Returns:
            Tuple of:
              - encoder_output: (batch, enc_len, hidden_dim)
              - decoder_output: (batch, dec_len, hidden_dim)
              - encoder_weights: (batch, enc_len, num_enc_features)
              - decoder_weights: (batch, dec_len, num_dec_features)
        """
        B = encoder_inputs.shape[0]
        enc_len = encoder_inputs.shape[1]
        dec_len = decoder_inputs.shape[1]

        # Variable selection with static context
        # Expand context for each timestep
        c_s_exp = c_s.unsqueeze(1).expand(-1, enc_len, -1)
        c_e_exp = c_e.unsqueeze(1).expand(-1, dec_len, -1)

        # Flatten for VSN: (B, T, num_features, input_dim)
        enc_selected_list = []
        enc_weights_list = []
        for t in range(enc_len):
            sel, w = self.encoder_vsn(encoder_inputs[:, t], context=c_s)
            enc_selected_list.append(sel)
            enc_weights_list.append(w)
        enc_selected = torch.stack(enc_selected_list, dim=1)  # (B, enc_len, H)
        enc_weights = torch.stack(enc_weights_list, dim=1)     # (B, enc_len, num_enc_feat)

        dec_selected_list = []
        dec_weights_list = []
        for t in range(dec_len):
            sel, w = self.decoder_vsn(decoder_inputs[:, t], context=c_e)
            dec_selected_list.append(sel)
            dec_weights_list.append(w)
        dec_selected = torch.stack(dec_selected_list, dim=1)
        dec_weights = torch.stack(dec_weights_list, dim=1)

        # LSTM encoding
        # Bidirectional LSTM expects hidden shape: (num_layers * 2, batch, hidden)
        num_directions = 2  # bidirectional
        lstm_state: Optional[Tuple[Tensor, Tensor]] = None
        if c_h is not None and c_c is not None:
            # Repeat for (num_layers * num_directions)
            h0 = c_h.unsqueeze(0).repeat(
                self.lstm_encoder.num_layers * num_directions, 1, 1
            )
            c0 = c_c.unsqueeze(0).repeat(
                self.lstm_encoder.num_layers * num_directions, 1, 1
            )
            lstm_state = (h0, c0)

        lstm_enc_out, (h_n, c_n) = self.lstm_encoder(
            enc_selected, lstm_state
        )

        # Project bidirectional output back to hidden_dim with gating
        proj = self.encoder_proj(lstm_enc_out)
        gate_input = torch.cat([proj, lstm_enc_out], dim=-1)
        gate_values = torch.sigmoid(self.gate(gate_input))
        encoder_output = self.post_lstm_grn(proj * gate_values)

        # Decoder LSTM (unidirectional), initialised from encoder final state
        # Take only forward direction states
        h_n_dec = h_n[::2]  # every other layer is forward direction
        c_n_dec = c_n[::2]
        lstm_dec_out, _ = self.lstm_decoder(
            dec_selected, (h_n_dec, c_n_dec)
        )
        decoder_output = self.post_lstm_grn(lstm_dec_out)

        return encoder_output, decoder_output, enc_weights, dec_weights


# ---------------------------------------------------------------------------
# Full Temporal Fusion Transformer
# ---------------------------------------------------------------------------

class TFTOutput:
    """Container for TFT model outputs.

    Attributes:
        quantiles: Predicted quantiles (batch, forecast_len, num_quantiles).
        attention: Interpretable attention weights (batch, forecast_len, context_len).
        static_weights: Static feature importance (batch, num_static_features).
        encoder_weights: Encoder feature importance (batch, enc_len, num_enc_features).
        decoder_weights: Decoder feature importance (batch, dec_len, num_dec_features).
    """

    __slots__ = [
        "quantiles",
        "attention",
        "static_weights",
        "encoder_weights",
        "decoder_weights",
    ]

    def __init__(
        self,
        quantiles: Tensor,
        attention: Tensor,
        static_weights: Tensor,
        encoder_weights: Tensor,
        decoder_weights: Tensor,
    ) -> None:
        self.quantiles = quantiles
        self.attention = attention
        self.static_weights = static_weights
        self.encoder_weights = encoder_weights
        self.decoder_weights = decoder_weights


class TemporalFusionTransformer(nn.Module):
    """Full Temporal Fusion Transformer for multi-horizon crypto forecasting.

    Architecture summary:
        1. Static covariates → StaticCovariateEncoder → context vectors
        2. Temporal covariates → VariableSelection + LSTM → sequence encodings
        3. Skip connections via gating
        4. Interpretable multi-head attention over encoder outputs
        5. Position-wise feed-forward via GRN
        6. Quantile projection layer

    Args:
        config: TFTConfig hyperparameter object.
    """

    def __init__(self, config: TFTConfig) -> None:
        super().__init__()
        self.config = config
        self.device_ = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Static covariate encoder
        self.static_encoder = StaticCovariateEncoder(
            input_dim=config.static_input_dim,
            hidden_dim=config.hidden_dim,
            num_features=config.static_input_dim,  # one feature per static dim
            dropout=config.dropout,
        )

        # Temporal covariate encoder
        self.temporal_encoder = TemporalCovariateEncoder(
            encoder_input_dim=config.encoder_input_dim,
            decoder_input_dim=config.decoder_input_dim,
            hidden_dim=config.hidden_dim,
            num_encoder_features=config.encoder_input_dim,
            num_decoder_features=config.decoder_input_dim,
            lstm_layers=config.lstm_layers,
            lstm_dropout=config.lstm_dropout,
            dropout=config.dropout,
        )

        # Interpretable multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            attention_dropout=config.attention_dropout,
        )

        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(
            config.hidden_dim, output_dim=config.hidden_dim, dropout=config.dropout
        )
        self.post_attn_gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # Position-wise feed-forward
        self.pos_wise_ff = GatedResidualNetwork(
            config.hidden_dim, output_dim=config.hidden_dim, dropout=config.dropout
        )

        # Quantile output head
        self.quantile_proj = nn.Linear(
            config.hidden_dim,
            config.output_dim * len(config.forecast_horizons) * config.num_quantiles,
        )

        # Layer norm for final output
        self.output_layer_norm = nn.LayerNorm(config.hidden_dim)

        # Move to device
        self.to(self.device_)

    # ------------------------------------------------------------------
    # Causal mask
    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
        """Create a causal (lower-triangular) mask.

        Returns:
            Boolean tensor of shape (1, seq_len, seq_len) where True = attend.
        """
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        )
        return mask.unsqueeze(0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        static_inputs: Tensor,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
    ) -> TFTOutput:
        """Full forward pass of the TFT.

        Args:
            static_inputs:  (batch, num_static_features, static_input_dim)
            encoder_inputs: (batch, enc_len, num_enc_features, enc_input_dim)
            decoder_inputs: (batch, dec_len, num_dec_features, dec_input_dim)

        Returns:
            TFTOutput containing quantile predictions and interpretability data.
        """
        cfg = self.config
        B = static_inputs.shape[0]
        enc_len = encoder_inputs.shape[1]
        dec_len = decoder_inputs.shape[1]

        # 1. Static covariate encoding
        c_s, c_e, c_c, c_h, static_weights = self.static_encoder(static_inputs)

        # 2. Temporal covariate encoding
        enc_out, dec_out, enc_weights, dec_weights = self.temporal_encoder(
            encoder_inputs, decoder_inputs, c_s, c_e, c_c, c_h
        )

        # 3. Combine encoder + decoder for attention input
        # Context = encoder output, Query = decoder output
        context = enc_out   # (B, enc_len, H)
        query = dec_out     # (B, dec_len, H)

        # 4. Self-attention with causal mask
        # Create mask: decoder can attend to all encoder positions + causal on decoder
        full_len = enc_len + dec_len
        causal = self._causal_mask(full_len, self.device_)
        # For cross-attention: decoder can attend to all encoder steps
        cross_mask = torch.ones(
            1, dec_len, enc_len, device=self.device_, dtype=torch.bool
        )
        mask = cross_mask  # simplified: full visibility over encoder

        attn_output, attn_weights = self.attention(
            query, context, context, mask=mask
        )

        # 5. Skip connection via gating
        gate_input = torch.cat([attn_output, query], dim=-1)
        gate = torch.sigmoid(self.post_attn_gate(gate_input))
        gated = self.post_attn_grn(attn_output) * gate + query * (1 - gate)

        # 6. Position-wise feed-forward
        transformed = self.pos_wise_ff(gated)
        transformed = self.output_layer_norm(transformed)

        # 7. Quantile projection
        quantile_logits = self.quantile_proj(transformed)
        # Reshape: (B, dec_len, output_dim * num_horizons * num_quantiles)
        quantile_logits = quantile_logits.view(
            B,
            dec_len,
            cfg.output_dim,
            len(cfg.forecast_horizons),
            cfg.num_quantiles,
        )
        # Select only the desired forecast horizons from the decoder sequence
        # For simplicity, we take the first step's multi-horizon prediction
        quantiles = quantile_logits[:, 0, :, :, :]  # (B, output_dim, num_horizons, num_quantiles)
        quantiles = quantiles.permute(0, 2, 1, 3)    # (B, num_horizons, output_dim, num_quantiles)
        quantiles = quantiles.reshape(
            B, len(cfg.forecast_horizons) * cfg.output_dim, cfg.num_quantiles
        )

        return TFTOutput(
            quantiles=quantiles,
            attention=attn_weights,
            static_weights=static_weights,
            encoder_weights=enc_weights,
            decoder_weights=dec_weights,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        predictions: TFTOutput,
        targets: Tensor,
    ) -> Tensor:
        """Compute quantile loss (pinball loss) across all quantiles.

        Args:
            predictions: TFTOutput from forward pass.
            targets: Ground-truth targets of shape (batch, horizon_dim).

        Returns:
            Scalar loss tensor.
        """
        quantiles = predictions.quantiles  # (B, horizon_dim, num_quantiles)
        cfg = self.config
        q_values = torch.tensor(
            cfg.quantiles, device=quantiles.device, dtype=quantiles.dtype
        ).view(1, 1, -1)

        errors = targets.unsqueeze(-1) - quantiles  # (B, H, Q)
        loss = torch.max(
            q_values * errors,
            (q_values - 1) * errors,
        )
        return loss.mean()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(
        self,
        static_inputs: Tensor,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
        targets: Tensor,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> Dict[str, float]:
        """Execute a single training step with optional mixed precision.

        Args:
            static_inputs:  (batch, num_static, static_dim)
            encoder_inputs: (batch, enc_len, num_enc, enc_dim)
            decoder_inputs: (batch, dec_len, num_dec, dec_dim)
            targets:        (batch, horizon_dim)
            optimizer:      PyTorch optimiser.
            scaler:         Optional GradScaler for AMP.

        Returns:
            Dict with loss value and any auxiliary metrics.
        """
        self.train()
        use_amp = (
            self.config.use_mixed_precision
            and self.device_.type == "cuda"
            and scaler is not None
        )

        if use_amp:
            with torch.amp.autocast("cuda"):
                output = self(static_inputs, encoder_inputs, decoder_inputs)
                loss = self.compute_loss(output, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = self(static_inputs, encoder_inputs, decoder_inputs)
            loss = self.compute_loss(output, targets)
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        return {
            "loss": loss.item(),
            "quantile_mean": output.quantiles.mean().item(),
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        static_inputs: Tensor,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
    ) -> Dict[str, Tensor]:
        """Produce quantile forecasts and interpretability outputs.

        Args:
            static_inputs:  (batch, num_static, static_dim)
            encoder_inputs: (batch, enc_len, num_enc, enc_dim)
            decoder_inputs: (batch, dec_len, num_dec, dec_dim)

        Returns:
            Dictionary with keys:
              - quantiles: (batch, num_horizons, num_quantiles)
              - median:    (batch, num_horizons) – 50th percentile forecast
              - attention: (batch, dec_len, enc_len)
              - static_importance:  (batch, num_static)
              - encoder_importance: (batch, enc_len, num_enc)
              - decoder_importance: (batch, dec_len, num_dec)
        """
        self.eval()
        output = self(static_inputs, encoder_inputs, decoder_inputs)
        cfg = self.config

        # Extract median (0.5 quantile)
        median_idx = cfg.quantiles.index(0.5) if 0.5 in cfg.quantiles else cfg.num_quantiles // 2
        median = output.quantiles[:, :, median_idx]

        # Reshape quantiles to (B, num_horizons, output_dim, num_quantiles)
        quantiles = output.quantiles.view(
            -1, len(cfg.forecast_horizons), cfg.output_dim, cfg.num_quantiles
        )

        return {
            "quantiles": quantiles,
            "median": median,
            "attention": output.attention,
            "static_importance": output.static_weights,
            "encoder_importance": output.encoder_weights,
            "decoder_importance": output.decoder_weights,
        }

    # ------------------------------------------------------------------
    # Attention extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_attention(
        self,
        static_inputs: Tensor,
        encoder_inputs: Tensor,
        decoder_inputs: Tensor,
    ) -> Dict[str, Tensor]:
        """Extract attention weights for model interpretability.

        Returns a dict of attention maps that can be visualised to understand
        which past timesteps the model focuses on for each forecast step.

        Args:
            static_inputs:  (batch, num_static, static_dim)
            encoder_inputs: (batch, enc_len, num_enc, enc_dim)
            decoder_inputs: (batch, dec_len, num_dec, dec_dim)

        Returns:
            Dictionary with:
              - temporal_attention: (batch, dec_len, enc_len)
              - static_weights:     (batch, num_static)
              - encoder_selection:  (batch, enc_len, num_enc)
              - decoder_selection:  (batch, dec_len, num_dec)
        """
        self.eval()
        output = self(static_inputs, encoder_inputs, decoder_inputs)
        return {
            "temporal_attention": output.attention,
            "static_weights": output.static_weights,
            "encoder_selection": output.encoder_weights,
            "decoder_selection": output.decoder_weights,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_device(self) -> torch.device:
        """Return the device the model is on."""
        return next(self.parameters()).device
