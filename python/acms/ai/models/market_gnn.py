"""
Graph Neural Network for Crypto Market Structure Analysis
==========================================================

Implements GNN-based models for capturing cross-asset relationships,
regime detection, and contagion risk prediction:

- MarketGraph: Dynamic graph construction from correlation/cointegration
- GraphAttentionLayer: Cross-asset influence via attention mechanism
- MarketGNN: Full model for market structure encoding
- RegimeDetector: Graph-based market regime classification
- ContagionRiskPredictor: Risk propagation through the market graph

All models support GPU training/inference with graceful CPU fallback.

Typical usage:
    >>> graph = MarketGraph(n_assets=50)
    >>> graph.update_from_returns(returns_matrix)
    >>> model = MarketGNN(node_dim=32, edge_dim=8, hidden_dim=64)
    >>> node_embeddings = model(graph.get_tensors())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
# Market Graph Construction
# ---------------------------------------------------------------------------

@dataclass
class MarketGraphConfig:
    """Configuration for MarketGraph construction.

    Attributes:
        n_assets: Number of assets (nodes) in the graph.
        correlation_threshold: Minimum absolute correlation to create an edge.
        cointegration_threshold: Maximum p-value for cointegration edge.
        use_partial_corr: Whether to use partial instead of Pearson correlation.
        edge_decay: Exponential decay factor for older correlations.
        update_interval: Minimum seconds between graph rebuilds.
    """

    n_assets: int = 50
    correlation_threshold: float = 0.3
    cointegration_threshold: float = 0.05
    use_partial_corr: bool = False
    edge_decay: float = 0.99
    update_interval: float = 60.0


class MarketGraph:
    """Dynamic market graph constructed from asset returns.

    Nodes represent crypto assets; edges represent statistical relationships
    (correlation, cointegration). The graph is updated dynamically as
    correlations change over time.

    Args:
        config: MarketGraphConfig with graph parameters.
        device: Torch device for tensor storage.
    """

    def __init__(
        self,
        config: Optional[MarketGraphConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config or MarketGraphConfig()
        self.device = device or _get_device()
        self.n = self.config.n_assets

        # Node features: (n_assets, node_feature_dim)
        self.node_features: Optional[Tensor] = None
        # Edge index: (2, num_edges)
        self.edge_index: Optional[Tensor] = None
        # Edge weights / features: (num_edges, edge_feature_dim)
        self.edge_features: Optional[Tensor] = None
        # Correlation matrix
        self._corr_matrix: Optional[np.ndarray] = None
        # Previous correlation for decay
        self._prev_corr: Optional[np.ndarray] = None

    def update_from_returns(
        self,
        returns: np.ndarray,
        node_features: Optional[np.ndarray] = None,
    ) -> None:
        """Rebuild the graph from a returns matrix.

        Args:
            returns: Array of shape (n_timesteps, n_assets) with log returns.
            node_features: Optional (n_assets, node_dim) array of per-asset features.
        """
        # Compute correlation matrix
        corr = np.corrcoef(returns.T)  # (n_assets, n_assets)
        # Auto-adjust n_assets to match actual data
        actual_n = corr.shape[0]
        if actual_n != self.n:
            self.n = actual_n

        # Apply exponential decay with previous correlation
        if self._prev_corr is not None:
            corr = self.config.edge_decay * self._prev_corr + (1 - self.config.edge_decay) * corr
        self._prev_corr = corr.copy()
        self._corr_matrix = corr

        # Build edge list from correlation threshold
        rows, cols = np.where(
            (np.abs(corr) >= self.config.correlation_threshold)
            & (np.arange(self.n)[:, None] < np.arange(self.n)[None, :])
        )
        if len(rows) == 0:
            # Fallback: connect each node to itself
            rows = np.arange(self.n)
            cols = np.arange(self.n)

        # Bidirectional edges
        src = np.concatenate([rows, cols])
        dst = np.concatenate([cols, rows])
        self.edge_index = torch.tensor(
            np.stack([src, dst]), dtype=torch.long, device=self.device
        )

        # Edge features: correlation value and absolute correlation
        edge_corrs = corr[src, dst]
        self.edge_features = torch.tensor(
            np.stack([edge_corrs, np.abs(edge_corrs)], axis=-1),
            dtype=torch.float32,
            device=self.device,
        )

        # Node features
        if node_features is not None:
            self.node_features = torch.tensor(
                node_features, dtype=torch.float32, device=self.device
            )
        else:
            # Default: use asset statistics as features
            mean_ret = returns.mean(axis=0)
            vol = returns.std(axis=0)
            skew = (
                ((returns - returns.mean(axis=0)) ** 3).mean(axis=0)
                / (vol ** 3 + 1e-8)
            )
            self.node_features = torch.tensor(
                np.stack([mean_ret, vol, skew], axis=-1),
                dtype=torch.float32,
                device=self.device,
            )

    def update_single_edge(
        self,
        asset_i: int,
        asset_j: int,
        correlation: float,
    ) -> None:
        """Update a single edge weight without rebuilding the entire graph.

        Args:
            asset_i: Index of the first asset.
            asset_j: Index of the second asset.
            correlation: New correlation value.
        """
        if self._corr_matrix is not None:
            self._corr_matrix[asset_i, asset_j] = correlation
            self._corr_matrix[asset_j, asset_i] = correlation

    def get_tensors(self) -> Dict[str, Optional[Tensor]]:
        """Return graph tensors for model input.

        Returns:
            Dict with 'node_features', 'edge_index', 'edge_features'.
        """
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "edge_features": self.edge_features,
        }

    @property
    def num_edges(self) -> int:
        """Return current number of edges (bidirectional counted separately)."""
        if self.edge_index is None:
            return 0
        return self.edge_index.shape[1]

    @property
    def num_nodes(self) -> int:
        """Return number of nodes."""
        return self.n

    def get_adjacency(self) -> Optional[Tensor]:
        """Return a dense adjacency matrix with correlation weights.

        Returns:
            Tensor of shape (n, n) or None if graph not built.
        """
        if self._corr_matrix is None:
            return None
        return torch.tensor(self._corr_matrix, dtype=torch.float32, device=self.device)

    def laplacian(self, normalised: bool = True) -> Optional[Tensor]:
        """Compute the graph Laplacian.

        Args:
            normalised: If True, return the symmetric normalised Laplacian.

        Returns:
            Tensor of shape (n, n) or None.
        """
        adj = self.get_adjacency()
        if adj is None:
            return None
        degree = adj.sum(dim=-1)
        D = torch.diag(degree)
        L = D - adj
        if normalised:
            d_inv_sqrt = torch.diag(1.0 / (degree.sqrt() + 1e-8))
            L = d_inv_sqrt @ L @ d_inv_sqrt
        return L


# ---------------------------------------------------------------------------
# Graph Attention Layer
# ---------------------------------------------------------------------------

class GraphAttentionLayer(nn.Module):
    """Graph Attention Network (GAT) layer for cross-asset influence.

    Computes attention-weighted aggregations of neighbour features,
    allowing the model to learn which assets influence each other most.

    Args:
        node_dim: Input node feature dimension.
        out_dim: Output node feature dimension.
        num_heads: Number of attention heads.
        concat: If True, concatenate heads; if False, average.
        dropout: Dropout on attention coefficients.
        negative_slope: LeakyReLU negative slope for attention.
    """

    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.negative_slope = negative_slope

        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.head_dim = out_dim // num_heads

        self.W_src = nn.Linear(node_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(node_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.zeros(num_heads, self.head_dim))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self._init_weights()

    def _init_weights(self) -> None:
        """Glorot uniform initialisation."""
        nn.init.xavier_uniform_(self.W_src.weight)
        nn.init.xavier_uniform_(self.W_dst.weight)
        nn.init.zeros_(self.attn_src)
        nn.init.zeros_(self.attn_dst)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            node_features: (n_nodes, node_dim)
            edge_index: (2, num_edges)
            edge_features: Optional (num_edges, edge_dim)

        Returns:
            Updated node features (n_nodes, out_dim) if concat, else (n_nodes, out_dim // num_heads * num_heads).
        """
        N = node_features.shape[0]
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Linear projections: (N, out_dim)
        h_src = self.W_src(node_features)
        h_dst = self.W_dst(node_features)

        # Reshape for heads: (N, num_heads, head_dim)
        h_src = h_src.view(N, self.num_heads, self.head_dim)
        h_dst = h_dst.view(N, self.num_heads, self.head_dim)

        # Attention coefficients: (num_edges, num_heads)
        e_src = (h_src[src_idx] * self.attn_src.unsqueeze(0)).sum(dim=-1)
        e_dst = (h_dst[dst_idx] * self.attn_dst.unsqueeze(0)).sum(dim=-1)
        e = self.leaky_relu(e_src + e_dst)

        # Softmax over destination nodes (scatter)
        alpha = torch.zeros(N, self.num_heads, device=node_features.device)
        alpha.scatter_reduce_(
            0,
            dst_idx.unsqueeze(-1).expand_as(e),
            e,
            reduce="amax",
            include_self=True,
        )
        e_exp = (e - alpha[dst_idx]).exp()
        e_sum = torch.zeros(N, self.num_heads, device=node_features.device)
        e_sum.scatter_add_(0, dst_idx.unsqueeze(-1).expand_as(e_exp), e_exp)
        alpha_weights = e_exp / (e_sum[dst_idx] + 1e-8)
        alpha_weights = self.dropout(alpha_weights)

        # Weighted message aggregation
        messages = h_src[src_idx] * alpha_weights.unsqueeze(-1)  # (E, H, D)
        out = torch.zeros(N, self.num_heads, self.head_dim, device=node_features.device)
        out.scatter_add_(
            0,
            dst_idx.unsqueeze(-1).unsqueeze(-1).expand_as(messages),
            messages,
        )

        if self.concat:
            out = out.reshape(N, self.out_dim)
        else:
            out = out.mean(dim=1)

        return out


# ---------------------------------------------------------------------------
# Market GNN Layer
# ---------------------------------------------------------------------------

class MarketGNNLayer(nn.Module):
    """Full GNN layer combining attention, edge-feature conditioning, and skip connection.

    Args:
        node_dim: Input/output node dimension.
        edge_dim: Edge feature dimension.
        num_heads: Attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gat = GraphAttentionLayer(
            node_dim=node_dim,
            out_dim=node_dim,
            num_heads=num_heads,
            concat=True,
            dropout=dropout,
        )
        self.edge_proj = nn.Linear(edge_dim, node_dim, bias=False)
        self.layer_norm = nn.LayerNorm(node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.ffn_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with residual connections.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: Optional (E, edge_dim)

        Returns:
            (N, node_dim)
        """
        # Attention aggregation
        h = self.gat(node_features, edge_index, edge_features)

        # Edge feature conditioning (add to nodes via edge aggregation)
        if edge_features is not None and edge_features.shape[0] > 0:
            edge_emb = self.edge_proj(edge_features)  # (E, node_dim)
            dst_idx = edge_index[1]
            edge_agg = torch.zeros_like(node_features)
            edge_agg.scatter_add_(
                0,
                dst_idx.unsqueeze(-1).expand_as(edge_emb),
                edge_emb,
            )
            # Normalise by degree
            degree = torch.zeros(node_features.shape[0], 1, device=node_features.device)
            degree.scatter_add_(
                0,
                dst_idx.unsqueeze(-1),
                torch.ones(dst_idx.shape[0], 1, device=node_features.device),
            )
            edge_agg = edge_agg / (degree + 1e-8)
            h = h + edge_agg

        # Residual + LayerNorm
        h = self.layer_norm(h + node_features)

        # FFN with residual
        h = self.ffn_norm(h + self.ffn(h))
        return h


# ---------------------------------------------------------------------------
# Market GNN Model
# ---------------------------------------------------------------------------

class MarketGNN(nn.Module):
    """Full Graph Neural Network for market structure encoding.

    Stacks multiple MarketGNNLayers and produces per-node embeddings that
    capture cross-asset influence and market structure.

    Args:
        node_dim: Input node feature dimension.
        edge_dim: Edge feature dimension.
        hidden_dim: Internal hidden dimension.
        num_layers: Number of GNN layers.
        num_heads: Attention heads per layer.
        output_dim: Output embedding dimension.
        dropout: Dropout rate.
        device: Torch device.
    """

    def __init__(
        self,
        node_dim: int = 32,
        edge_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        output_dim: int = 64,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or _get_device()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input projection
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GNN layers
        self.layers = nn.ModuleList(
            [
                MarketGNNLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.to(self.device)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Produce node embeddings.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: Optional (E, edge_dim)

        Returns:
            Node embeddings of shape (N, output_dim).
        """
        h = self.input_proj(node_features)
        for layer in self.layers:
            h = layer(h, edge_index, edge_features)
        return self.output_proj(h)

    def get_graph_embedding(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        method: str = "mean",
    ) -> Tensor:
        """Produce a single graph-level embedding via readout.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: Optional (E, edge_dim)
            method: Readout method – 'mean', 'max', 'sum', or 'attention'.

        Returns:
            Graph embedding of shape (output_dim,).
        """
        node_emb = self.forward(node_features, edge_index, edge_features)

        if method == "mean":
            return node_emb.mean(dim=0)
        elif method == "max":
            return node_emb.max(dim=0).values
        elif method == "sum":
            return node_emb.sum(dim=0)
        elif method == "attention":
            # Self-attention pooling
            attn_scores = F.softmax(
                torch.matmul(node_emb, node_emb.mean(dim=0, keepdim=True).T).squeeze(-1),
                dim=0,
            )
            return (node_emb * attn_scores.unsqueeze(-1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown readout method: {method}")

    def compute_link_prediction(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        candidate_edges: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict edge existence scores for candidate pairs.

        Uses inner product of node embeddings.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: Optional (E, edge_dim)
            candidate_edges: (2, M) pairs to score; if None, scores all pairs.

        Returns:
            Scores of shape (M,).
        """
        node_emb = self.forward(node_features, edge_index, edge_features)

        if candidate_edges is None:
            # All pairs
            scores = torch.matmul(node_emb, node_emb.T)
            return scores.flatten()
        else:
            src_emb = node_emb[candidate_edges[0]]
            dst_emb = node_emb[candidate_edges[1]]
            return (src_emb * dst_emb).sum(dim=-1)


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector(nn.Module):
    """Graph-based market regime detection.

    Uses graph-level embeddings from the MarketGNN to classify the current
    market regime (e.g., bull, bear, sideways, crisis).

    Args:
        gnn: Pre-built MarketGNN model.
        num_regimes: Number of regime classes.
        hidden_dim: Hidden dimension for the classification head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        gnn: MarketGNN,
        num_regimes: int = 4,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.num_regimes = num_regimes
        self.classifier = nn.Sequential(
            nn.Linear(gnn.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_regimes),
        )
        self.to(gnn.device)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tensor:
        """Classify the current market regime.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: Optional (E, edge_dim)

        Returns:
            Logits of shape (num_regimes,).
        """
        graph_emb = self.gnn.get_graph_embedding(
            node_features, edge_index, edge_features, method="attention"
        )
        return self.classifier(graph_emb)

    @torch.no_grad()
    def predict_regime(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
    ) -> Tuple[int, float, Tensor]:
        """Predict the current regime with confidence.

        Returns:
            Tuple of (regime_id, confidence, probabilities).
        """
        self.eval()
        logits = self.forward(node_features, edge_index, edge_features)
        probs = F.softmax(logits, dim=-1)
        regime_id = probs.argmax().item()
        confidence = probs[regime_id].item()
        return regime_id, confidence, probs


# ---------------------------------------------------------------------------
# Contagion Risk Predictor
# ---------------------------------------------------------------------------

class ContagionRiskPredictor(nn.Module):
    """Predicts contagion risk through graph propagation.

    Given node features and a graph structure, predicts the probability
    that distress from one or more source nodes will propagate to other
    nodes in the graph.

    Uses a diffusion-style propagation model with learnable parameters.

    Args:
        gnn: MarketGNN backbone.
        node_dim: Node feature dimension (from GNN output).
        num_propagation_steps: Number of graph propagation steps.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        gnn: MarketGNN,
        num_propagation_steps: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gnn = gnn
        self.num_propagation_steps = num_propagation_steps

        # Propagation gate: controls how much distress flows through each edge
        self.prop_gate = nn.Sequential(
            nn.Linear(gnn.output_dim * 2, gnn.output_dim),
            nn.Sigmoid(),
        )
        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(gnn.output_dim, gnn.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gnn.output_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.to(gnn.device)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        source_risk: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict contagion risk for each node.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: Optional (E, edge_dim)
            source_risk: Optional (N,) initial risk scores (1 = distressed, 0 = healthy).

        Returns:
            Risk scores of shape (N,) in [0, 1].
        """
        N = node_features.shape[0]
        device = node_features.device

        # Get GNN embeddings
        node_emb = self.gnn(node_features, edge_index, edge_features)

        # Initialise risk signal
        if source_risk is None:
            risk_signal = torch.zeros(N, device=device)
        else:
            risk_signal = source_risk

        # Iterative propagation
        for _ in range(self.num_propagation_steps):
            risk_emb = risk_signal.unsqueeze(-1) * node_emb  # (N, D)
            src_idx, dst_idx = edge_index[0], edge_index[1]

            # Message: source risk embedding
            messages = risk_emb[src_idx]  # (E, D)

            # Gating: decide how much risk propagates through each edge
            gate_input = torch.cat([messages, node_emb[dst_idx]], dim=-1)
            gate = self.prop_gate(gate_input)  # (E, D)
            gated_messages = messages * gate

            # Aggregate
            propagated = torch.zeros_like(node_emb)
            propagated.scatter_add_(
                0,
                dst_idx.unsqueeze(-1).expand_as(gated_messages),
                gated_messages,
            )
            # Average by degree
            degree = torch.zeros(N, 1, device=device)
            degree.scatter_add_(
                0,
                dst_idx.unsqueeze(-1),
                torch.ones(dst_idx.shape[0], 1, device=device),
            )
            propagated = propagated / (degree + 1e-8)

            # Update risk signal: combine with propagated risk
            risk_signal = torch.sigmoid(
                risk_signal + propagated.mean(dim=-1) * 0.5
            )

        # Final risk prediction
        risk_scores = self.risk_head(node_emb + risk_signal.unsqueeze(-1) * node_emb)
        return risk_scores.squeeze(-1)

    @torch.no_grad()
    def predict_contagion(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_features: Optional[Tensor] = None,
        source_nodes: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        """Predict contagion risk with source node specification.

        Args:
            node_features: (N, node_dim)
            edge_index: (2, E)
            edge_features: Optional (E, edge_dim)
            source_nodes: Indices of distressed source nodes.

        Returns:
            Dict with 'risk_scores' (N,) and 'at_risk_mask' (N,) boolean.
        """
        self.eval()
        N = node_features.shape[0]

        if source_nodes is not None:
            source_risk = torch.zeros(N, device=node_features.device)
            source_risk[source_nodes] = 1.0
        else:
            source_risk = None

        risk = self.forward(node_features, edge_index, edge_features, source_risk)
        at_risk = risk > 0.5

        return {
            "risk_scores": risk,
            "at_risk_mask": at_risk,
        }

    def compute_loss(
        self,
        predicted_risk: Tensor,
        target_risk: Tensor,
    ) -> Tensor:
        """Compute binary cross-entropy loss for risk prediction.

        Args:
            predicted_risk: (N,) predicted risk scores.
            target_risk: (N,) ground-truth risk labels.

        Returns:
            Scalar loss.
        """
        return F.binary_cross_entropy(predicted_risk, target_risk)
