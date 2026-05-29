"""
MarketKnowledgeGraph - Self-Learning Knowledge Graph for Crypto Markets
========================================================================

A Neo4j-like knowledge graph implementation backed by Redis/Postgres that
captures market relationships, temporal facts, and enables pattern mining
for the Algorithmic Crypto Management System (ACMS).

Features:
- NodeType/EdgeType enums for structured graph modeling
- Temporal knowledge tracking with confidence-weighted edges
- Automatic graph updates from trading outcomes
- Pattern mining for discovering market relationships
- Knowledge transfer between similar assets
- GPU-accelerated similarity computations via PyTorch
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Types of entities that can be represented as nodes in the graph."""

    ASSET = "asset"
    STRATEGY = "strategy"
    INDICATOR = "indicator"
    REGIME = "regime"
    EVENT = "event"


class EdgeType(str, Enum):
    """Types of relationships between graph nodes."""

    CORRELATED = "correlated"
    INFLUENCES = "influences"
    COINTEGRATED = "cointegrated"
    LEADS = "leads"
    DEPENDS = "depends"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A node in the market knowledge graph.

    Attributes:
        node_id: Unique identifier for the node.
        node_type: The type of entity this node represents.
        name: Human-readable name.
        properties: Arbitrary key-value metadata.
        embedding: Dense vector representation for similarity search.
        created_at: Timestamp of node creation.
        updated_at: Timestamp of last update.
    """

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.ASSET
    name: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to a dictionary for storage."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "properties": json.dumps(self.properties, default=str),
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphNode:
        """Deserialize a node from a dictionary."""
        emb = data.get("embedding")
        return cls(
            node_id=data["node_id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            properties=json.loads(data["properties"]) if isinstance(data["properties"], str) else data["properties"],
            embedding=np.array(emb) if emb is not None else None,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


@dataclass
class GraphEdge:
    """A directed, confidence-weighted edge in the knowledge graph.

    Attributes:
        edge_id: Unique identifier.
        source_id: Source node ID.
        target_id: Target node ID.
        edge_type: Type of relationship.
        confidence: Confidence score in [0, 1].
        weight: Numerical weight (e.g., correlation coefficient).
        properties: Arbitrary metadata.
        valid_from: When this edge became true.
        valid_to: When this edge ceased to be true (None = still valid).
        created_at: Creation timestamp.
    """

    edge_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.CORRELATED
    confidence: float = 0.5
    weight: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_to: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge for storage."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "weight": self.weight,
            "properties": json.dumps(self.properties, default=str),
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat() if self.valid_to else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphEdge:
        """Deserialize an edge from a dictionary."""
        return cls(
            edge_id=data["edge_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            edge_type=EdgeType(data["edge_type"]),
            confidence=data["confidence"],
            weight=data["weight"],
            properties=json.loads(data["properties"]) if isinstance(data["properties"], str) else data["properties"],
            valid_from=datetime.fromisoformat(data["valid_from"]),
            valid_to=datetime.fromisoformat(data["valid_to"]) if data.get("valid_to") else None,
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class TemporalFact:
    """A time-stamped fact with exponential decay of relevance.

    Facts age out over time, simulating the idea that older information
    is less relevant for current decision making.

    Attributes:
        fact_id: Unique identifier.
        subject_id: Node ID the fact is about.
        predicate: Relationship or property name.
        object_value: Value of the fact.
        confidence: Initial confidence.
        timestamp: When the fact was recorded.
        decay_rate: Exponential decay rate (higher = faster forgetting).
        half_life_hours: Hours until confidence drops by half.
    """

    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    predicate: str = ""
    object_value: Any = None
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    decay_rate: float = 0.01
    half_life_hours: float = 168.0  # 1 week default

    def current_confidence(self, now: Optional[datetime] = None) -> float:
        """Compute confidence after temporal decay.

        Args:
            now: Current time (defaults to UTC now).

        Returns:
            Decayed confidence value in [0, 1].
        """
        if now is None:
            now = datetime.now(timezone.utc)
        elapsed_hours = max(0.0, (now - self.timestamp).total_seconds() / 3600.0)
        decay = np.exp(-self.decay_rate * elapsed_hours / self.half_life_hours)
        return float(self.confidence * decay)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "fact_id": self.fact_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_value": json.dumps(self.object_value, default=str),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "decay_rate": self.decay_rate,
            "half_life_hours": self.half_life_hours,
        }


# ---------------------------------------------------------------------------
# Pattern Miner
# ---------------------------------------------------------------------------

class PatternMiner:
    """Discovers recurring structural patterns in the knowledge graph.

    Uses subgraph isomorphism and frequency counting to find motifs
    such as triangular arbitrage loops, lead-lag chains, and regime
    transition sequences.

    GPU acceleration is used for embedding-based pattern similarity
    when PyTorch + CUDA are available.
    """

    def __init__(self, min_support: int = 3, max_pattern_size: int = 5) -> None:
        """Initialise the pattern miner.

        Args:
            min_support: Minimum occurrences for a pattern to be considered valid.
            max_pattern_size: Maximum number of edges in a discovered pattern.
        """
        self.min_support = min_support
        self.max_pattern_size = max_pattern_size
        self._pattern_cache: Dict[str, int] = {}

    def mine_triangular_patterns(
        self, edges: List[GraphEdge]
    ) -> List[List[GraphEdge]]:
        """Find all triangular cycles (A->B->C->A) in the edge set.

        Triangular patterns often indicate arbitrage loops or
        multi-step influence chains.

        Args:
            edges: List of graph edges to search.

        Returns:
            List of triangular edge triplets.
        """
        adj: Dict[str, List[GraphEdge]] = {}
        for edge in edges:
            adj.setdefault(edge.source_id, []).append(edge)

        triangles: List[List[GraphEdge]] = []
        seen: set = set()

        for e1 in edges:
            for e2 in adj.get(e1.target_id, []):
                for e3 in adj.get(e2.target_id, []):
                    if e3.target_id == e1.source_id:
                        tri_key = tuple(sorted([e1.edge_id, e2.edge_id, e3.edge_id]))
                        if tri_key not in seen:
                            seen.add(tri_key)
                            triangles.append([e1, e2, e3])

        logger.debug("Found %d triangular patterns", len(triangles))
        return triangles

    def mine_frequent_edge_sequences(
        self, edge_sequences: List[List[GraphEdge]], window: int = 3
    ) -> Dict[str, int]:
        """Mine frequent sequential patterns from ordered edge lists.

        Args:
            edge_sequences: Lists of edges (e.g. temporal event chains).
            window: Maximum sequence length to consider.

        Returns:
            Dictionary mapping pattern signatures to support counts.
        """
        freq: Dict[str, int] = {}

        for seq in edge_sequences:
            for length in range(2, min(window + 1, len(seq) + 1)):
                for start in range(len(seq) - length + 1):
                    subseq = seq[start : start + length]
                    pattern_key = "->".join(
                        f"{e.source_id}:{e.edge_type.value}:{e.target_id}" for e in subseq
                    )
                    freq[pattern_key] = freq.get(pattern_key, 0) + 1

        significant = {k: v for k, v in freq.items() if v >= self.min_support}
        self._pattern_cache.update(significant)
        return significant

    def compute_embedding_similarity(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[Tuple[str, np.ndarray]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Find the most similar embeddings to a query using cosine similarity.

        GPU-accelerated when CUDA is available.

        Args:
            query_embedding: The query vector.
            candidate_embeddings: List of (id, vector) pairs.
            top_k: Number of top results to return.

        Returns:
            List of (id, similarity) sorted by descending similarity.
        """
        if not candidate_embeddings:
            return []

        if GPU_AVAILABLE and torch is not None:
            q = torch.tensor(query_embedding, dtype=torch.float32, device="cuda")
            q = F.normalize(q, dim=0)
            ids = [c[0] for c in candidate_embeddings]
            mat = torch.tensor(
                np.stack([c[1] for c in candidate_embeddings]),
                dtype=torch.float32,
                device="cuda",
            )
            mat = F.normalize(mat, dim=1)
            sims = torch.matmul(mat, q).cpu().numpy()
        else:
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
            ids = [c[0] for c in candidate_embeddings]
            mat = np.stack([c[1] for c in candidate_embeddings])
            mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
            sims = mat_norm @ q_norm

        ranked = sorted(zip(ids, sims.tolist()), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Market Knowledge Graph
# ---------------------------------------------------------------------------

class MarketKnowledgeGraph:
    """Self-learning knowledge graph for crypto market intelligence.

    Maintains nodes (assets, strategies, indicators, regimes, events) and
    confidence-weighted, temporal edges. Supports graph queries for similar
    regimes, strategy recommendations, and automatic updates from trading
    outcomes. Backed by Redis for fast graph traversal and Postgres for
    durable storage.

    Attributes:
        graph_id: Unique identifier for this graph instance.
        nodes: In-memory node cache (node_id -> GraphNode).
        edges: In-memory edge cache (edge_id -> GraphEdge).
        facts: Temporal facts list.
        pattern_miner: Pattern mining engine.
    """

    def __init__(
        self,
        graph_id: str = "default",
        redis_client: Any = None,
        postgres_client: Any = None,
        device: str = "auto",
    ) -> None:
        """Initialise the knowledge graph.

        Args:
            graph_id: Unique identifier for this graph instance.
            redis_client: Optional async Redis client for caching.
            postgres_client: Optional async Postgres client for persistence.
            device: Compute device ('auto', 'cuda', 'cpu').
        """
        self.graph_id = graph_id
        self._redis = redis_client
        self._postgres = postgres_client
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.facts: List[TemporalFact] = []
        self.pattern_miner = PatternMiner()

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        self._adj_out: Dict[str, List[str]] = {}  # node_id -> [edge_ids]
        self._adj_in: Dict[str, List[str]] = {}   # node_id -> [edge_ids]
        self._node_name_index: Dict[str, str] = {} # name -> node_id

        logger.info(
            "MarketKnowledgeGraph initialised [id=%s, device=%s]",
            self.graph_id,
            self._device,
        )

    # -- Node Operations ---------------------------------------------------

    def add_node(self, node: GraphNode) -> str:
        """Add a node to the graph.

        If a node with the same name and type already exists, its properties
        are merged and the embedding is updated.

        Args:
            node: The GraphNode to add.

        Returns:
            The node_id of the added or updated node.
        """
        # Deduplicate by name + type
        existing_id = self._node_name_index.get(f"{node.node_type.value}:{node.name}")
        if existing_id and existing_id in self.nodes:
            existing = self.nodes[existing_id]
            existing.properties.update(node.properties)
            if node.embedding is not None:
                existing.embedding = node.embedding
            existing.updated_at = datetime.now(timezone.utc)
            logger.debug("Updated existing node: %s (%s)", node.name, node.node_type.value)
            return existing_id

        self.nodes[node.node_id] = node
        self._adj_out.setdefault(node.node_id, [])
        self._adj_in.setdefault(node.node_id, [])
        self._node_name_index[f"{node.node_type.value}:{node.name}"] = node.node_id
        logger.debug("Added node: %s (%s)", node.name, node.node_type.value)
        return node.node_id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Retrieve a node by ID.

        Args:
            node_id: The unique node identifier.

        Returns:
            The GraphNode if found, else None.
        """
        return self.nodes.get(node_id)

    def find_nodes(
        self,
        node_type: Optional[NodeType] = None,
        name_pattern: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> List[GraphNode]:
        """Search for nodes matching criteria.

        Args:
            node_type: Filter by node type.
            name_pattern: Substring match on node name.
            min_confidence: Minimum average edge confidence.

        Returns:
            List of matching GraphNode objects.
        """
        results: List[GraphNode] = []
        for node in self.nodes.values():
            if node_type and node.node_type != node_type:
                continue
            if name_pattern and name_pattern.lower() not in node.name.lower():
                continue
            if min_confidence > 0:
                avg_conf = self._avg_node_confidence(node.node_id)
                if avg_conf < min_confidence:
                    continue
            results.append(node)
        return results

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connected edges.

        Args:
            node_id: ID of the node to remove.

        Returns:
            True if the node was found and removed.
        """
        if node_id not in self.nodes:
            return False

        # Remove connected edges
        edge_ids_to_remove = list(
            self._adj_out.get(node_id, []) + self._adj_in.get(node_id, [])
        )
        for eid in edge_ids_to_remove:
            self.edges.pop(eid, None)

        # Clean adjacency lists
        self._adj_out.pop(node_id, None)
        self._adj_in.pop(node_id, None)

        # Clean name index
        node = self.nodes[node_id]
        key = f"{node.node_type.value}:{node.name}"
        self._node_name_index.pop(key, None)

        del self.nodes[node_id]
        logger.debug("Removed node: %s", node_id)
        return True

    # -- Edge Operations ---------------------------------------------------

    def add_edge(self, edge: GraphEdge) -> str:
        """Add a confidence-weighted edge to the graph.

        If a same-type edge already exists between the same source and target,
        the confidence and weight are updated using exponential moving average.

        Args:
            edge: The GraphEdge to add.

        Returns:
            The edge_id of the added or updated edge.
        """
        # Check for existing edge of same type between same nodes
        for eid in self._adj_out.get(edge.source_id, []):
            existing = self.edges.get(eid)
            if (
                existing
                and existing.target_id == edge.target_id
                and existing.edge_type == edge.edge_type
            ):
                # EMA update
                alpha = 0.3
                existing.confidence = alpha * edge.confidence + (1 - alpha) * existing.confidence
                existing.weight = alpha * edge.weight + (1 - alpha) * existing.weight
                existing.updated_at = datetime.now(timezone.utc) if hasattr(existing, 'updated_at') else None
                logger.debug(
                    "Updated edge %s: conf=%.3f, weight=%.3f",
                    eid,
                    existing.confidence,
                    existing.weight,
                )
                return eid

        self.edges[edge.edge_id] = edge
        self._adj_out.setdefault(edge.source_id, []).append(edge.edge_id)
        self._adj_in.setdefault(edge.target_id, []).append(edge.edge_id)
        logger.debug(
            "Added edge: %s -[%s]-> %s (conf=%.3f)",
            edge.source_id,
            edge.edge_type.value,
            edge.target_id,
            edge.confidence,
        )
        return edge.edge_id

    def get_neighbors(
        self, node_id: str, edge_type: Optional[EdgeType] = None, direction: str = "out"
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes connected by edges.

        Args:
            node_id: The node to find neighbors for.
            edge_type: Filter by edge type.
            direction: 'out', 'in', or 'both'.

        Returns:
            List of (neighbor_node, connecting_edge) pairs.
        """
        adj_key = self._adj_out if direction == "out" else self._adj_in
        edge_ids = list(adj_key.get(node_id, []))
        if direction == "both":
            edge_ids += list(self._adj_in.get(node_id, []))

        results: List[Tuple[GraphNode, GraphEdge]] = []
        for eid in edge_ids:
            edge = self.edges.get(eid)
            if edge is None:
                continue
            if edge_type and edge.edge_type != edge_type:
                continue
            neighbor_id = edge.target_id if edge.source_id == node_id else edge.source_id
            neighbor = self.nodes.get(neighbor_id)
            if neighbor:
                results.append((neighbor, edge))
        return results

    def get_edge(self, source_id: str, target_id: str, edge_type: Optional[EdgeType] = None) -> Optional[GraphEdge]:
        """Get a specific edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Optional edge type filter.

        Returns:
            The first matching GraphEdge, or None.
        """
        for eid in self._adj_out.get(source_id, []):
            edge = self.edges.get(eid)
            if edge and edge.target_id == target_id:
                if edge_type is None or edge.edge_type == edge_type:
                    return edge
        return None

    # -- Temporal Fact Operations ------------------------------------------

    def add_fact(self, fact: TemporalFact) -> str:
        """Record a temporal fact with decay tracking.

        Args:
            fact: The TemporalFact to record.

        Returns:
            The fact_id.
        """
        self.facts.append(fact)
        logger.debug("Added fact: %s %s (conf=%.3f)", fact.subject_id, fact.predicate, fact.confidence)
        return fact.fact_id

    def get_active_facts(
        self, subject_id: str, min_confidence: float = 0.1, now: Optional[datetime] = None
    ) -> List[TemporalFact]:
        """Retrieve facts that are still relevant after temporal decay.

        Args:
            subject_id: Node ID to filter facts by.
            min_confidence: Minimum decayed confidence threshold.
            now: Current time for decay computation.

        Returns:
            List of active TemporalFact objects.
        """
        return [
            f
            for f in self.facts
            if f.subject_id == subject_id and f.current_confidence(now) >= min_confidence
        ]

    # -- Graph Queries -----------------------------------------------------

    def find_similar_regimes(
        self, current_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[GraphNode, float]]:
        """Find historical regimes most similar to the current market state.

        Uses cosine similarity on regime node embeddings, GPU-accelerated
        when available.

        Args:
            current_embedding: Embedding vector for current regime.
            top_k: Number of similar regimes to return.

        Returns:
            List of (regime_node, similarity) pairs, sorted descending.
        """
        regime_nodes = [n for n in self.nodes.values() if n.node_type == NodeType.REGIME]
        candidates: List[Tuple[str, np.ndarray]] = []
        for n in regime_nodes:
            if n.embedding is not None:
                candidates.append((n.node_id, n.embedding))

        if not candidates:
            return []

        similarities = self.pattern_miner.compute_embedding_similarity(
            current_embedding, candidates, top_k=top_k
        )
        return [(self.nodes[nid], sim) for nid, sim in similarities if nid in self.nodes]

    def recommend_strategies(
        self, regime_node_id: str, top_k: int = 5
    ) -> List[Tuple[GraphNode, float]]:
        """Recommend strategies for a given regime based on graph relationships.

        Follows INFLUENCES and DEPENDS edges from the regime to strategy
        nodes, aggregating confidence scores.

        Args:
            regime_node_id: ID of the regime node.
            top_k: Number of strategies to recommend.

        Returns:
            List of (strategy_node, aggregated_confidence) pairs.
        """
        strategy_scores: Dict[str, float] = {}

        # Direct connections
        neighbors = self.get_neighbors(regime_node_id, direction="both")
        for neighbor, edge in neighbors:
            if neighbor.node_type == NodeType.STRATEGY:
                strategy_scores[neighbor.node_id] = max(
                    strategy_scores.get(neighbor.node_id, 0.0), edge.confidence
                )

        # Two-hop connections via indicators
        for neighbor, edge in neighbors:
            if neighbor.node_type == NodeType.INDICATOR:
                indicator_neighbors = self.get_neighbors(neighbor.node_id)
                for strat_node, strat_edge in indicator_neighbors:
                    if strat_node.node_type == NodeType.STRATEGY:
                        combined = edge.confidence * strat_edge.confidence
                        strategy_scores[strat_node.node_id] = max(
                            strategy_scores.get(strat_node.node_id, 0.0), combined
                        )

        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (self.nodes[nid], score)
            for nid, score in sorted_strategies[:top_k]
            if nid in self.nodes
        ]

    # -- Automatic Updates from Trading Outcomes ---------------------------

    def update_from_trade_outcome(
        self,
        asset_name: str,
        strategy_name: str,
        regime_name: str,
        pnl: float,
        sharpe: float,
        indicators_used: Optional[List[str]] = None,
    ) -> None:
        """Automatically update the graph based on a completed trade.

        Creates or updates nodes and edges reflecting the trading outcome.
        Positive outcomes strengthen edges; negative outcomes weaken them.

        Args:
            asset_name: Name of the traded asset.
            strategy_name: Name of the strategy used.
            regime_name: Name of the market regime.
            pnl: Profit and loss from the trade.
            sharpe: Sharpe ratio of the trade.
            indicators_used: List of indicator names involved.
        """
        # Ensure nodes exist
        asset_id = self.add_node(GraphNode(node_type=NodeType.ASSET, name=asset_name))
        strategy_id = self.add_node(GraphNode(node_type=NodeType.STRATEGY, name=strategy_name))
        regime_id = self.add_node(GraphNode(node_type=NodeType.REGIME, name=regime_name))

        # Outcome-based confidence adjustment
        outcome_score = float(np.clip(sharpe / 3.0, -1.0, 1.0))  # Normalise Sharpe
        new_confidence = 0.5 + 0.5 * outcome_score  # Map to [0, 1]

        # Regime -> Strategy edge
        self.add_edge(
            GraphEdge(
                source_id=regime_id,
                target_id=strategy_id,
                edge_type=EdgeType.INFLUENCES,
                confidence=new_confidence,
                weight=sharpe,
            )
        )

        # Strategy -> Asset edge
        self.add_edge(
            GraphEdge(
                source_id=strategy_id,
                target_id=asset_id,
                edge_type=EdgeType.DEPENDS,
                confidence=max(0.1, new_confidence),
                weight=pnl,
            )
        )

        # Indicator connections
        if indicators_used:
            for ind_name in indicators_used:
                ind_id = self.add_node(
                    GraphNode(node_type=NodeType.INDICATOR, name=ind_name)
                )
                self.add_edge(
                    GraphEdge(
                        source_id=ind_id,
                        target_id=strategy_id,
                        edge_type=EdgeType.INFLUENCES,
                        confidence=new_confidence * 0.8,
                    )
                )

        # Record temporal fact
        self.add_fact(
            TemporalFact(
                subject_id=regime_id,
                predicate=f"strategy:{strategy_name}:pnl",
                object_value={"pnl": pnl, "sharpe": sharpe},
                confidence=new_confidence,
            )
        )

        logger.info(
            "Updated graph from trade: asset=%s strategy=%s regime=%s pnl=%.4f sharpe=%.2f",
            asset_name,
            strategy_name,
            regime_name,
            pnl,
            sharpe,
        )

    # -- Knowledge Transfer ------------------------------------------------

    def transfer_knowledge(
        self, source_asset: str, target_asset: str, edge_types: Optional[List[EdgeType]] = None
    ) -> int:
        """Transfer knowledge (edges and facts) from one asset to another.

        Useful when a new asset enters the portfolio that shares
        characteristics with an existing well-modelled asset.

        Args:
            source_asset: Name of the source asset.
            target_asset: Name of the target asset.
            edge_types: Only transfer edges of these types (None = all).

        Returns:
            Number of edges transferred.
        """
        source_id = self._node_name_index.get(f"{NodeType.ASSET.value}:{source_asset}")
        target_id = self._node_name_index.get(f"{NodeType.ASSET.value}:{target_asset}")

        if not source_id or not target_id:
            logger.warning("Knowledge transfer failed: source=%s target=%s not found", source_asset, target_asset)
            return 0

        transferred = 0
        neighbors = self.get_neighbors(source_id, direction="both")
        for neighbor, edge in neighbors:
            if edge_types and edge.edge_type not in edge_types:
                continue
            # Create corresponding edge for target asset
            if edge.source_id == source_id:
                self.add_edge(
                    GraphEdge(
                        source_id=target_id,
                        target_id=neighbor.node_id,
                        edge_type=edge.edge_type,
                        confidence=edge.confidence * 0.7,  # Discount for transfer
                        weight=edge.weight,
                    )
                )
            else:
                self.add_edge(
                    GraphEdge(
                        source_id=neighbor.node_id,
                        target_id=target_id,
                        edge_type=edge.edge_type,
                        confidence=edge.confidence * 0.7,
                        weight=edge.weight,
                    )
                )
            transferred += 1

        # Transfer relevant facts
        source_facts = self.get_active_facts(source_id, min_confidence=0.2)
        for fact in source_facts:
            self.add_fact(
                TemporalFact(
                    subject_id=target_id,
                    predicate=f"transferred:{fact.predicate}",
                    object_value=fact.object_value,
                    confidence=fact.current_confidence() * 0.7,
                    decay_rate=fact.decay_rate * 1.5,  # Decay faster for transferred facts
                    half_life_hours=fact.half_life_hours * 0.5,
                )
            )

        logger.info(
            "Transferred %d edges and %d facts from %s to %s",
            transferred,
            len(source_facts),
            source_asset,
            target_asset,
        )
        return transferred

    # -- Pattern Mining ----------------------------------------------------

    def mine_patterns(self) -> Dict[str, Any]:
        """Run all pattern mining algorithms on the current graph.

        Returns:
            Dictionary with discovered patterns and their support counts.
        """
        edge_list = list(self.edges.values())

        triangles = self.pattern_miner.mine_triangular_patterns(edge_list)
        frequent = self.pattern_miner.mine_frequent_edge_sequences([edge_list])

        return {
            "triangular_patterns": len(triangles),
            "frequent_sequences": frequent,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_facts": len(self.facts),
        }

    # -- Graph Statistics --------------------------------------------------

    def _avg_node_confidence(self, node_id: str) -> float:
        """Compute the average confidence of edges connected to a node.

        Args:
            node_id: The node ID.

        Returns:
            Average confidence, or 0.0 if no edges.
        """
        edge_ids = self._adj_out.get(node_id, []) + self._adj_in.get(node_id, [])
        if not edge_ids:
            return 0.0
        confidences = [
            self.edges[eid].confidence for eid in edge_ids if eid in self.edges
        ]
        return float(np.mean(confidences)) if confidences else 0.0

    def graph_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the knowledge graph.

        Returns:
            Dictionary of graph statistics.
        """
        node_type_counts: Dict[str, int] = {}
        for n in self.nodes.values():
            node_type_counts[n.node_type.value] = node_type_counts.get(n.node_type.value, 0) + 1

        edge_type_counts: Dict[str, int] = {}
        avg_conf_by_type: Dict[str, List[float]] = {}
        for e in self.edges.values():
            edge_type_counts[e.edge_type.value] = edge_type_counts.get(e.edge_type.value, 0) + 1
            avg_conf_by_type.setdefault(e.edge_type.value, []).append(e.confidence)

        avg_conf = {
            k: float(np.mean(v)) for k, v in avg_conf_by_type.items() if v
        }

        active_facts = sum(
            1 for f in self.facts if f.current_confidence() >= 0.1
        )

        return {
            "graph_id": self.graph_id,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_facts": len(self.facts),
            "active_facts": active_facts,
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
            "avg_confidence_by_edge_type": avg_conf,
            "device": self._device,
        }

    # -- Persistence -------------------------------------------------------

    async def save_to_postgres(self) -> int:
        """Persist the entire graph to Postgres.

        Returns:
            Number of records saved.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping save")
            return 0

        count = 0
        try:
            async with self._postgres.transaction():
                for node in self.nodes.values():
                    await self._postgres.execute(
                        """
                        INSERT INTO knowledge_nodes (node_id, graph_id, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (node_id) DO UPDATE SET data = $3
                        """,
                        node.node_id,
                        self.graph_id,
                        json.dumps(node.to_dict()),
                    )
                    count += 1
                for edge in self.edges.values():
                    await self._postgres.execute(
                        """
                        INSERT INTO knowledge_edges (edge_id, graph_id, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (edge_id) DO UPDATE SET data = $3
                        """,
                        edge.edge_id,
                        self.graph_id,
                        json.dumps(edge.to_dict()),
                    )
                    count += 1
            logger.info("Saved %d records to Postgres", count)
        except Exception as exc:
            logger.error("Failed to save graph to Postgres: %s", exc)
        return count

    async def load_from_postgres(self) -> int:
        """Load the graph from Postgres.

        Returns:
            Number of records loaded.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping load")
            return 0

        count = 0
        try:
            rows = await self._postgres.fetch(
                "SELECT data FROM knowledge_nodes WHERE graph_id = $1", self.graph_id
            )
            for row in rows:
                node = GraphNode.from_dict(json.loads(row["data"]))
                self.nodes[node.node_id] = node
                self._adj_out.setdefault(node.node_id, [])
                self._adj_in.setdefault(node.node_id, [])
                self._node_name_index[f"{node.node_type.value}:{node.name}"] = node.node_id
                count += 1

            rows = await self._postgres.fetch(
                "SELECT data FROM knowledge_edges WHERE graph_id = $1", self.graph_id
            )
            for row in rows:
                edge = GraphEdge.from_dict(json.loads(row["data"]))
                self.edges[edge.edge_id] = edge
                self._adj_out.setdefault(edge.source_id, []).append(edge.edge_id)
                self._adj_in.setdefault(edge.target_id, []).append(edge.edge_id)
                count += 1

            logger.info("Loaded %d records from Postgres", count)
        except Exception as exc:
            logger.error("Failed to load graph from Postgres: %s", exc)
        return count

    async def cache_to_redis(self) -> int:
        """Cache frequently accessed graph data to Redis.

        Returns:
            Number of keys cached.
        """
        if self._redis is None:
            logger.warning("No Redis client configured; skipping cache")
            return 0

        count = 0
        try:
            # Cache node embeddings for similarity search
            for node in self.nodes.values():
                if node.embedding is not None:
                    key = f"graph:{self.graph_id}:emb:{node.node_id}"
                    await self._redis.set(
                        key,
                        json.dumps(
                            {
                                "node_id": node.node_id,
                                "embedding": node.embedding.tolist(),
                                "node_type": node.node_type.value,
                            }
                        ),
                    )
                    count += 1
            logger.info("Cached %d entries to Redis", count)
        except Exception as exc:
            logger.error("Failed to cache to Redis: %s", exc)
        return count
