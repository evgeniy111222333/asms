"""
MarketMemory - Episodic and Semantic Memory for Crypto Markets
==============================================================

Implements a dual-store memory system inspired by human memory:
- Episodic Memory: Full market scenarios with context, actions, and outcomes
- Semantic Memory: Abstracted patterns, rules, and compressed knowledge
- Memory Index for fast retrieval of similar historical situations
- Memory Consolidation for periodic compression of old memories
- Forgetting curve implementation for relevance decay
- Memory replay for model training and reinforcement learning

GPU-ready with PyTorch-based similarity computations.
"""

from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    """Classification of memory types."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class EpisodeOutcome(str, Enum):
    """Possible outcomes for an episodic memory."""

    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    PARTIAL = "partial"


class RegimeType(str, Enum):
    """Market regime classifications for memory tagging."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CRISIS = "crisis"
    RECOVERY = "recovery"


# ---------------------------------------------------------------------------
# Forgetting Curve
# ---------------------------------------------------------------------------

class ForgettingCurve:
    """Implements Ebbinghaus-style forgetting curves for memory relevance.

    Supports multiple decay models:
    - Exponential: R(t) = R0 * exp(-lambda * t)
    - Power: R(t) = R0 * (1 + beta * t)^(-alpha)
    - Logarithmic: R(t) = R0 * log(1 + gamma) / log(1 + gamma * t)

    The curve determines how quickly a memory's retrieval strength
    decays over time, and can be boosted by "rehearsal" (reuse).

    Attributes:
        decay_model: Name of the decay model.
        initial_strength: Initial retention strength (0-1).
        params: Model-specific parameters.
        rehearsal_count: Number of times this memory has been rehearsed.
    """

    def __init__(
        self,
        decay_model: str = "exponential",
        initial_strength: float = 1.0,
        params: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialise the forgetting curve.

        Args:
            decay_model: One of 'exponential', 'power', 'logarithmic'.
            initial_strength: Starting retention strength.
            params: Model parameters (lambda, alpha, beta, gamma).
        """
        self.decay_model = decay_model
        self.initial_strength = initial_strength
        self.params = params or self._default_params(decay_model)
        self.rehearsal_count = 0

    @staticmethod
    def _default_params(model: str) -> Dict[str, float]:
        """Return default parameters for a decay model."""
        defaults = {
            "exponential": {"lambda": 0.05},
            "power": {"alpha": 0.8, "beta": 0.1},
            "logarithmic": {"gamma": 1.0},
        }
        return defaults.get(model, defaults["exponential"])

    def retention(self, elapsed_hours: float) -> float:
        """Compute retention strength after elapsed time.

        Args:
            elapsed_hours: Hours since the memory was last rehearsed.

        Returns:
            Retention value in [0, 1].
        """
        if elapsed_hours <= 0:
            return self.initial_strength

        # Rehearsal boost: each rehearsal slows decay
        rehearsal_factor = 1.0 + 0.2 * self.rehearsal_count
        t = elapsed_hours / rehearsal_factor

        if self.decay_model == "exponential":
            lam = self.params.get("lambda", 0.05)
            strength = self.initial_strength * np.exp(-lam * t)
        elif self.decay_model == "power":
            alpha = self.params.get("alpha", 0.8)
            beta = self.params.get("beta", 0.1)
            strength = self.initial_strength * (1 + beta * t) ** (-alpha)
        elif self.decay_model == "logarithmic":
            gamma = self.params.get("gamma", 1.0)
            strength = self.initial_strength * np.log(1 + gamma) / np.log(1 + gamma * max(t, 0.01))
        else:
            strength = self.initial_strength

        return float(np.clip(strength, 0.0, 1.0))

    def rehearse(self) -> None:
        """Boost retention by rehearsing (re-accessing) the memory."""
        self.rehearsal_count += 1
        logger.debug("Memory rehearsed (count=%d)", self.rehearsal_count)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "decay_model": self.decay_model,
            "initial_strength": self.initial_strength,
            "params": self.params,
            "rehearsal_count": self.rehearsal_count,
        }


# ---------------------------------------------------------------------------
# Episode (Episodic Memory)
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A complete market scenario stored in episodic memory.

    Captures the full context of a trading situation, including
    market state, actions taken, outcomes achieved, and the
    embedding for similarity-based retrieval.

    Attributes:
        episode_id: Unique identifier.
        timestamp: When this episode occurred.
        regime: Market regime at the time.
        market_context: Dict of market features (prices, volumes, etc.).
        actions_taken: List of trading actions executed.
        outcomes: Result metrics (PnL, Sharpe, drawdown, etc.).
        outcome_label: Categorical outcome classification.
        context_embedding: Dense vector for similarity search.
        tags: Searchable tags for categorical filtering.
        forgetting_curve: Retention decay model for this episode.
        replay_count: How many times this episode has been replayed.
    """

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    regime: RegimeType = RegimeType.RANGING
    market_context: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    outcomes: Dict[str, float] = field(default_factory=dict)
    outcome_label: EpisodeOutcome = EpisodeOutcome.NEUTRAL
    context_embedding: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)
    forgetting_curve: ForgettingCurve = field(default_factory=ForgettingCurve)
    replay_count: int = 0

    @property
    def retention(self) -> float:
        """Current retention strength of this episode."""
        elapsed = (datetime.now(timezone.utc) - self.timestamp).total_seconds() / 3600.0
        return self.forgetting_curve.retention(elapsed)

    @property
    def pnl(self) -> float:
        """Total PnL from this episode."""
        return self.outcomes.get("pnl", 0.0)

    @property
    def sharpe(self) -> float:
        """Sharpe ratio from this episode."""
        return self.outcomes.get("sharpe", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp.isoformat(),
            "regime": self.regime.value,
            "market_context": json.dumps(self.market_context, default=str),
            "actions_taken": json.dumps(self.actions_taken, default=str),
            "outcomes": json.dumps(self.outcomes, default=str),
            "outcome_label": self.outcome_label.value,
            "context_embedding": self.context_embedding.tolist() if self.context_embedding is not None else None,
            "tags": self.tags,
            "forgetting_curve": self.forgetting_curve.to_dict(),
            "replay_count": self.replay_count,
        }


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

@dataclass
class SemanticMemory:
    """Abstracted market knowledge stored as rules and patterns.

    Semantic memories are compressed generalizations derived from
    many episodic experiences. They represent "what generally works"
    rather than specific instances.

    Attributes:
        rule_id: Unique identifier.
        rule_type: Category of rule (e.g., 'regime_strategy', 'correlation').
        condition: Description of when this rule applies.
        conclusion: What the rule predicts or recommends.
        confidence: Confidence in this rule [0, 1].
        support_count: Number of episodes that contributed to this rule.
        last_updated: When this rule was last modified.
        exceptions: Known conditions where this rule fails.
        embedding: Dense vector representation.
    """

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_type: str = ""
    condition: str = ""
    conclusion: str = ""
    confidence: float = 0.5
    support_count: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exceptions: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None

    def strengthen(self, evidence_strength: float = 1.0) -> None:
        """Strengthen this rule based on supporting evidence.

        Args:
            evidence_strength: Strength of the supporting evidence [0, 1].
        """
        self.support_count += 1
        # Bayesian-style update: incrementally increase confidence
        alpha = 0.1 * evidence_strength
        self.confidence = min(1.0, self.confidence + alpha * (1.0 - self.confidence))
        self.last_updated = datetime.now(timezone.utc)

    def weaken(self, evidence_strength: float = 1.0) -> None:
        """Weaken this rule based on contradicting evidence.

        Args:
            evidence_strength: Strength of the contradicting evidence [0, 1].
        """
        alpha = 0.15 * evidence_strength
        self.confidence = max(0.0, self.confidence * (1.0 - alpha))
        self.last_updated = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "rule_id": self.rule_id,
            "rule_type": self.rule_type,
            "condition": self.condition,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "last_updated": self.last_updated.isoformat(),
            "exceptions": self.exceptions,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
        }


# ---------------------------------------------------------------------------
# Memory Index
# ---------------------------------------------------------------------------

class MemoryIndex:
    """Fast retrieval index for finding similar historical situations.

    Uses approximate nearest neighbor (ANN) search based on episode
    embeddings. For production, this would integrate with FAISS or
    similar libraries; here we provide a numpy/PyTorch fallback.

    Attributes:
        dimension: Embedding dimensionality.
        device: Compute device ('cuda' or 'cpu').
    """

    def __init__(self, dimension: int = 128, device: str = "auto") -> None:
        """Initialise the memory index.

        Args:
            dimension: Dimensionality of episode embeddings.
            device: Compute device for similarity computations.
        """
        self.dimension = dimension
        self._ids: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._type_map: Dict[str, MemoryType] = {}

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

    @property
    def size(self) -> int:
        """Number of entries in the index."""
        return len(self._ids)

    def add(self, memory_id: str, embedding: np.ndarray, memory_type: MemoryType) -> None:
        """Add a memory to the index.

        Args:
            memory_id: Unique identifier for the memory.
            embedding: Dense vector representation.
            memory_type: Whether this is episodic or semantic.
        """
        if embedding.shape[0] != self.dimension:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d",
                self.dimension,
                embedding.shape[0],
            )
            return

        self._ids.append(memory_id)
        self._embeddings.append(embedding)
        self._type_map[memory_id] = memory_type

    def remove(self, memory_id: str) -> bool:
        """Remove a memory from the index.

        Args:
            memory_id: ID of the memory to remove.

        Returns:
            True if the memory was found and removed.
        """
        if memory_id not in self._ids:
            return False
        idx = self._ids.index(memory_id)
        self._ids.pop(idx)
        self._embeddings.pop(idx)
        del self._type_map[memory_id]
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        memory_type: Optional[MemoryType] = None,
        min_similarity: float = 0.0,
    ) -> List[Tuple[str, float, MemoryType]]:
        """Find the most similar memories to a query embedding.

        Args:
            query_embedding: Query vector.
            top_k: Maximum number of results.
            memory_type: Filter by memory type.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of (memory_id, similarity, memory_type) tuples.
        """
        if not self._embeddings:
            return []

        if GPU_AVAILABLE and torch is not None and self._device == "cuda":
            q = torch.tensor(query_embedding, dtype=torch.float32, device="cuda")
            q = F.normalize(q, dim=0)
            mat = torch.tensor(
                np.stack(self._embeddings), dtype=torch.float32, device="cuda"
            )
            mat = F.normalize(mat, dim=1)
            sims = torch.matmul(mat, q).cpu().numpy()
        else:
            q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-12)
            mat = np.stack(self._embeddings)
            mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
            sims = mat_norm @ q_norm

        results: List[Tuple[str, float, MemoryType]] = []
        for i, (mid, sim_val) in enumerate(zip(self._ids, sims.tolist())):
            mtype = self._type_map.get(mid, MemoryType.EPISODIC)
            if memory_type and mtype != memory_type:
                continue
            if sim_val >= min_similarity:
                results.append((mid, sim_val, mtype))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Memory Consolidation
# ---------------------------------------------------------------------------

class MemoryConsolidation:
    """Periodic compression and reorganization of market memories.

    Inspired by sleep-dependent memory consolidation in neuroscience:
    - Old, low-retention episodic memories are compressed into semantic rules
    - Redundant memories are merged
    - The forgetting curve is applied to prune irrelevant memories

    Attributes:
        consolidation_threshold: Minimum retention to keep an episode.
        semantic_extraction_count: Minimum episodes needed to extract a rule.
    """

    def __init__(
        self,
        consolidation_threshold: float = 0.1,
        semantic_extraction_count: int = 5,
    ) -> None:
        """Initialise the consolidation engine.

        Args:
            consolidation_threshold: Episodes below this retention are candidates for consolidation.
            semantic_extraction_count: Minimum episodes required to extract a semantic rule.
        """
        self.consolidation_threshold = consolidation_threshold
        self.semantic_extraction_count = semantic_extraction_count

    def consolidate(
        self, episodes: List[Episode]
    ) -> Tuple[List[Episode], List[SemanticMemory], List[str]]:
        """Run consolidation on a list of episodic memories.

        Args:
            episodes: Current episodic memories.

        Returns:
            Tuple of:
                - Surviving episodes (above retention threshold or important)
                - Newly extracted semantic memories
                - IDs of removed episodes
        """
        surviving: List[Episode] = []
        consolidated_episodes: List[Episode] = []
        removed_ids: List[str] = []

        for ep in episodes:
            retention = ep.retention
            if retention >= self.consolidation_threshold:
                surviving.append(ep)
            else:
                # Keep high-value episodes regardless of retention
                if ep.outcome_label in (EpisodeOutcome.SUCCESS, EpisodeOutcome.FAILURE) and ep.pnl != 0:
                    surviving.append(ep)
                    ep.forgetting_curve.rehearse()  # Rehearse important memories
                else:
                    consolidated_episodes.append(ep)
                    removed_ids.append(ep.episode_id)

        # Extract semantic rules from consolidated episodes
        new_semantics = self._extract_semantic_rules(consolidated_episodes)

        logger.info(
            "Consolidation: %d surviving, %d removed, %d new semantic rules",
            len(surviving),
            len(removed_ids),
            len(new_semantics),
        )
        return surviving, new_semantics, removed_ids

    def _extract_semantic_rules(
        self, episodes: List[Episode]
    ) -> List[SemanticMemory]:
        """Extract semantic rules from a group of similar episodes.

        Groups episodes by regime and outcome, then creates rules
        for patterns with sufficient support.

        Args:
            episodes: Episodes to analyze.

        Returns:
            List of extracted SemanticMemory objects.
        """
        rules: List[SemanticMemory] = []

        # Group by regime + outcome
        groups: Dict[Tuple[str, str], List[Episode]] = {}
        for ep in episodes:
            key = (ep.regime.value, ep.outcome_label.value)
            groups.setdefault(key, []).append(ep)

        for (regime, outcome), group in groups.items():
            if len(group) < self.semantic_extraction_count:
                continue

            # Compute average outcomes
            avg_pnl = float(np.mean([e.pnl for e in group]))
            avg_sharpe = float(np.mean([e.sharpe for e in group]))
            success_rate = sum(
                1 for e in group if e.outcome_label == EpisodeOutcome.SUCCESS
            ) / len(group)

            rule = SemanticMemory(
                rule_type="regime_strategy",
                condition=f"regime={regime}",
                conclusion=f"avg_pnl={avg_pnl:.4f}, avg_sharpe={avg_sharpe:.2f}, success_rate={success_rate:.2f}",
                confidence=min(1.0, len(group) / 20.0 * success_rate),
                support_count=len(group),
            )
            rules.append(rule)

        return rules

    def merge_similar_episodes(
        self, episodes: List[Episode], similarity_threshold: float = 0.95
    ) -> List[Episode]:
        """Merge near-duplicate episodes to reduce memory footprint.

        Args:
            episodes: Episodes to deduplicate.
            similarity_threshold: Cosine similarity above which episodes are merged.

        Returns:
            Deduplicated list of episodes.
        """
        if len(episodes) <= 1:
            return episodes

        embeddings = []
        for ep in episodes:
            if ep.context_embedding is not None:
                embeddings.append(ep.context_embedding)
            else:
                embeddings.append(np.zeros(1))

        merged: List[Episode] = []
        used: set = set()

        for i, ep_i in enumerate(episodes):
            if i in used:
                continue
            group = [ep_i]
            used.add(i)

            for j in range(i + 1, len(episodes)):
                if j in used:
                    continue
                ep_j = episodes[j]
                if ep_i.regime != ep_j.regime:
                    continue
                if ep_i.context_embedding is not None and ep_j.context_embedding is not None:
                    sim = float(
                        np.dot(ep_i.context_embedding, ep_j.context_embedding)
                        / (
                            np.linalg.norm(ep_i.context_embedding)
                            * np.linalg.norm(ep_j.context_embedding)
                            + 1e-12
                        )
                    )
                    if sim >= similarity_threshold:
                        group.append(ep_j)
                        used.add(j)

            # Merge group into a single representative episode
            if len(group) == 1:
                merged.append(group[0])
            else:
                representative = deepcopy(group[0])
                representative.outcomes = {
                    k: float(np.mean([e.outcomes.get(k, 0.0) for e in group]))
                    for k in group[0].outcomes
                }
                representative.tags = list(
                    set(t for e in group for t in e.tags)
                )
                representative.replay_count = sum(e.replay_count for e in group)
                merged.append(representative)

        logger.debug("Merged %d episodes into %d", len(episodes), len(merged))
        return merged


# ---------------------------------------------------------------------------
# Market Memory (Main Interface)
# ---------------------------------------------------------------------------

class MarketMemory:
    """Dual-store market memory system with episodic and semantic components.

    Provides a unified interface for storing, retrieving, and
    consolidating market knowledge. Supports:
    - Storing complete trading episodes with context and outcomes
    - Extracting and maintaining abstract semantic rules
    - Fast similarity-based retrieval of relevant past experiences
    - Periodic consolidation to compress and reorganize memories
    - Memory replay for reinforcement learning and model training

    Attributes:
        memory_id: Unique identifier for this memory system.
        episodes: Episodic memory store.
        semantics: Semantic memory store.
        index: Fast retrieval index.
        consolidation: Memory consolidation engine.
    """

    def __init__(
        self,
        memory_id: str = "default",
        embedding_dim: int = 128,
        device: str = "auto",
        redis_client: Any = None,
        postgres_client: Any = None,
    ) -> None:
        """Initialise the market memory system.

        Args:
            memory_id: Unique identifier.
            embedding_dim: Dimensionality of context embeddings.
            device: Compute device ('auto', 'cuda', 'cpu').
            redis_client: Optional Redis client for caching.
            postgres_client: Optional Postgres client for persistence.
        """
        self.memory_id = memory_id
        self.embedding_dim = embedding_dim
        self._redis = redis_client
        self._postgres = postgres_client

        self.episodes: Dict[str, Episode] = {}
        self.semantics: Dict[str, SemanticMemory] = {}
        self.index = MemoryIndex(dimension=embedding_dim, device=device)
        self.consolidation = MemoryConsolidation()

        if device == "auto":
            self._device = "cuda" if GPU_AVAILABLE else "cpu"
        else:
            self._device = device

        logger.info(
            "MarketMemory initialised [id=%s, dim=%d, device=%s]",
            memory_id,
            embedding_dim,
            self._device,
        )

    # -- Episodic Memory Operations ----------------------------------------

    def store_episode(self, episode: Episode) -> str:
        """Store a new episodic memory.

        Args:
            episode: The Episode to store.

        Returns:
            The episode_id.
        """
        self.episodes[episode.episode_id] = episode

        if episode.context_embedding is not None:
            self.index.add(
                episode.episode_id,
                episode.context_embedding,
                MemoryType.EPISODIC,
            )

        logger.debug(
            "Stored episode: %s (regime=%s, outcome=%s)",
            episode.episode_id,
            episode.regime.value,
            episode.outcome_label.value,
        )
        return episode.episode_id

    def retrieve_similar_episodes(
        self,
        context_embedding: np.ndarray,
        top_k: int = 10,
        regime: Optional[RegimeType] = None,
        min_retention: float = 0.0,
    ) -> List[Tuple[Episode, float]]:
        """Retrieve episodes similar to the given market context.

        Args:
            context_embedding: Current market context embedding.
            top_k: Maximum number of episodes to return.
            regime: Optional regime filter.
            min_retention: Minimum retention threshold.

        Returns:
            List of (episode, similarity) pairs.
        """
        results = self.index.search(
            context_embedding,
            top_k=top_k * 3,  # Over-retrieve for filtering
            memory_type=MemoryType.EPISODIC,
        )

        filtered: List[Tuple[Episode, float]] = []
        for ep_id, similarity, _ in results:
            ep = self.episodes.get(ep_id)
            if ep is None:
                continue
            if regime and ep.regime != regime:
                continue
            if ep.retention < min_retention:
                continue
            filtered.append((ep, similarity))
            # Rehearse retrieved memories to slow forgetting
            ep.forgetting_curve.rehearse()
            if len(filtered) >= top_k:
                break

        return filtered

    # -- Semantic Memory Operations ----------------------------------------

    def store_semantic(self, semantic: SemanticMemory) -> str:
        """Store a semantic memory rule.

        Args:
            semantic: The SemanticMemory to store.

        Returns:
            The rule_id.
        """
        self.semantics[semantic.rule_id] = semantic

        if semantic.embedding is not None:
            self.index.add(
                semantic.rule_id,
                semantic.embedding,
                MemoryType.SEMANTIC,
            )

        logger.debug("Stored semantic rule: %s (type=%s)", semantic.rule_id, semantic.rule_type)
        return semantic.rule_id

    def query_semantics(
        self,
        condition: Optional[str] = None,
        rule_type: Optional[str] = None,
        min_confidence: float = 0.3,
    ) -> List[SemanticMemory]:
        """Query semantic rules by condition, type, and confidence.

        Args:
            condition: Substring match on condition text.
            rule_type: Filter by rule type.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of matching SemanticMemory objects.
        """
        results: List[SemanticMemory] = []
        for rule in self.semantics.values():
            if rule_type and rule.rule_type != rule_type:
                continue
            if condition and condition.lower() not in rule.condition.lower():
                continue
            if rule.confidence < min_confidence:
                continue
            results.append(rule)

        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def update_semantic_from_outcome(
        self, regime: RegimeType, strategy: str, outcome: EpisodeOutcome
    ) -> None:
        """Update semantic rules based on a new trading outcome.

        Args:
            regime: The market regime.
            strategy: The strategy used.
            outcome: The outcome achieved.
        """
        condition = f"regime={regime.value}"
        matching = self.query_semantics(condition=condition, rule_type="regime_strategy")

        if matching:
            for rule in matching:
                if outcome == EpisodeOutcome.SUCCESS:
                    rule.strengthen()
                elif outcome == EpisodeOutcome.FAILURE:
                    rule.weaken()
                    rule.exceptions.append(f"{strategy}:failed")
        else:
            # Create new rule if none exists
            rule = SemanticMemory(
                rule_type="regime_strategy",
                condition=condition,
                conclusion=f"strategy={strategy}",
                confidence=0.5,
                support_count=1,
            )
            if outcome == EpisodeOutcome.SUCCESS:
                rule.strengthen()
            elif outcome == EpisodeOutcome.FAILURE:
                rule.weaken()
            self.store_semantic(rule)

    # -- Consolidation -----------------------------------------------------

    def run_consolidation(self) -> Dict[str, int]:
        """Run memory consolidation to compress and reorganize.

        Returns:
            Dictionary with consolidation statistics.
        """
        # Step 1: Merge similar episodes
        episode_list = list(self.episodes.values())
        merged_episodes = self.consolidation.merge_similar_episodes(episode_list)

        # Step 2: Consolidate low-retention episodes
        surviving, new_semantics, removed_ids = self.consolidation.consolidate(
            merged_episodes
        )

        # Update stores
        for rid in removed_ids:
            self.episodes.pop(rid, None)
            self.index.remove(rid)

        self.episodes = {ep.episode_id: ep for ep in surviving}

        for rule in new_semantics:
            self.store_semantic(rule)

        stats = {
            "surviving_episodes": len(surviving),
            "removed_episodes": len(removed_ids),
            "new_semantic_rules": len(new_semantics),
            "total_semantic_rules": len(self.semantics),
        }

        logger.info("Consolidation complete: %s", stats)
        return stats

    # -- Memory Replay -----------------------------------------------------

    def replay_memories(
        self,
        n_episodes: int = 32,
        strategy: str = "prioritized",
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> List[Episode]:
        """Replay stored memories for model training.

        Supports prioritized replay where episodes with larger
        TD-errors (approximated by outcome magnitude) are sampled
        more frequently.

        Args:
            n_episodes: Number of episodes to sample.
            strategy: Replay strategy ('uniform', 'prioritized', 'recent').
            alpha: Prioritization exponent (0=uniform, 1=full prioritization).
            beta: Importance sampling correction exponent.

        Returns:
            List of sampled episodes.
        """
        episode_list = list(self.episodes.values())
        if not episode_list:
            return []

        n_samples = min(n_episodes, len(episode_list))

        if strategy == "uniform":
            indices = np.random.choice(len(episode_list), size=n_samples, replace=False)
            sampled = [episode_list[i] for i in indices]
        elif strategy == "recent":
            sorted_episodes = sorted(episode_list, key=lambda e: e.timestamp, reverse=True)
            sampled = sorted_episodes[:n_samples]
        elif strategy == "prioritized":
            # Priority based on outcome magnitude and retention
            priorities = np.array([
                (abs(e.pnl) + 1e-4) ** alpha * max(e.retention, 0.01)
                for e in episode_list
            ])
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(episode_list), size=n_samples, replace=False, p=probs)
            sampled = [episode_list[i] for i in indices]
        else:
            sampled = episode_list[:n_samples]

        # Mark as replayed
        for ep in sampled:
            ep.replay_count += 1
            ep.forgetting_curve.rehearse()

        return sampled

    # -- Statistics --------------------------------------------------------

    def memory_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the memory system.

        Returns:
            Dictionary of memory statistics.
        """
        regime_counts: Dict[str, int] = {}
        outcome_counts: Dict[str, int] = {}
        total_retention = 0.0

        for ep in self.episodes.values():
            regime_counts[ep.regime.value] = regime_counts.get(ep.regime.value, 0) + 1
            outcome_counts[ep.outcome_label.value] = outcome_counts.get(ep.outcome_label.value, 0) + 1
            total_retention += ep.retention

        avg_retention = total_retention / len(self.episodes) if self.episodes else 0.0
        avg_confidence = (
            float(np.mean([s.confidence for s in self.semantics.values()]))
            if self.semantics
            else 0.0
        )

        return {
            "memory_id": self.memory_id,
            "total_episodes": len(self.episodes),
            "total_semantic_rules": len(self.semantics),
            "index_size": self.index.size,
            "avg_episode_retention": round(avg_retention, 4),
            "avg_semantic_confidence": round(avg_confidence, 4),
            "regime_distribution": regime_counts,
            "outcome_distribution": outcome_counts,
            "device": self._device,
        }

    # -- Persistence -------------------------------------------------------

    async def save_to_postgres(self) -> int:
        """Persist memories to Postgres.

        Returns:
            Number of records saved.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping save")
            return 0

        count = 0
        try:
            async with self._postgres.transaction():
                for ep in self.episodes.values():
                    await self._postgres.execute(
                        """
                        INSERT INTO memory_episodes (episode_id, memory_id, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (episode_id) DO UPDATE SET data = $3
                        """,
                        ep.episode_id,
                        self.memory_id,
                        json.dumps(ep.to_dict()),
                    )
                    count += 1
                for rule in self.semantics.values():
                    await self._postgres.execute(
                        """
                        INSERT INTO memory_semantics (rule_id, memory_id, data)
                        VALUES ($1, $2, $3)
                        ON CONFLICT (rule_id) DO UPDATE SET data = $3
                        """,
                        rule.rule_id,
                        self.memory_id,
                        json.dumps(rule.to_dict()),
                    )
                    count += 1
            logger.info("Saved %d memory records to Postgres", count)
        except Exception as exc:
            logger.error("Failed to save memories to Postgres: %s", exc)
        return count

    async def load_from_postgres(self) -> int:
        """Load memories from Postgres.

        Returns:
            Number of records loaded.
        """
        if self._postgres is None:
            logger.warning("No Postgres client configured; skipping load")
            return 0

        count = 0
        try:
            rows = await self._postgres.fetch(
                "SELECT data FROM memory_episodes WHERE memory_id = $1", self.memory_id
            )
            for row in rows:
                data = json.loads(row["data"])
                ep = Episode(
                    episode_id=data["episode_id"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    regime=RegimeType(data["regime"]),
                    market_context=json.loads(data["market_context"]),
                    actions_taken=json.loads(data["actions_taken"]),
                    outcomes=json.loads(data["outcomes"]),
                    outcome_label=EpisodeOutcome(data["outcome_label"]),
                    context_embedding=np.array(data["context_embedding"]) if data.get("context_embedding") else None,
                    tags=data.get("tags", []),
                    replay_count=data.get("replay_count", 0),
                )
                self.episodes[ep.episode_id] = ep
                if ep.context_embedding is not None:
                    self.index.add(ep.episode_id, ep.context_embedding, MemoryType.EPISODIC)
                count += 1

            rows = await self._postgres.fetch(
                "SELECT data FROM memory_semantics WHERE memory_id = $1", self.memory_id
            )
            for row in rows:
                data = json.loads(row["data"])
                rule = SemanticMemory(
                    rule_id=data["rule_id"],
                    rule_type=data["rule_type"],
                    condition=data["condition"],
                    conclusion=data["conclusion"],
                    confidence=data["confidence"],
                    support_count=data["support_count"],
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                    exceptions=data.get("exceptions", []),
                    embedding=np.array(data["embedding"]) if data.get("embedding") else None,
                )
                self.semantics[rule.rule_id] = rule
                if rule.embedding is not None:
                    self.index.add(rule.rule_id, rule.embedding, MemoryType.SEMANTIC)
                count += 1

            logger.info("Loaded %d memory records from Postgres", count)
        except Exception as exc:
            logger.error("Failed to load memories from Postgres: %s", exc)
        return count
