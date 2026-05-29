"""
ACMS AI Knowledge Module
========================

Self-learning knowledge systems for the Algorithmic Crypto Management System.

This module provides:
- MarketKnowledgeGraph: Neo4j-like knowledge graph for market relationships
- MarketMemory: Episodic and semantic memory for market scenarios
- SelfAdaptationEngine: Self-adaptive learning and performance tracking

GPU-ready, PyTorch-based where applicable, with Redis/Postgres backing stores.
"""

from acms.ai.knowledge.graph import (
    MarketKnowledgeGraph,
    NodeType,
    EdgeType,
    GraphNode,
    GraphEdge,
    TemporalFact,
    PatternMiner,
)
from acms.ai.knowledge.memory import (
    MarketMemory,
    Episode,
    SemanticMemory,
    MemoryIndex,
    MemoryConsolidation,
    ForgettingCurve,
)
from acms.ai.knowledge.adaptation import (
    SelfAdaptationEngine,
    PerformanceTracker,
    AdaptationTrigger,
    AdaptationAction,
    AdaptationRecord,
    MetaAdaptation,
)

__all__ = [
    # Knowledge Graph
    "MarketKnowledgeGraph",
    "NodeType",
    "EdgeType",
    "GraphNode",
    "GraphEdge",
    "TemporalFact",
    "PatternMiner",
    # Memory
    "MarketMemory",
    "Episode",
    "SemanticMemory",
    "MemoryIndex",
    "MemoryConsolidation",
    "ForgettingCurve",
    # Adaptation
    "SelfAdaptationEngine",
    "PerformanceTracker",
    "AdaptationTrigger",
    "AdaptationAction",
    "AdaptationRecord",
    "MetaAdaptation",
]

__version__ = "0.1.0"
