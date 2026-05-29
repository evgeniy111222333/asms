"""
ACMS AI Models Subpackage
=========================

Comprehensive neural network architectures for the Algorithmic Crypto Management System.

Modules:
    temporal_fusion_transformer - Multi-horizon price forecasting with interpretable attention
    deep_rl_agents              - Deep reinforcement learning agents for execution optimization
    market_gnn                  - Graph neural networks for market structure analysis
    sentiment_nlp               - NLP and sentiment analysis for crypto markets
    meta_learner                - Meta-learning for fast adaptation to new market conditions
    self_supervised             - Self-supervised pretraining for market representations
    ensemble_orchestrator       - Dynamic model ensemble orchestration

All models support GPU training/inference via PyTorch with graceful CPU fallback.
"""

from acms.ai.models.temporal_fusion_transformer import (
    TFTConfig,
    GatedResidualNetwork,
    VariableSelectionNetwork,
    InterpretableMultiHeadAttention,
    StaticCovariateEncoder,
    TemporalCovariateEncoder,
    TemporalFusionTransformer,
)
from acms.ai.models.deep_rl_agents import (
    ReplayBuffer,
    PrioritizedReplayBuffer,
    OrnsteinUhlenbeckNoise,
    ActorNetwork,
    CriticNetwork,
    PPOAgent,
    SACAgent,
    TD3Agent,
    ActionMask,
    RewardShaper,
)
from acms.ai.models.market_gnn import (
    MarketGraph,
    GraphAttentionLayer,
    MarketGNNLayer,
    MarketGNN,
    RegimeDetector,
    ContagionRiskPredictor,
)
from acms.ai.models.sentiment_nlp import (
    CryptoSentimentModel,
    NewsArticleProcessor,
    SocialMediaSentimentAnalyzer,
    FearGreedIndexPredictor,
    SentimentAggregator,
    SentimentDataPoint,
    EventDetector,
)
from acms.ai.models.meta_learner import (
    MAML,
    Reptile,
    TaskSampler,
    MarketRegimeTask,
    MetaLearner,
)
from acms.ai.models.self_supervised import (
    ContrastiveLearning,
    MaskedAutoEncoder,
    TemporalContrastiveLoss,
    MarketDataAugmenter,
    SelfSupervisedPretrainer,
)
from acms.ai.models.ensemble_orchestrator import (
    DynamicWeightedEnsemble,
    AdaptiveEnsemble,
    StackingEnsemble,
    EnsembleDiversityTracker,
    ModelWrapper,
)

__all__ = [
    # Temporal Fusion Transformer
    "TFTConfig",
    "GatedResidualNetwork",
    "VariableSelectionNetwork",
    "InterpretableMultiHeadAttention",
    "StaticCovariateEncoder",
    "TemporalCovariateEncoder",
    "TemporalFusionTransformer",
    # Deep RL Agents
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "OrnsteinUhlenbeckNoise",
    "ActorNetwork",
    "CriticNetwork",
    "PPOAgent",
    "SACAgent",
    "TD3Agent",
    "ActionMask",
    "RewardShaper",
    # Market GNN
    "MarketGraph",
    "GraphAttentionLayer",
    "MarketGNNLayer",
    "MarketGNN",
    "RegimeDetector",
    "ContagionRiskPredictor",
    # Sentiment NLP
    "CryptoSentimentModel",
    "NewsArticleProcessor",
    "SocialMediaSentimentAnalyzer",
    "FearGreedIndexPredictor",
    "SentimentAggregator",
    "SentimentDataPoint",
    "EventDetector",
    # Meta Learner
    "MAML",
    "Reptile",
    "TaskSampler",
    "MarketRegimeTask",
    "MetaLearner",
    # Self Supervised
    "ContrastiveLearning",
    "MaskedAutoEncoder",
    "TemporalContrastiveLoss",
    "MarketDataAugmenter",
    "SelfSupervisedPretrainer",
    # Ensemble Orchestrator
    "DynamicWeightedEnsemble",
    "AdaptiveEnsemble",
    "StackingEnsemble",
    "EnsembleDiversityTracker",
    "ModelWrapper",
]
