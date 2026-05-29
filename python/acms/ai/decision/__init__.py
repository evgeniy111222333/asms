"""
ACMS AI Decision Module
=======================

AI-driven decision systems for the Algorithmic Crypto Management System.

This module provides:
- AIStrategySelector: Multi-armed bandit strategy selection with Thompson sampling
- AIPortfolioManager: Neural portfolio optimization with attention-based weighting
- AIRiskManager: Neural VaR, GAN stress testing, adaptive risk limits
- ModelExplainer: SHAP-based explainability and narrative generation

GPU-ready, PyTorch-based where applicable.
"""

from acms.ai.decision.strategy_selector import (
    AIStrategySelector,
    StrategyArm,
    ThompsonSampler,
    RegimeStrategyMapper,
    StrategyCombinationOptimizer,
)
from acms.ai.decision.portfolio_ai import (
    AIPortfolioManager,
    NeuralMarkowitzOptimizer,
    AttentionAssetWeighter,
    DynamicRiskBudgetAllocator,
    MultiObjectiveOptimizer,
    PortfolioStateEncoder,
    HierarchicalPortfolioDecider,
)
from acms.ai.decision.risk_ai import (
    AIRiskManager,
    NeuralVaR,
    StressScenarioGenerator,
    RealTimeRiskPredictor,
    AdaptiveRiskLimiter,
    TailRiskHedger,
    RiskExplainer,
)
from acms.ai.decision.explainer import (
    ModelExplainer,
    PredictionExplainer,
    FeatureAttribution,
    DecisionNarrativeGenerator,
    ExplanationDashboard,
    RegulatoryExplainer,
)

__all__ = [
    # Strategy Selection
    "AIStrategySelector",
    "StrategyArm",
    "ThompsonSampler",
    "RegimeStrategyMapper",
    "StrategyCombinationOptimizer",
    # Portfolio AI
    "AIPortfolioManager",
    "NeuralMarkowitzOptimizer",
    "AttentionAssetWeighter",
    "DynamicRiskBudgetAllocator",
    "MultiObjectiveOptimizer",
    "PortfolioStateEncoder",
    "HierarchicalPortfolioDecider",
    # Risk AI
    "AIRiskManager",
    "NeuralVaR",
    "StressScenarioGenerator",
    "RealTimeRiskPredictor",
    "AdaptiveRiskLimiter",
    "TailRiskHedger",
    "RiskExplainer",
    # Explainer
    "ModelExplainer",
    "PredictionExplainer",
    "FeatureAttribution",
    "DecisionNarrativeGenerator",
    "ExplanationDashboard",
    "RegulatoryExplainer",
]

__version__ = "0.1.0"
