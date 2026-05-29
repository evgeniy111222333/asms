"""Counterparty Risk Scoring for ACMS."""

from typing import Dict, Optional


class CounterpartyRiskScorer:
    """Counterparty risk scoring for exchange/counterparty assessment.

    Scores counterparties on multiple dimensions:
    - Exchange reliability
    - Regulatory compliance
    - Financial health (reserve proofs)
    - Operational stability (withdrawal status)
    """

    def __init__(self):
        """Initialize with default scores for known exchanges."""
        self._exchange_scores: Dict[str, Dict[str, float]] = {
            "binance": {"reliability": 85, "regulation": 65, "financial": 80, "operational": 85},
            "bybit": {"reliability": 75, "regulation": 55, "financial": 70, "operational": 75},
            "okx": {"reliability": 80, "regulation": 70, "financial": 75, "operational": 80},
            "coinbase": {"reliability": 90, "regulation": 90, "financial": 85, "operational": 90},
            "kraken": {"reliability": 85, "regulation": 80, "financial": 80, "operational": 85},
            "paper": {"reliability": 100, "regulation": 100, "financial": 100, "operational": 100},
        }

    def score_counterparty(self, exchange: str) -> Dict:
        """Compute composite counterparty risk score.

        Args:
            exchange: Exchange identifier.

        Returns:
            Dict with individual and composite scores, risk level, and warnings.
        """
        scores = self._exchange_scores.get(exchange, {
            "reliability": 50, "regulation": 50, "financial": 50, "operational": 50,
        })
        weights = {"reliability": 0.35, "regulation": 0.25, "financial": 0.25, "operational": 0.15}
        composite = sum(scores[k] * weights[k] for k in weights)
        risk_level = "low" if composite > 80 else "medium" if composite > 60 else "high"

        warnings = []
        for dim, score in scores.items():
            if score < 60:
                warnings.append(f"Low {dim} score: {score}")

        return {
            "exchange": exchange,
            "scores": scores,
            "composite_score": composite,
            "risk_level": risk_level,
            "warnings": warnings,
        }

    def update_score(self, exchange: str, dimension: str, score: float):
        """Update a specific score dimension for an exchange.

        Args:
            exchange: Exchange identifier.
            dimension: Score dimension name.
            score: New score value (0-100).
        """
        if exchange not in self._exchange_scores:
            self._exchange_scores[exchange] = {
                "reliability": 50, "regulation": 50, "financial": 50, "operational": 50
            }
        self._exchange_scores[exchange][dimension] = max(0, min(100, score))

    def update_from_reserve_proof(self, exchange: str, proof_ratio: float,
                                  last_proof_date: Optional[str] = None) -> Dict:
        """Update counterparty score based on reserve proof data.

        Args:
            exchange: Exchange identifier.
            proof_ratio: Ratio of reserves to liabilities (>1 is healthy).
            last_proof_date: Date of last reserve proof.

        Returns:
            Updated score dict.
        """
        if proof_ratio >= 1.5:
            financial_score = 95
        elif proof_ratio >= 1.2:
            financial_score = 80
        elif proof_ratio >= 1.0:
            financial_score = 60
        else:
            financial_score = 30

        self.update_score(exchange, "financial", financial_score)
        return self.score_counterparty(exchange)

    def update_from_withdrawal_status(self, exchange: str,
                                       withdrawals_normal: bool,
                                       delay_hours: float = 0) -> Dict:
        """Update counterparty score based on withdrawal status.

        Args:
            exchange: Exchange identifier.
            withdrawals_normal: Whether withdrawals are functioning normally.
            delay_hours: Average withdrawal delay in hours.

        Returns:
            Updated score dict.
        """
        if withdrawals_normal and delay_hours < 2:
            operational_score = 95
        elif withdrawals_normal and delay_hours < 12:
            operational_score = 70
        elif withdrawals_normal:
            operational_score = 50
        else:
            operational_score = 20

        self.update_score(exchange, "operational", operational_score)
        return self.score_counterparty(exchange)

__all__ = ['CounterpartyRiskScorer']
