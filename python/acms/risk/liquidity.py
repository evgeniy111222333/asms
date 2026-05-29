"""Liquidity Risk Assessment for ACMS."""

import numpy as np
from typing import Dict, List, Tuple


class LiquidityRiskAssessor:
    """Liquidity risk assessment.

    Monitors bid-ask spread widening, order book depth thinning,
    and market impact costs. Provides alerts when liquidity conditions
    deteriorate beyond acceptable thresholds.
    """

    def __init__(self, normal_spread_bps: float = 5.0, max_spread_bps: float = 50.0,
                 min_depth_usd: float = 10000.0):
        """Initialize liquidity risk assessor.

        Args:
            normal_spread_bps: Normal bid-ask spread in basis points.
            max_spread_bps: Maximum acceptable spread in basis points.
            min_depth_usd: Minimum acceptable order book depth in USD.
        """
        self.normal_spread_bps = normal_spread_bps
        self.max_spread_bps = max_spread_bps
        self.min_depth_usd = min_depth_usd
        self._spread_history: List[float] = []
        self._depth_history: List[Tuple[float, float]] = []

    def assess_spread_risk(self, current_spread_bps: float) -> Dict[str, float]:
        """Assess risk from bid-ask spread widening.

        Args:
            current_spread_bps: Current spread in basis points.

        Returns:
            Dict with spread risk metrics and alert level.
        """
        self._spread_history.append(current_spread_bps)
        spread_ratio = current_spread_bps / self.normal_spread_bps if self.normal_spread_bps > 0 else 1.0

        if spread_ratio > 5:
            risk_level = "critical"
        elif spread_ratio > 3:
            risk_level = "high"
        elif spread_ratio > 2:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Detect widening trend
        widening_trend = False
        if len(self._spread_history) >= 5:
            recent = self._spread_history[-5:]
            if all(recent[i] > recent[i-1] for i in range(1, len(recent))):
                widening_trend = True

        return {
            "current_spread_bps": current_spread_bps,
            "normal_spread_bps": self.normal_spread_bps,
            "spread_ratio": spread_ratio,
            "risk_level": risk_level,
            "slippage_estimate_bps": current_spread_bps * 0.5,
            "widening_trend_detected": widening_trend,
        }

    def assess_depth_risk(self, bid_depth_usd: float, ask_depth_usd: float,
                          order_size_usd: float) -> Dict[str, float]:
        """Assess risk from order book depth thinning.

        Args:
            bid_depth_usd: Total bid side depth in USD.
            ask_depth_usd: Total ask side depth in USD.
            order_size_usd: Intended order size in USD.

        Returns:
            Dict with depth risk metrics and thinning alert.
        """
        self._depth_history.append((bid_depth_usd, ask_depth_usd))
        min_depth = min(bid_depth_usd, ask_depth_usd)
        depth_ratio = min_depth / order_size_usd if order_size_usd > 0 else float('inf')
        fill_estimate = min(order_size_usd / (min_depth + 1e-10), 1.0)

        if min_depth < self.min_depth_usd:
            risk_level = "critical"
        elif depth_ratio < 2:
            risk_level = "high"
        elif depth_ratio < 5:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Detect depth thinning trend
        thinning_alert = False
        if len(self._depth_history) >= 5:
            recent_min_depths = [min(b, a) for b, a in self._depth_history[-5:]]
            if all(recent_min_depths[i] < recent_min_depths[i-1] for i in range(1, len(recent_min_depths))):
                thinning_alert = True

        return {
            "min_depth_usd": min_depth,
            "depth_ratio": depth_ratio,
            "fill_estimate": fill_estimate,
            "risk_level": risk_level,
            "depth_thinning_alert": thinning_alert,
            "imbalance_ratio": bid_depth_usd / (ask_depth_usd + 1e-10),
        }

    def compute_market_impact(self, order_size_usd: float, avg_daily_volume_usd: float,
                              alpha: float = 0.5) -> float:
        """Estimate market impact using square-root model.

        Impact = alpha * sigma * sqrt(order_size / daily_volume)

        Args:
            order_size_usd: Order size in USD.
            avg_daily_volume_usd: Average daily volume in USD.
            alpha: Impact coefficient (0.5 typical).

        Returns:
            Estimated market impact in basis points.
        """
        if avg_daily_volume_usd <= 0 or order_size_usd <= 0:
            return 0.0
        participation_rate = order_size_usd / avg_daily_volume_usd
        impact_bps = alpha * np.sqrt(participation_rate) * 10000
        return float(impact_bps)

__all__ = ['LiquidityRiskAssessor']
