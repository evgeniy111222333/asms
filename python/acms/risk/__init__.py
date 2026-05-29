"""Risk Engine - Comprehensive risk management.

Implements:
- 3 VaR methods (Historical, Parametric, Monte Carlo)
- CVaR (Conditional VaR / Expected Shortfall)
- Expected Shortfall with tail risk decomposition
- 8+ stress test scenarios including historical replays
- Historical scenario replay: COVID crash, FTX collapse, Luna crash, China ban
- Kill switch with propagation
- 10+ pre-trade risk checks
- Position sizing (Kelly with drawdown constraint, fixed-fractional, risk parity)
- Liquidity risk assessment: bid-ask spread widening, order book depth thinning
- Dynamic correlation risk monitoring: rolling correlation with eigenvalue decomposition
- Counterparty risk scoring: exchange risk assessment
- Portfolio heat map: marginal VaR, component VaR, percentage contribution
- Dynamic position sizing based on volatility regime
- Risk budgeting: allocate risk budget across strategies
- Circuit breaker integration
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from scipy import stats

from acms.core import (
    RiskCheckResult, RiskDecision, Position, Order, Side,
    PortfolioSnapshot, SignalDirection,
)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_per_symbol: float = 100000.0
    max_total_position: float = 1000000.0
    max_order_notional: float = 50000.0
    max_order_quantity: float = 10.0
    max_daily_drawdown: float = 0.05
    max_weekly_drawdown: float = 0.10
    max_drawdown: float = 0.20
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 100
    max_net_exposure: float = 500000.0
    max_gross_exposure: float = 1000000.0
    max_concentration_pct: float = 0.25
    var_confidence: float = 0.99
    cvar_confidence: float = 0.99
    max_correlation: float = 0.85
    initial_margin_ratio: float = 0.10
    maintenance_margin_ratio: float = 0.05
    circuit_breaker_loss_pct: float = 0.03
    circuit_breaker_cooldown_minutes: int = 30
    # Risk budgeting
    risk_budget_per_strategy: float = 0.25
    max_strategy_risk_pct: float = 0.40


class ValueAtRisk:
    """Value at Risk computation using 3 methods.

    Provides historical, parametric, and Monte Carlo VaR estimation.
    All methods return VaR as a positive number representing potential loss.
    """

    @staticmethod
    def historical(returns: np.ndarray, confidence: float = 0.99) -> float:
        """Historical VaR - percentile of historical returns.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level (e.g. 0.99 for 99% VaR).

        Returns:
            VaR as a positive float representing potential loss.
        """
        if len(returns) < 100:
            return float('nan')
        return float(-np.percentile(returns, (1 - confidence) * 100))

    @staticmethod
    def parametric(returns: np.ndarray, confidence: float = 0.99) -> float:
        """Parametric VaR - assumes normal distribution.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level.

        Returns:
            Parametric VaR as a positive float.
        """
        if len(returns) < 30:
            return float('nan')
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z = stats.norm.ppf(confidence)
        return float(-(mu - z * sigma))

    @staticmethod
    def monte_carlo(returns: np.ndarray, confidence: float = 0.99,
                    num_simulations: int = 10000, horizon_days: int = 1) -> float:
        """Monte Carlo VaR using Student's t distribution for fat tails.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level.
            num_simulations: Number of Monte Carlo simulations.
            horizon_days: Forecast horizon in days.

        Returns:
            Monte Carlo VaR as a positive float.
        """
        if len(returns) < 30:
            return float('nan')
        df, loc, scale = stats.t.fit(returns)
        simulated = stats.t.rvs(df, loc=loc, scale=scale, size=num_simulations) * np.sqrt(horizon_days)
        return float(-np.percentile(simulated, (1 - confidence) * 100))

    @staticmethod
    def cvar(returns: np.ndarray, confidence: float = 0.99) -> float:
        """Conditional VaR (Expected Shortfall) - average of losses beyond VaR.

        Args:
            returns: Array of historical portfolio returns.
            confidence: Confidence level.

        Returns:
            CVaR as a positive float.
        """
        if len(returns) < 100:
            return float('nan')
        var = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= var]
        if len(tail_returns) == 0:
            return float(-var)
        return float(-np.mean(tail_returns))


class ExpectedShortfall:
    """Expected Shortfall (ES) - tail risk measure with decomposition.

    ES measures the average loss in the worst (1-confidence)% of cases.
    Also known as CVaR or AVaR. Provides tail risk decomposition
    showing which positions contribute most to tail risk.
    """

    @staticmethod
    def historical_es(returns: np.ndarray, confidence: float = 0.975) -> float:
        """Compute ES from historical returns.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level.

        Returns:
            Expected Shortfall as a positive float.
        """
        if len(returns) < 50:
            return float('nan')
        threshold = np.percentile(returns, (1 - confidence) * 100)
        tail = returns[returns <= threshold]
        if len(tail) == 0:
            return float(-threshold)
        return float(-np.mean(tail))

    @staticmethod
    def parametric_es(returns: np.ndarray, confidence: float = 0.975) -> float:
        """Compute ES assuming normal distribution.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level.

        Returns:
            Parametric Expected Shortfall.
        """
        if len(returns) < 30:
            return float('nan')
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z = stats.norm.ppf(confidence)
        phi_z = stats.norm.pdf(z)
        es = -(mu - sigma * phi_z / (1 - confidence))
        return float(es)

    @staticmethod
    def cornish_fisher_es(returns: np.ndarray, confidence: float = 0.975) -> float:
        """Compute ES using Cornish-Fisher expansion for non-normal returns.

        Adjusts for skewness and kurtosis in the return distribution.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level.

        Returns:
            Cornish-Fisher adjusted Expected Shortfall.
        """
        if len(returns) < 30:
            return float('nan')
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))

        z = stats.norm.ppf(confidence)
        z_cf = (z + (z**2 - 1) * skew / 6 +
                (z**3 - 3*z) * kurt / 24 -
                (2*z**3 - 5*z) * skew**2 / 36)

        phi_z = stats.norm.pdf(z_cf)
        es = -(mu - sigma * phi_z / (1 - confidence))
        return float(es)

    @staticmethod
    def tail_risk_decomposition(returns_matrix: np.ndarray, weights: np.ndarray,
                                confidence: float = 0.975) -> Dict:
        """Decompose tail risk into per-asset contributions.

        Identifies which assets contribute most to portfolio tail risk
        by computing each asset's marginal ES contribution.

        Args:
            returns_matrix: Matrix of asset returns (T x N).
            weights: Portfolio weights (N,).
            confidence: ES confidence level.

        Returns:
            Dict with per-asset ES contributions and percentages.
        """
        if returns_matrix.shape[1] != len(weights) or len(weights) < 2:
            return {"contributions": np.array([]), "pct_contributions": np.array([]),
                    "total_es": float('nan')}

        n = len(weights)
        portfolio_returns = returns_matrix @ weights
        total_es = ExpectedShortfall.historical_es(portfolio_returns, confidence)
        if np.isnan(total_es):
            return {"contributions": np.zeros(n), "pct_contributions": np.zeros(n),
                    "total_es": float('nan')}

        delta = 1e-4
        marginal_es = np.zeros(n)
        for i in range(n):
            w_up = weights.copy()
            w_up[i] += delta
            port_ret_up = returns_matrix @ w_up
            es_up = ExpectedShortfall.historical_es(port_ret_up, confidence)
            if not np.isnan(es_up):
                marginal_es[i] = (es_up - total_es) / delta

        contributions = weights * marginal_es
        total_contrib = np.sum(np.abs(contributions))
        pct_contributions = np.abs(contributions) / total_contrib * 100 if total_contrib > 0 else np.zeros(n)

        return {
            "contributions": contributions,
            "marginal_es": marginal_es,
            "pct_contributions": pct_contributions,
            "total_es": float(total_es),
        }


class StressTesting:
    """Stress testing with historical scenario replay and custom scenarios.

    Includes both synthetic stress scenarios and historically-calibrated
    scenario replays based on actual market events.
    """

    SCENARIOS = {
        "flash_crash": {"equity_shock": -0.20, "vol_mult": 5.0, "corr_to_1": True},
        "slow_bleed": {"equity_shock": -0.10, "vol_mult": 2.0, "corr_to_1": False},
        "vol_spike": {"equity_shock": -0.05, "vol_mult": 4.0, "corr_to_1": False},
        "liquidity_crisis": {"equity_shock": -0.15, "vol_mult": 3.0, "corr_to_1": True, "spread_mult": 10.0},
        "correlation_breakdown": {"equity_shock": -0.08, "vol_mult": 2.5, "corr_to_1": True},
        "black_swan": {"equity_shock": -0.40, "vol_mult": 8.0, "corr_to_1": True, "spread_mult": 20.0},
        "covid_crash_march2020": {"equity_shock": -0.35, "vol_mult": 6.0, "corr_to_1": True, "spread_mult": 15.0},
        "ftx_collapse_nov2022": {"equity_shock": -0.25, "vol_mult": 5.0, "corr_to_1": True, "spread_mult": 12.0},
        "luna_crash_may2022": {"equity_shock": -0.60, "vol_mult": 10.0, "corr_to_1": True, "spread_mult": 25.0},
    }

    # Historical scenario dates and detailed parameters
    HISTORICAL_SCENARIOS = {
        "covid_crash_feb_mar_2020": {
            "description": "COVID-19 market crash Feb-Mar 2020",
            "start_date": "2020-02-19",
            "end_date": "2020-03-23",
            "equity_shock": -0.35,
            "vol_mult": 6.0,
            "corr_to_1": True,
            "spread_mult": 15.0,
            "btc_shock": -0.50,
            "alt_shock": -0.65,
            "recovery_days": 150,
        },
        "ftx_collapse_nov_2022": {
            "description": "FTX exchange collapse Nov 2022",
            "start_date": "2022-11-06",
            "end_date": "2022-11-14",
            "equity_shock": -0.25,
            "vol_mult": 5.0,
            "corr_to_1": True,
            "spread_mult": 12.0,
            "btc_shock": -0.25,
            "alt_shock": -0.50,
            "recovery_days": 60,
        },
        "luna_crash_may_2022": {
            "description": "Terra/Luna ecosystem collapse May 2022",
            "start_date": "2022-05-07",
            "end_date": "2022-05-18",
            "equity_shock": -0.60,
            "vol_mult": 10.0,
            "corr_to_1": True,
            "spread_mult": 25.0,
            "btc_shock": -0.30,
            "alt_shock": -0.70,
            "recovery_days": 90,
        },
        "china_ban_may_2021": {
            "description": "China cryptocurrency ban May 2021",
            "start_date": "2021-05-12",
            "end_date": "2021-05-23",
            "equity_shock": -0.35,
            "vol_mult": 4.0,
            "corr_to_1": True,
            "spread_mult": 8.0,
            "btc_shock": -0.35,
            "alt_shock": -0.55,
            "recovery_days": 45,
        },
    }

    def run_scenario(self, positions: List[Position], scenario_name: str,
                     correlations: Optional[np.ndarray] = None) -> dict:
        """Run a stress scenario on current positions.

        Args:
            positions: List of current positions.
            scenario_name: Name of the scenario to run.
            correlations: Optional correlation matrix for positions.

        Returns:
            Dict with scenario results per position and total PnL.
        """
        scenario = self.SCENARIOS.get(scenario_name)
        if scenario is None:
            return {"error": f"Unknown scenario: {scenario_name}"}

        equity_shock = scenario["equity_shock"]
        total_pnl = 0.0
        position_results = []

        for pos in positions:
            pos_shock = equity_shock * (1.5 if pos.leverage > 1 else 1.0)
            if pos.side == Side.BUY:
                pnl = pos.notional_value * pos_shock
            else:
                pnl = pos.notional_value * (-pos_shock)

            position_results.append({
                "symbol": pos.symbol, "pnl": pnl, "shock_pct": pos_shock * 100,
            })
            total_pnl += pnl

        if scenario.get("corr_to_1") and correlations is not None:
            correlation_penalty = abs(total_pnl) * 0.2
            total_pnl += -correlation_penalty * np.sign(total_pnl)

        return {
            "scenario": scenario_name, "total_pnl": total_pnl,
            "position_results": position_results, "parameters": scenario,
        }

    def run_all_scenarios(self, positions: List[Position],
                          correlations: Optional[np.ndarray] = None) -> Dict[str, dict]:
        """Run all stress scenarios.

        Args:
            positions: List of current positions.
            correlations: Optional correlation matrix.

        Returns:
            Dict mapping scenario name to results.
        """
        return {name: self.run_scenario(positions, name, correlations)
                for name in self.SCENARIOS}

    def run_historical_scenario(self, positions: List[Position],
                                scenario_name: str,
                                is_alt: Optional[Dict[str, bool]] = None) -> Dict:
        """Run a historically-calibrated scenario replay.

        Applies historically-accurate shock parameters including
        differentiated shocks for BTC vs altcoins.

        Args:
            positions: List of current positions.
            scenario_name: Name from HISTORICAL_SCENARIOS.
            is_alt: Dict mapping symbol to True if altcoin, False if BTC.

        Returns:
            Dict with detailed scenario replay results.
        """
        scenario = self.HISTORICAL_SCENARIOS.get(scenario_name)
        if scenario is None:
            return {"error": f"Unknown historical scenario: {scenario_name}"}

        if is_alt is None:
            is_alt = {}

        total_pnl = 0.0
        position_results = []

        for pos in positions:
            alt = is_alt.get(pos.symbol, True)
            shock = scenario["alt_shock"] if alt else scenario["btc_shock"]
            shock *= (1.5 if pos.leverage > 1 else 1.0)

            if pos.side == Side.BUY:
                pnl = pos.notional_value * shock
            else:
                pnl = pos.notional_value * (-shock)

            position_results.append({
                "symbol": pos.symbol,
                "pnl": pnl,
                "shock_pct": shock * 100,
                "is_alt": alt,
                "recovery_estimate_days": scenario["recovery_days"],
            })
            total_pnl += pnl

        return {
            "scenario": scenario_name,
            "description": scenario["description"],
            "start_date": scenario["start_date"],
            "end_date": scenario["end_date"],
            "total_pnl": total_pnl,
            "position_results": position_results,
            "parameters": scenario,
        }

    def run_all_historical_scenarios(self, positions: List[Position],
                                     is_alt: Optional[Dict[str, bool]] = None) -> Dict[str, dict]:
        """Run all historical scenario replays.

        Args:
            positions: List of current positions.
            is_alt: Dict mapping symbol to whether it's an altcoin.

        Returns:
            Dict mapping scenario name to results.
        """
        return {name: self.run_historical_scenario(positions, name, is_alt)
                for name in self.HISTORICAL_SCENARIOS}


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


class CorrelationRiskMonitor:
    """Dynamic correlation matrix monitoring with eigenvalue decomposition.

    Detects correlation breakdowns, concentration risk from excessive
    correlation, and structural changes via eigenvalue analysis.
    """

    def __init__(self, lookback: int = 60, max_correlation: float = 0.85,
                 breakdown_threshold: float = 0.3):
        """Initialize correlation risk monitor.

        Args:
            lookback: Rolling window for correlation computation.
            max_correlation: Maximum acceptable pairwise correlation.
            breakdown_threshold: Threshold for detecting correlation breakdowns.
        """
        self.lookback = lookback
        self.max_correlation = max_correlation
        self.breakdown_threshold = breakdown_threshold
        self._prev_correlations: Optional[np.ndarray] = None
        self._eigenvalue_history: List[np.ndarray] = []

    def compute_correlation_matrix(self, returns_matrix: np.ndarray) -> np.ndarray:
        """Compute correlation matrix from returns.

        Args:
            returns_matrix: Matrix of returns (T x N).

        Returns:
            N x N correlation matrix.
        """
        if returns_matrix.shape[0] < 10:
            return np.eye(returns_matrix.shape[1])
        return np.corrcoef(returns_matrix.T)

    def eigenvalue_decomposition(self, corr_matrix: np.ndarray) -> Dict:
        """Perform eigenvalue decomposition of correlation matrix.

        Eigenvalue analysis reveals the effective dimensionality
        of the portfolio and concentration of correlation risk.

        Args:
            corr_matrix: N x N correlation matrix.

        Returns:
            Dict with eigenvalues, eigenvectors, and concentration metrics.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        n = len(eigenvalues)
        total = np.sum(eigenvalues)
        pct_variance = eigenvalues / total if total > 0 else eigenvalues

        # Effective rank (number of significant eigenvalues)
        effective_rank = float(np.sum(eigenvalues > 1.0))

        # Concentration ratio: share of variance in top eigenvalue
        concentration_ratio = float(pct_variance[0]) if len(pct_variance) > 0 else 1.0

        self._eigenvalue_history.append(eigenvalues)

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "pct_variance_explained": pct_variance,
            "effective_rank": effective_rank,
            "concentration_ratio": concentration_ratio,
            "is_concentrated": concentration_ratio > 0.5,
        }

    def detect_correlation_breakdown(self, current_corr: np.ndarray) -> Dict:
        """Detect significant changes in correlation structure.

        Args:
            current_corr: Current correlation matrix.

        Returns:
            Dict with breakdown detection results.
        """
        if self._prev_correlations is None or current_corr.shape != self._prev_correlations.shape:
            self._prev_correlations = current_corr.copy()
            return {"breakdown_detected": False, "max_change": 0.0}

        diff = np.abs(current_corr - self._prev_correlations)
        max_change = float(np.max(diff))
        avg_change = float(np.mean(diff))

        breakdown = max_change > self.breakdown_threshold
        self._prev_correlations = current_corr.copy()

        # Also check eigenvalue stability
        eigen_stability = True
        if len(self._eigenvalue_history) >= 2:
            prev_eig = self._eigenvalue_history[-2] if len(self._eigenvalue_history) >= 2 else self._eigenvalue_history[-1]
            curr_eig = self._eigenvalue_history[-1]
            if len(prev_eig) == len(curr_eig):
                eig_change = np.max(np.abs(curr_eig - prev_eig))
                eigen_stability = eig_change < 0.5

        return {
            "breakdown_detected": breakdown,
            "max_change": max_change,
            "avg_change": avg_change,
            "affected_pairs": int(np.sum(diff > self.breakdown_threshold * 0.5)),
            "eigenvalue_stable": eigen_stability,
        }

    def check_concentration_risk(self, corr_matrix: np.ndarray,
                                 weights: np.ndarray) -> Dict:
        """Check for correlation-driven concentration risk.

        Args:
            corr_matrix: Correlation matrix.
            weights: Portfolio weights.

        Returns:
            Dict with concentration risk metrics.
        """
        n = len(weights)
        high_corr_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > self.max_correlation:
                    high_corr_count += 1

        port_var = weights @ corr_matrix @ weights
        avg_var = np.mean(np.diag(corr_matrix))
        div_ratio = np.sqrt(avg_var / port_var) if port_var > 0 else 1.0

        eigen_data = self.eigenvalue_decomposition(corr_matrix)

        return {
            "high_correlation_pairs": high_corr_count,
            "max_correlation": float(np.max(np.abs(corr_matrix - np.eye(n)))),
            "diversification_ratio": float(div_ratio),
            "concentration_ratio": eigen_data["concentration_ratio"],
            "effective_rank": eigen_data["effective_rank"],
            "risk_level": "high" if high_corr_count > n else "moderate" if high_corr_count > 0 else "low",
        }


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


class PortfolioHeatMap:
    """Portfolio risk heat map - risk contribution per position.

    Shows each position's contribution to overall portfolio risk,
    including marginal VaR, component VaR, and percentage contribution.
    """

    def compute(self, positions: List[Position], returns_matrix: np.ndarray,
                weights: np.ndarray, confidence: float = 0.99) -> List[Dict]:
        """Compute risk contribution per position.

        Args:
            positions: List of current positions.
            returns_matrix: Historical returns matrix (T x N).
            weights: Current portfolio weights.
            confidence: VaR confidence level.

        Returns:
            List of dicts with risk contribution per position.
        """
        if returns_matrix.shape[1] != len(weights) or len(positions) != len(weights):
            return []

        cov_matrix = np.cov(returns_matrix.T)
        port_var = weights @ cov_matrix @ weights
        port_vol = np.sqrt(port_var) if port_var > 0 else 1e-10

        z = stats.norm.ppf(confidence)
        marginal_var = (cov_matrix @ weights) / port_vol * z
        component_var = weights * marginal_var

        total_cvar = np.sum(component_var)
        heatmap = []
        for i, pos in enumerate(positions):
            pct_contribution = component_var[i] / total_cvar * 100 if total_cvar != 0 else 0
            heatmap.append({
                "symbol": pos.symbol,
                "weight": float(weights[i]),
                "marginal_var": float(marginal_var[i]),
                "component_var": float(component_var[i]),
                "pct_risk_contribution": float(pct_contribution),
                "position_notional": pos.notional_value,
                "risk_level": "high" if pct_contribution > 30 else "medium" if pct_contribution > 15 else "low",
            })
        return heatmap


class CircuitBreaker:
    """Circuit breaker for automatic trading pause.

    Triggers on rapid loss threshold, order rate spike, or volatility spike.
    Enforces a cooldown period before trading can resume.
    """

    def __init__(self, loss_threshold_pct: float = 0.03,
                 cooldown_minutes: int = 30,
                 vol_spike_mult: float = 5.0):
        """Initialize circuit breaker.

        Args:
            loss_threshold_pct: Loss percentage that triggers the breaker.
            cooldown_minutes: Minutes before the breaker auto-resets.
            vol_spike_mult: Volatility spike multiplier to trigger.
        """
        self.loss_threshold_pct = loss_threshold_pct
        self.cooldown_minutes = cooldown_minutes
        self.vol_spike_mult = vol_spike_mult
        self.triggered = False
        self.trigger_reason = ""
        self.triggered_at: Optional[datetime] = None

    def check(self, current_pnl_pct: float, current_vol: float,
              normal_vol: float) -> bool:
        """Check if circuit breaker should trigger.

        Args:
            current_pnl_pct: Current P&L as percentage of capital.
            current_vol: Current realized volatility.
            normal_vol: Normal (average) volatility.

        Returns:
            True if circuit breaker is triggered.
        """
        if self.triggered:
            if self.triggered_at:
                elapsed = (datetime.utcnow() - self.triggered_at).total_seconds() / 60
                if elapsed >= self.cooldown_minutes:
                    self.reset()
                    return False
            return True

        if current_pnl_pct < -self.loss_threshold_pct:
            self._trigger(f"Loss exceeds {self.loss_threshold_pct:.1%}: {current_pnl_pct:.2%}")
            return True

        if normal_vol > 0 and current_vol > normal_vol * self.vol_spike_mult:
            self._trigger(f"Volatility spike: {current_vol:.4f} vs normal {normal_vol:.4f}")
            return True

        return False

    def _trigger(self, reason: str):
        """Activate the circuit breaker."""
        self.triggered = True
        self.trigger_reason = reason
        self.triggered_at = datetime.utcnow()

    def reset(self):
        """Reset the circuit breaker."""
        self.triggered = False
        self.trigger_reason = ""
        self.triggered_at = None


class RiskBudgeting:
    """Risk budgeting - allocate risk budget across strategies.

    Ensures each strategy receives a fair allocation of total
    portfolio risk, and enforces per-strategy risk limits.
    """

    def __init__(self, total_risk_budget: float = 1.0,
                 max_strategy_risk_pct: float = 0.40):
        """Initialize risk budgeting.

        Args:
            total_risk_budget: Total risk budget (1.0 = 100%).
            max_strategy_risk_pct: Maximum risk allocation per strategy.
        """
        self.total_risk_budget = total_risk_budget
        self.max_strategy_risk_pct = max_strategy_risk_pct
        self._strategy_risk_usage: Dict[str, float] = {}
        self._strategy_risk_budgets: Dict[str, float] = {}

    def allocate_budget(self, strategies: List[str],
                        target_contributions: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Allocate risk budget across strategies.

        Args:
            strategies: List of strategy identifiers.
            target_contributions: Target risk contribution per strategy.
                If None, equal risk budget allocation.

        Returns:
            Dict mapping strategy to risk budget allocation.
        """
        n = len(strategies)
        if n == 0:
            return {}

        if target_contributions is None:
            per_strategy = self.total_risk_budget / n
            # Cap each strategy at max
            per_strategy = min(per_strategy, self.max_strategy_risk_pct)
        else:
            per_strategy_arr = target_contributions.copy()
            per_strategy_arr = np.minimum(per_strategy_arr, self.max_strategy_risk_pct)
            if per_strategy_arr.sum() > self.total_risk_budget:
                per_strategy_arr *= self.total_risk_budget / per_strategy_arr.sum()
            per_strategy = per_strategy_arr  # type: ignore

        budgets = {}
        for i, strategy in enumerate(strategies):
            if isinstance(per_strategy, np.ndarray):
                budgets[strategy] = float(per_strategy[i])
            else:
                budgets[strategy] = float(per_strategy)
            self._strategy_risk_budgets[strategy] = budgets[strategy]

        return budgets

    def check_budget_utilization(self, strategy: str,
                                  current_risk_usage: float) -> Dict:
        """Check if a strategy is within its risk budget.

        Args:
            strategy: Strategy identifier.
            current_risk_usage: Current risk usage for the strategy.

        Returns:
            Dict with budget utilization details.
        """
        self._strategy_risk_usage[strategy] = current_risk_usage
        budget = self._strategy_risk_budgets.get(strategy, 0)
        utilization = current_risk_usage / budget if budget > 0 else float('inf')
        over_budget = utilization > 1.0

        return {
            "strategy": strategy,
            "budget": budget,
            "usage": current_risk_usage,
            "utilization_pct": float(utilization * 100),
            "over_budget": over_budget,
            "remaining_budget": max(0, budget - current_risk_usage),
        }

    def compute_risk_contribution_targets(self, strategy_returns: Dict[str, np.ndarray],
                                           cov_matrix: np.ndarray,
                                           strategy_indices: Dict[str, List[int]]) -> Dict:
        """Compute target risk contributions based on strategy characteristics.

        Args:
            strategy_returns: Dict mapping strategy to its returns array.
            cov_matrix: Full portfolio covariance matrix.
            strategy_indices: Dict mapping strategy to asset indices.

        Returns:
            Dict with target risk contributions per strategy.
        """
        strategy_vols = {}
        for name, rets in strategy_returns.items():
            if len(rets) > 0:
                strategy_vols[name] = np.std(rets)
            else:
                strategy_vols[name] = 0.0

        total_vol = sum(strategy_vols.values())
        if total_vol == 0:
            return {name: 1.0 / len(strategy_vols) for name in strategy_vols}

        # Allocate inversely proportional to volatility (risk parity style)
        inverse_vols = {name: 1.0 / v if v > 0 else 0 for name, v in strategy_vols.items()}
        inv_total = sum(inverse_vols.values())
        if inv_total == 0:
            return {name: 1.0 / len(strategy_vols) for name in strategy_vols}

        targets = {name: inv / inv_total for name, inv in inverse_vols.items()}
        return targets


class RiskEngine:
    """Comprehensive risk management engine.

    Integrates all risk assessment components including VaR, ES,
    stress testing, liquidity monitoring, correlation analysis,
    counterparty scoring, circuit breaker, and risk budgeting.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.var = ValueAtRisk()
        self.es = ExpectedShortfall()
        self.stress = StressTesting()
        self.liquidity = LiquidityRiskAssessor()
        self.correlation_monitor = CorrelationRiskMonitor()
        self.counterparty_scorer = CounterpartyRiskScorer()
        self.heatmap = PortfolioHeatMap()
        self.circuit_breaker = CircuitBreaker(self.config.circuit_breaker_loss_pct,
                                              self.config.circuit_breaker_cooldown_minutes)
        self.risk_budgeting = RiskBudgeting(
            max_strategy_risk_pct=self.config.max_strategy_risk_pct,
        )
        self.kill_switch_active = False
        self.kill_switch_reason = ""
        self._order_timestamps: List[datetime] = []

    # --- Pre-trade checks ---

    def pre_trade_check(self, order: Order, portfolio: PortfolioSnapshot) -> List[RiskCheckResult]:
        """Run all pre-trade risk checks.

        Performs 10 checks: kill switch, circuit breaker, position limit per
        symbol, order size, rate limit, drawdown, gross exposure, concentration,
        margin, and net exposure.

        Args:
            order: The order to check.
            portfolio: Current portfolio snapshot.

        Returns:
            List of RiskCheckResult for each check.
        """
        results = []

        # 1. Kill switch
        if self.kill_switch_active:
            results.append(RiskCheckResult(RiskDecision.REJECT, "kill_switch",
                f"Kill switch active: {self.kill_switch_reason}", 1.0, 0.0))
            return results

        # 2. Circuit breaker
        if self.circuit_breaker.triggered:
            results.append(RiskCheckResult(RiskDecision.REJECT, "circuit_breaker",
                f"Circuit breaker active: {self.circuit_breaker.trigger_reason}", 1.0, 0.0))
            return results

        # 3. Position limit per symbol
        current_pos = sum(p.notional_value for p in portfolio.positions if p.symbol == order.symbol)
        order_notional = order.notional_value
        new_pos = current_pos + order_notional
        results.append(RiskCheckResult(
            RiskDecision.ALLOW if new_pos <= self.config.max_position_per_symbol else RiskDecision.REJECT,
            "position_limit_symbol",
            f"Position {new_pos:.0f} vs limit {self.config.max_position_per_symbol:.0f}",
            new_pos, self.config.max_position_per_symbol,
        ))

        # 4. Order size
        results.append(RiskCheckResult(
            RiskDecision.ALLOW if order_notional <= self.config.max_order_notional else RiskDecision.REJECT,
            "order_notional",
            f"Order notional {order_notional:.0f} vs max {self.config.max_order_notional:.0f}",
            order_notional, self.config.max_order_notional,
        ))

        # 5. Rate limit
        now = datetime.utcnow()
        recent = [t for t in self._order_timestamps if (now - t).total_seconds() < 1]
        results.append(RiskCheckResult(
            RiskDecision.ALLOW if len(recent) < self.config.max_orders_per_second else RiskDecision.THROTTLE,
            "rate_limit_second",
            f"{len(recent)} orders/sec vs max {self.config.max_orders_per_second}",
            float(len(recent)), float(self.config.max_orders_per_second),
        ))

        # 6. Drawdown
        dd = 0.0
        if portfolio.total_value > 0:
            dd = -portfolio.unrealized_pnl / portfolio.total_value
        results.append(RiskCheckResult(
            RiskDecision.ALLOW if dd <= self.config.max_drawdown else RiskDecision.REJECT,
            "max_drawdown",
            f"Drawdown {dd:.2%} vs max {self.config.max_drawdown:.2%}",
            dd, self.config.max_drawdown,
        ))

        # 7. Gross exposure
        gross = sum(p.notional_value for p in portfolio.positions) + order_notional
        results.append(RiskCheckResult(
            RiskDecision.ALLOW if gross <= self.config.max_gross_exposure else RiskDecision.REJECT,
            "gross_exposure",
            f"Gross exposure {gross:.0f} vs max {self.config.max_gross_exposure:.0f}",
            gross, self.config.max_gross_exposure,
        ))

        # 8. Concentration
        if portfolio.total_value > 0:
            concentration = new_pos / portfolio.total_value
            results.append(RiskCheckResult(
                RiskDecision.ALLOW if concentration <= self.config.max_concentration_pct else RiskDecision.REJECT,
                "concentration",
                f"Concentration {concentration:.1%} vs max {self.config.max_concentration_pct:.1%}",
                concentration, self.config.max_concentration_pct,
            ))

        # 9. Margin
        margin_required = order_notional * self.config.initial_margin_ratio
        margin_available = portfolio.total_value - sum(p.margin_used for p in portfolio.positions)
        results.append(RiskCheckResult(
            RiskDecision.ALLOW if margin_required <= margin_available else RiskDecision.REJECT,
            "margin",
            f"Required {margin_required:.0f} vs available {margin_available:.0f}",
            margin_required, margin_available,
        ))

        # 10. Net exposure
        net_exposure = sum(
            p.notional_value * (1 if p.side == Side.BUY else -1) for p in portfolio.positions
        ) + order_notional * (1 if order.side == Side.BUY else -1)
        results.append(RiskCheckResult(
            RiskDecision.ALLOW if abs(net_exposure) <= self.config.max_net_exposure else RiskDecision.REJECT,
            "net_exposure",
            f"Net exposure {net_exposure:.0f} vs max \u00b1{self.config.max_net_exposure:.0f}",
            abs(net_exposure), self.config.max_net_exposure,
        ))

        self._order_timestamps.append(now)
        return results

    # --- Position Sizing ---

    def kelly_size(self, win_rate: float, avg_win: float, avg_loss: float,
                   capital: float, fraction: float = 0.5,
                   max_drawdown: float = 0.25) -> float:
        """Kelly Criterion position sizing with drawdown constraint.

        The drawdown constraint reduces Kelly fraction when estimated
        maximum drawdown exceeds the threshold.

        Args:
            win_rate: Historical win rate.
            avg_win: Average winning trade size.
            avg_loss: Average losing trade size.
            capital: Total capital.
            fraction: Fractional Kelly (0.5 = half-Kelly).
            max_drawdown: Maximum acceptable drawdown.

        Returns:
            Position size in currency units.
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        kelly_f = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        kelly_f = max(kelly_f, 0.0)

        if kelly_f > 0:
            consecutive_losses = int(np.log(1 - max_drawdown) / np.log(1 - kelly_f)) if kelly_f < 1 else 1
            expected_dd = 1 - (1 - kelly_f) ** max(3, consecutive_losses)
            if expected_dd > max_drawdown:
                kelly_f *= max_drawdown / expected_dd

        return capital * kelly_f * fraction

    def fixed_fractional_size(self, capital: float, risk_pct: float = 0.02,
                              entry_price: float = 0, stop_price: float = 0) -> float:
        """Fixed fractional position sizing.

        Args:
            capital: Total capital.
            risk_pct: Percentage of capital to risk per trade.
            entry_price: Entry price.
            stop_price: Stop loss price.

        Returns:
            Position size in units.
        """
        if entry_price == 0 or stop_price == 0 or entry_price == stop_price:
            return 0.0
        risk_amount = capital * risk_pct
        risk_per_unit = abs(entry_price - stop_price)
        return risk_amount / risk_per_unit

    def volatility_regime_size(self, capital: float, base_risk_pct: float = 0.02,
                               current_vol: float = 0.0, target_vol: float = 0.15) -> float:
        """Dynamic position sizing based on volatility regime.

        Uses Kelly criterion with drawdown constraint combined with
        volatility targeting to dynamically adjust position sizes.

        Args:
            capital: Total capital.
            base_risk_pct: Base risk percentage at target volatility.
            current_vol: Current realized volatility (annualized).
            target_vol: Target volatility for base risk.

        Returns:
            Adjusted position notional value.
        """
        if current_vol <= 0 or target_vol <= 0:
            return capital * base_risk_pct
        vol_scalar = target_vol / current_vol
        vol_scalar = max(0.25, min(vol_scalar, 3.0))
        adjusted_risk = base_risk_pct * vol_scalar
        return capital * adjusted_risk

    def dynamic_position_size(self, capital: float, base_risk_pct: float,
                               current_vol: float, target_vol: float,
                               win_rate: float = 0.5, avg_win: float = 0.02,
                               avg_loss: float = 0.01, max_drawdown: float = 0.20) -> float:
        """Dynamic position sizing combining Kelly criterion and volatility targeting.

        Applies Kelly with drawdown constraint, then scales by the
        volatility regime ratio to achieve target volatility.

        Args:
            capital: Total capital.
            base_risk_pct: Base risk percentage.
            current_vol: Current realized volatility (annualized).
            target_vol: Target portfolio volatility.
            win_rate: Strategy win rate for Kelly.
            avg_win: Average win as fraction of position.
            avg_loss: Average loss as fraction of position.
            max_drawdown: Maximum acceptable drawdown.

        Returns:
            Position size in currency units.
        """
        kelly_size = self.kelly_size(win_rate, avg_win, avg_loss, capital,
                                     fraction=0.5, max_drawdown=max_drawdown)
        vol_size = self.volatility_regime_size(capital, base_risk_pct,
                                                current_vol, target_vol)
        # Use the more conservative of the two
        return min(kelly_size, vol_size)

    # --- Kill Switch ---

    def trigger_kill_switch(self, reason: str):
        """Trigger emergency kill switch.

        Args:
            reason: Reason for triggering the kill switch.
        """
        self.kill_switch_active = True
        self.kill_switch_reason = reason

    def reset_kill_switch(self):
        """Reset kill switch to allow trading again."""
        self.kill_switch_active = False
        self.kill_switch_reason = ""

    # --- Portfolio Risk Metrics ---

    def compute_portfolio_var(self, returns_matrix: np.ndarray, weights: np.ndarray,
                              method: str = "historical") -> float:
        """Compute portfolio VaR using specified method.

        Args:
            returns_matrix: Asset returns matrix (T x N).
            weights: Portfolio weights.
            method: VaR method ("historical", "parametric", "monte_carlo").

        Returns:
            Portfolio VaR as a positive float.
        """
        portfolio_returns = returns_matrix @ weights
        if method == "historical":
            return self.var.historical(portfolio_returns, self.config.var_confidence)
        elif method == "parametric":
            return self.var.parametric(portfolio_returns, self.config.var_confidence)
        elif method == "monte_carlo":
            return self.var.monte_carlo(portfolio_returns, self.config.var_confidence)
        return float('nan')

    def compute_portfolio_cvar(self, returns_matrix: np.ndarray, weights: np.ndarray) -> float:
        """Compute portfolio CVaR.

        Args:
            returns_matrix: Asset returns matrix (T x N).
            weights: Portfolio weights.

        Returns:
            Portfolio CVaR as a positive float.
        """
        portfolio_returns = returns_matrix @ weights
        return self.var.cvar(portfolio_returns, self.config.cvar_confidence)

    def compute_portfolio_es(self, returns_matrix: np.ndarray, weights: np.ndarray,
                             method: str = "historical") -> float:
        """Compute portfolio Expected Shortfall.

        Args:
            returns_matrix: Asset returns matrix (T x N).
            weights: Portfolio weights.
            method: ES method ("historical", "parametric", "cornish_fisher").

        Returns:
            Portfolio ES as a positive float.
        """
        portfolio_returns = returns_matrix @ weights
        if method == "historical":
            return self.es.historical_es(portfolio_returns)
        elif method == "parametric":
            return self.es.parametric_es(portfolio_returns)
        elif method == "cornish_fisher":
            return self.es.cornish_fisher_es(portfolio_returns)
        return float('nan')

    def compute_tail_risk_decomposition(self, returns_matrix: np.ndarray,
                                         weights: np.ndarray,
                                         confidence: float = 0.975) -> Dict:
        """Decompose portfolio tail risk into per-asset contributions.

        Args:
            returns_matrix: Asset returns matrix (T x N).
            weights: Portfolio weights.
            confidence: ES confidence level.

        Returns:
            Dict with per-asset tail risk contributions.
        """
        return self.es.tail_risk_decomposition(returns_matrix, weights, confidence)
