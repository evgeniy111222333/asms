"""Portfolio Engine - Portfolio optimization and management.

Implements:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Kelly Criterion allocation
- Black-Litterman
- Hierarchical Risk Parity (HRP) - Lopez de Prado
- Maximum Diversification Portfolio
- Minimum Correlation Algorithm
- CVaR Portfolio Optimization with linear programming
- Risk budgeting with CVaR constraints
- Dynamic rebalancing triggers (threshold, time, drift-based)
- Transaction cost modeling (fixed + proportional + market impact)
- Leverage optimization (Kelly + volatility targeting)
- Hedging strategies
- Portfolio reconciliation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from scipy.optimize import minimize, linprog
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from acms.core import Position, PortfolioSnapshot, Side


@dataclass
class PortfolioConfig:
    """Portfolio management configuration."""
    target_return: Optional[float] = None
    risk_free_rate: float = 0.0
    max_weight: float = 0.40
    min_weight: float = 0.0
    rebalance_threshold: float = 0.05
    transaction_cost_bps: float = 10.0
    max_leverage: float = 3.0
    # Transaction cost model parameters
    fixed_cost_usd: float = 1.0
    proportional_cost_bps: float = 5.0
    market_impact_alpha: float = 0.1
    # Rebalancing
    rebalance_interval_days: int = 30
    max_drift: float = 0.10


@dataclass
class TransactionCostModel:
    """Transaction cost model with fixed, proportional, and market impact components.

    Models three components of trading costs:
    - Fixed cost: constant per-trade cost
    - Proportional cost: percentage of trade notional
    - Market impact: square-root model based on participation rate
    """
    fixed_cost_usd: float = 1.0
    proportional_cost_bps: float = 5.0
    market_impact_alpha: float = 0.1
    avg_daily_volume_usd: float = 1000000.0

    def compute_cost(self, trade_notional: float, current_weights: np.ndarray,
                     target_weights: np.ndarray, portfolio_value: float) -> Dict:
        """Compute total transaction cost for a rebalance.

        Args:
            trade_notional: Total notional amount traded.
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights.
            portfolio_value: Total portfolio value.

        Returns:
            Dict with cost breakdown.
        """
        n_trades = int(np.sum(np.abs(target_weights - current_weights) > 0.001))
        fixed = self.fixed_cost_usd * n_trades
        proportional = trade_notional * self.proportional_cost_bps / 10000
        participation = trade_notional / self.avg_daily_volume_usd if self.avg_daily_volume_usd > 0 else 0
        market_impact = self.market_impact_alpha * np.sqrt(participation) * trade_notional
        total = fixed + proportional + market_impact

        return {
            "fixed_cost": float(fixed),
            "proportional_cost": float(proportional),
            "market_impact_cost": float(market_impact),
            "total_cost": float(total),
            "total_cost_bps": float(total / portfolio_value * 10000) if portfolio_value > 0 else 0.0,
            "n_trades": n_trades,
        }

    def cost_adjusted_weights(self, current_weights: np.ndarray,
                               target_weights: np.ndarray,
                               portfolio_value: float) -> np.ndarray:
        """Adjust target weights to account for transaction costs.

        Net target after costs. Ensures weights still sum to 1.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights before costs.
            portfolio_value: Total portfolio value.

        Returns:
            Adjusted target weights accounting for costs.
        """
        trade_notional = np.sum(np.abs(target_weights - current_weights)) * portfolio_value
        cost_info = self.compute_cost(trade_notional, current_weights, target_weights, portfolio_value)
        total_cost = cost_info["total_cost"]

        # Reduce all weights proportionally to account for cost
        cost_fraction = total_cost / portfolio_value if portfolio_value > 0 else 0
        adjusted = target_weights * (1 - cost_fraction)
        if adjusted.sum() > 0:
            adjusted /= adjusted.sum()
        return adjusted


class MeanVarianceOptimizer:
    """Mean-Variance Optimization (Markowitz).

    Finds optimal portfolio weights to maximize Sharpe ratio
    or minimize volatility for a target return.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()

    def optimize(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                 target_return: Optional[float] = None) -> dict:
        """Find optimal portfolio weights.

        Args:
            expected_returns: Array of expected returns per asset.
            cov_matrix: Covariance matrix of returns.
            target_return: Target portfolio return (None = max Sharpe).

        Returns:
            Dict with weights, expected_return, volatility, sharpe_ratio.
        """
        n = len(expected_returns)
        if n < 2:
            return {"weights": np.array([1.0]), "return": expected_returns[0] if len(expected_returns) > 0 else 0.0,
                    "volatility": 0.0, "sharpe_ratio": 0.0}

        target = target_return or self.config.target_return

        def portfolio_vol(weights):
            return np.sqrt(weights @ cov_matrix @ weights)

        def portfolio_ret(weights):
            return weights @ expected_returns

        def neg_sharpe(weights):
            ret = portfolio_ret(weights)
            vol = portfolio_vol(weights)
            if vol == 0:
                return 0.0
            return -(ret - self.config.risk_free_rate) / vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        if target is not None:
            constraints.append({"type": "eq", "fun": lambda w: portfolio_ret(w) - target})

        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            neg_sharpe if target is None else portfolio_vol,
            x0, method="SLSQP", bounds=bounds, constraints=constraints,
        )

        weights = result.x
        ret = portfolio_ret(weights)
        vol = portfolio_vol(weights)
        sharpe = (ret - self.config.risk_free_rate) / vol if vol > 0 else 0.0

        return {"weights": weights, "return": float(ret), "volatility": float(vol), "sharpe_ratio": float(sharpe)}

    def efficient_frontier(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                           num_points: int = 50) -> List[dict]:
        """Compute the efficient frontier.

        Args:
            expected_returns: Array of expected returns per asset.
            cov_matrix: Covariance matrix.
            num_points: Number of points on the frontier.

        Returns:
            List of dicts with portfolio metrics at each point.
        """
        min_ret = np.min(expected_returns)
        max_ret = np.max(expected_returns)
        targets = np.linspace(min_ret, max_ret, num_points)
        frontier = []
        for target in targets:
            result = self.optimize(expected_returns, cov_matrix, target)
            if result.get("volatility", float('inf')) < float('inf'):
                frontier.append(result)
        return frontier


class RiskParityOptimizer:
    """Risk Parity - equal risk contribution from each asset.

    Each asset contributes equally to total portfolio risk.
    """

    def optimize(self, cov_matrix: np.ndarray) -> dict:
        """Find risk parity weights.

        Args:
            cov_matrix: Covariance matrix of returns.

        Returns:
            Dict with weights and risk contributions.
        """
        n = cov_matrix.shape[0]
        if n < 2:
            return {"weights": np.array([1.0]), "risk_contributions": np.array([1.0])}

        def risk_contribution(weights):
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol == 0:
                return np.zeros(n)
            marginal = cov_matrix @ weights
            return weights * marginal / port_vol

        def objective(weights):
            rc = risk_contribution(weights)
            target_rc = np.mean(rc)
            return np.sum((rc - target_rc) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.99) for _ in range(n)]
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x
        rc = risk_contribution(weights)
        return {"weights": weights, "risk_contributions": rc}


class HierarchicalRiskParity:
    """Hierarchical Risk Parity (HRP) - Lopez de Prado algorithm.

    Addresses the instability of Markowitz optimization by:
    1. Clustering assets by correlation
    2. Allocating capital top-down through the dendrogram
    3. Using inverse-variance weighting within clusters

    Reference: Lopez de Prado, M. (2016) "Building Diversified Portfolios
    that Outperform Out of Sample"
    """

    def optimize(self, returns_matrix: np.ndarray) -> dict:
        """Compute HRP portfolio allocation.

        Args:
            returns_matrix: Matrix of returns (T x N).

        Returns:
            Dict with weights and linkage matrix.
        """
        if returns_matrix.shape[1] < 2:
            return {"weights": np.array([1.0]), "linkage": None}

        corr = np.corrcoef(returns_matrix.T)
        cov = np.cov(returns_matrix.T)
        n = corr.shape[0]

        corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        dist = squareform(1 - np.abs(corr), checks=False)
        if len(dist) == 0 or np.any(np.isnan(dist)):
            return {"weights": np.ones(n) / n, "linkage": None}

        link = linkage(dist, method='ward')

        sort_ix = self._get_quasi_diag(link)
        sort_ix = [i for i in sort_ix if i < n]

        weights = self._recursive_bisection(cov, sort_ix)
        return {"weights": weights, "linkage": link}

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> List[int]:
        """Extract sorted list of original items from linkage matrix.

        Performs quasi-diagonalization by traversing the dendrogram.

        Args:
            link: Scipy linkage matrix.

        Returns:
            List of original item indices in sorted order.
        """
        n = link.shape[0] + 1
        sort_ix = [int(link[-1, 0]), int(link[-1, 1])]

        max_id = 2 * n - 2
        while max(sort_ix) >= n:
            new_sort = []
            for i in sort_ix:
                if int(i) >= n:
                    idx = int(i) - n
                    if idx < link.shape[0]:
                        new_sort.append(int(link[idx, 0]))
                        new_sort.append(int(link[idx, 1]))
                    else:
                        new_sort.append(i)
                else:
                    new_sort.append(i)
            sort_ix = new_sort
            if max(sort_ix) < n:
                break
        return [int(i) for i in sort_ix]

    @staticmethod
    def _recursive_bisection(cov: np.ndarray, sort_ix: List[int]) -> np.ndarray:
        """Allocate weights through recursive bisection.

        Splits clusters and allocates inverse-variance weights
        to each sub-cluster.

        Args:
            cov: Covariance matrix.
            sort_ix: Quasi-diagonalized index list.

        Returns:
            Array of portfolio weights.
        """
        n = len(sort_ix)
        weights = np.ones(n)
        clusters = [sort_ix]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                cov_left = cov[np.ix_(left, left)]
                var_left = np.diag(cov_left)
                inv_var_left = 1.0 / (var_left + 1e-10)
                w_left = inv_var_left / np.sum(inv_var_left)
                v_left = w_left @ cov_left @ w_left

                cov_right = cov[np.ix_(right, right)]
                var_right = np.diag(cov_right)
                inv_var_right = 1.0 / (var_right + 1e-10)
                w_right = inv_var_right / np.sum(inv_var_right)
                v_right = w_right @ cov_right @ w_right

                alpha = 1.0 - v_left / (v_left + v_right) if (v_left + v_right) > 0 else 0.5

                for i in left:
                    weights[sort_ix.index(i)] *= alpha
                for i in right:
                    weights[sort_ix.index(i)] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)
            clusters = new_clusters

        return weights


class MaximumDiversificationPortfolio:
    """Maximum Diversification Portfolio.

    Maximizes the diversification ratio:
    DR = (w' * sigma) / sqrt(w' * Sigma * w)
    where sigma is the vector of asset volatilities.
    """

    def optimize(self, cov_matrix: np.ndarray) -> dict:
        """Find maximum diversification portfolio weights.

        Args:
            cov_matrix: Covariance matrix of returns.

        Returns:
            Dict with weights and diversification ratio.
        """
        n = cov_matrix.shape[0]
        if n < 2:
            return {"weights": np.array([1.0]), "diversification_ratio": 1.0}

        vols = np.sqrt(np.diag(cov_matrix))
        if np.any(vols == 0):
            return {"weights": np.ones(n) / n, "diversification_ratio": 1.0}

        def neg_div_ratio(weights):
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol == 0:
                return 0.0
            weighted_vol = weights @ vols
            return -(weighted_vol / port_vol)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.99) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(neg_div_ratio, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        dr = (weights @ vols) / port_vol if port_vol > 0 else 1.0

        return {"weights": weights, "diversification_ratio": float(dr)}


class MinimumCorrelationAlgorithm:
    """Minimum Correlation Algorithm.

    Heuristic approach that finds weights minimizing
    average portfolio correlation. Long-only constraint enforced.
    """

    def optimize(self, corr_matrix: np.ndarray) -> dict:
        """Find minimum correlation portfolio weights.

        Args:
            corr_matrix: Correlation matrix.

        Returns:
            Dict with weights and average correlation.
        """
        n = corr_matrix.shape[0]
        if n < 2:
            return {"weights": np.array([1.0]), "avg_correlation": 0.0}

        avg_corr = np.mean(np.abs(corr_matrix - np.eye(n)), axis=1)
        ranks = np.argsort(avg_corr)

        weights = np.zeros(n)
        for i, rank in enumerate(ranks):
            weights[rank] = (n - i) / (n * (n + 1) / 2)

        wc = weights @ np.abs(corr_matrix) @ weights
        avg = float(wc)

        return {"weights": weights, "avg_correlation": avg}


class CVaRPortfolioOptimization:
    """CVaR Portfolio Optimization using linear programming.

    Minimizes CVaR subject to return and weight constraints
    using the Rockafellar-Uryasev reformulation.

    Reference: Rockafellar & Uryasev (2000) "Optimization of
    Conditional Value-at-Risk"
    """

    def __init__(self, confidence: float = 0.95, min_return: Optional[float] = None,
                 max_weight: float = 0.40):
        """Initialize CVaR portfolio optimizer.

        Args:
            confidence: CVaR confidence level.
            min_return: Minimum required portfolio return.
            max_weight: Maximum weight per asset.
        """
        self.confidence = confidence
        self.min_return = min_return
        self.max_weight = max_weight

    def optimize(self, returns_matrix: np.ndarray) -> dict:
        """Optimize portfolio to minimize CVaR using linear programming.

        Uses the Rockafellar-Uryasev reformulation:
        minimize VaR + 1/((1-alpha)*T) * sum(max(loss_i - VaR, 0))

        Args:
            returns_matrix: Historical returns (T x N).

        Returns:
            Dict with optimal weights, CVaR, and VaR.
        """
        T, N = returns_matrix.shape
        if N < 2 or T < 50:
            return {"weights": np.ones(N) / N, "cvar": float('nan'), "var": float('nan')}

        alpha = self.confidence

        # Decision variables: [w_1, ..., w_N, VaR, z_1, ..., z_T]
        # Minimize: VaR + 1/((1-alpha)*T) * sum(z_i)
        c = np.zeros(N + 1 + T)
        c[N] = 1.0  # VaR coefficient
        c[N + 1:] = 1.0 / ((1 - alpha) * T)  # z_i coefficients

        # Inequality constraints: z_i >= -(returns_i @ w) - VaR
        # i.e., z_i + returns_i @ w + VaR >= 0
        # In standard form: -returns_i @ w - VaR - z_i <= 0
        A_ub = np.zeros((T, N + 1 + T))
        for t in range(T):
            A_ub[t, :N] = returns_matrix[t]  # returns_i @ w (loss = -return)
            A_ub[t, N] = 1.0  # VaR
            A_ub[t, N + 1 + t] = 1.0  # z_i
        b_ub = np.zeros(T)

        # Equality constraints: sum(w) = 1
        A_eq = np.zeros((1, N + 1 + T))
        A_eq[0, :N] = 1.0
        b_eq = np.array([1.0])

        # Optional return constraint
        if self.min_return is not None:
            mean_returns = np.mean(returns_matrix, axis=0)
            ret_row = np.zeros(N + 1 + T)
            ret_row[:N] = -mean_returns
            A_ub_return = ret_row.reshape(1, -1)
            b_ub_return = np.array([-self.min_return])
            A_ub = np.vstack([A_ub, A_ub_return])
            b_ub = np.concatenate([b_ub, b_ub_return])

        # Bounds
        bounds = [(0.01, self.max_weight)] * N + [(None, None)] + [(0, None)] * T

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if result.success:
                weights = result.x[:N]
                var_val = result.x[N]
                cvar_val = result.fun
                # Normalize weights
                if weights.sum() > 0:
                    weights /= weights.sum()
                return {
                    "weights": weights,
                    "cvar": float(cvar_val),
                    "var": float(var_val),
                    "success": True,
                }
        except Exception:
            pass

        # Fallback to equal weight
        return {"weights": np.ones(N) / N, "cvar": float('nan'), "var": float('nan'), "success": False}


class CVaRRiskBudgeting:
    """Risk budgeting with CVaR constraints.

    Allocates risk budget to each asset such that
    the CVaR contribution matches the target.
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def optimize(self, returns_matrix: np.ndarray,
                 risk_budget: Optional[np.ndarray] = None) -> dict:
        """Find CVaR risk budgeting weights.

        Args:
            returns_matrix: Historical returns (T x N).
            risk_budget: Target risk budget per asset (sums to 1).

        Returns:
            Dict with weights and CVaR contributions.
        """
        n = returns_matrix.shape[1]
        if n < 2:
            return {"weights": np.array([1.0]), "cvar_contributions": np.array([0.0])}

        if risk_budget is None:
            risk_budget = np.ones(n) / n

        alpha = self.confidence

        def portfolio_cvar(weights):
            port_returns = returns_matrix @ weights
            threshold = np.percentile(port_returns, (1 - alpha) * 100)
            tail = port_returns[port_returns <= threshold]
            if len(tail) == 0:
                return -threshold
            return -np.mean(tail)

        def cvar_contribution(weights):
            """Approximate CVaR contribution via marginal CVaR."""
            cvar = portfolio_cvar(weights)
            delta = 1e-5
            contributions = np.zeros(n)
            for i in range(n):
                w_up = weights.copy()
                w_up[i] += delta
                cvar_up = portfolio_cvar(w_up)
                contributions[i] = (cvar_up - cvar) / delta * weights[i]
            return contributions

        def objective(weights):
            contribs = cvar_contribution(weights)
            total = np.sum(np.abs(contribs))
            if total == 0:
                return 1e10
            pct_contrib = np.abs(contribs) / total
            return np.sum((pct_contrib - risk_budget) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.99) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = result.x
        contribs = cvar_contribution(weights)

        return {"weights": weights, "cvar_contributions": contribs}


class DynamicRebalancing:
    """Dynamic portfolio rebalancing triggers.

    Supports three rebalancing approaches:
    - Threshold-based: rebalance when any weight drifts beyond threshold
    - Time-based: rebalance at fixed intervals
    - Drift-based: rebalance based on cumulative portfolio drift
    """

    def __init__(self, threshold: float = 0.05, time_interval_days: int = 30,
                 max_drift: float = 0.10, transaction_cost_bps: float = 10.0):
        self.threshold = threshold
        self.time_interval_days = time_interval_days
        self.max_drift = max_drift
        self.transaction_cost_bps = transaction_cost_bps
        self._last_rebalance: Optional[datetime] = None

    def check_threshold_rebalance(self, current_weights: np.ndarray,
                                  target_weights: np.ndarray) -> bool:
        """Check if any weight has drifted beyond threshold.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights.

        Returns:
            True if rebalancing needed.
        """
        drift = np.abs(current_weights - target_weights)
        return bool(np.any(drift > self.threshold))

    def check_time_rebalance(self, current_time: datetime) -> bool:
        """Check if time-based rebalance is due.

        Args:
            current_time: Current timestamp.

        Returns:
            True if time interval has elapsed since last rebalance.
        """
        if self._last_rebalance is None:
            return True
        elapsed_days = (current_time - self._last_rebalance).days
        return elapsed_days >= self.time_interval_days

    def check_drift_rebalance(self, current_weights: np.ndarray,
                              target_weights: np.ndarray) -> bool:
        """Check if cumulative drift exceeds maximum.

        Args:
            current_weights: Current weights.
            target_weights: Target weights.

        Returns:
            True if total drift exceeds max_drift.
        """
        total_drift = np.sum(np.abs(current_weights - target_weights))
        return total_drift > self.max_drift

    def should_rebalance(self, current_weights: np.ndarray, target_weights: np.ndarray,
                         current_time: datetime) -> Dict:
        """Check all rebalancing triggers.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights.
            current_time: Current timestamp.

        Returns:
            Dict with rebalance decision and reason.
        """
        threshold_trigger = self.check_threshold_rebalance(current_weights, target_weights)
        time_trigger = self.check_time_rebalance(current_time)
        drift_trigger = self.check_drift_rebalance(current_weights, target_weights)

        should = threshold_trigger or time_trigger or drift_trigger
        reasons = []
        if threshold_trigger:
            reasons.append("threshold_breach")
        if time_trigger:
            reasons.append("time_interval")
        if drift_trigger:
            reasons.append("drift_exceeded")

        return {
            "should_rebalance": should,
            "reasons": reasons,
            "max_weight_drift": float(np.max(np.abs(current_weights - target_weights))),
            "total_drift": float(np.sum(np.abs(current_weights - target_weights))),
        }

    def compute_rebalance_cost(self, current_weights: np.ndarray,
                               target_weights: np.ndarray,
                               portfolio_value: float) -> Dict:
        """Compute transaction costs for rebalancing.

        Args:
            current_weights: Current weights.
            target_weights: Target weights.
            portfolio_value: Total portfolio value.

        Returns:
            Dict with cost estimates.
        """
        trades = np.abs(target_weights - current_weights)
        total_turnover = np.sum(trades) * portfolio_value
        cost = total_turnover * self.transaction_cost_bps / 10000
        return {
            "total_turnover": float(total_turnover),
            "transaction_cost": float(cost),
            "cost_bps": float(self.transaction_cost_bps),
        }


class LeverageOptimizer:
    """Leverage optimization based on Kelly and risk constraints.

    Finds the optimal leverage level based on:
    - Target volatility
    - Risk tolerance
    - Kelly criterion with drawdown constraint
    """

    def __init__(self, target_vol: float = 0.15, max_leverage: float = 3.0,
                 max_drawdown: float = 0.25):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.max_drawdown = max_drawdown

    def volatility_target_leverage(self, current_vol: float) -> float:
        """Compute leverage to achieve target volatility.

        leverage = target_vol / current_vol

        Args:
            current_vol: Current unlevered portfolio volatility.

        Returns:
            Optimal leverage factor (capped at max_leverage).
        """
        if current_vol <= 0:
            return 1.0
        leverage = self.target_vol / current_vol
        return max(1.0, min(leverage, self.max_leverage))

    def kelly_leverage(self, expected_return: float, volatility: float,
                       risk_free_rate: float = 0.0) -> float:
        """Compute Kelly-optimal leverage.

        Kelly leverage = (mu - rf) / sigma^2

        Args:
            expected_return: Expected portfolio return.
            volatility: Portfolio volatility.
            risk_free_rate: Risk-free rate.

        Returns:
            Kelly leverage (capped at max_leverage).
        """
        if volatility <= 0:
            return 1.0
        excess_return = expected_return - risk_free_rate
        leverage = excess_return / (volatility ** 2)
        return max(0.0, min(leverage, self.max_leverage))

    def optimal_leverage(self, expected_return: float, volatility: float,
                         risk_free_rate: float = 0.0,
                         win_rate: float = 0.5) -> Dict:
        """Compute optimal leverage combining Kelly and volatility targeting.

        Uses the more conservative of Kelly and volatility-targeted leverage,
        with additional drawdown constraint.

        Args:
            expected_return: Expected portfolio return.
            volatility: Portfolio volatility.
            risk_free_rate: Risk-free rate.
            win_rate: Strategy win rate.

        Returns:
            Dict with leverage recommendations.
        """
        kelly_lev = self.kelly_leverage(expected_return, volatility, risk_free_rate)
        vol_lev = self.volatility_target_leverage(volatility)

        # Apply half-Kelly for safety
        half_kelly = kelly_lev * 0.5

        # Drawdown constraint: max DD ≈ leverage^2 * sigma^2 / 2
        if volatility > 0:
            dd_constrained_lev = np.sqrt(2 * self.max_drawdown) / volatility
        else:
            dd_constrained_lev = self.max_leverage

        optimal = min(half_kelly, vol_lev, dd_constrained_lev, self.max_leverage)
        optimal = max(0.0, optimal)

        return {
            "optimal_leverage": float(optimal),
            "kelly_leverage": float(kelly_lev),
            "half_kelly": float(half_kelly),
            "vol_target_leverage": float(vol_lev),
            "drawdown_constrained_leverage": float(dd_constrained_lev),
        }


class KellyAllocator:
    """Kelly Criterion allocation across multiple assets."""

    def allocate(self, win_rates: np.ndarray, win_loss_ratios: np.ndarray,
                 capital: float, fraction: float = 0.5) -> dict:
        """Compute Kelly-optimal allocations.

        Args:
            win_rates: Win rate per asset.
            win_loss_ratios: Average win/loss ratio per asset.
            capital: Total capital.
            fraction: Fractional Kelly (0.5 = half-Kelly).

        Returns:
            Dict with weights and allocations.
        """
        n = len(win_rates)
        kelly_f = np.zeros(n)
        for i in range(n):
            if win_loss_ratios[i] > 0:
                kelly_f[i] = win_rates[i] - (1 - win_rates[i]) / win_loss_ratios[i]
                kelly_f[i] = max(kelly_f[i], 0.0)
        kelly_f *= fraction
        total = np.sum(kelly_f)
        if total > 1.0:
            kelly_f /= total
        allocations = kelly_f * capital
        return {"weights": kelly_f, "allocations": allocations}


class BlackLitterman:
    """Black-Litterman model for portfolio allocation.

    Combines market equilibrium with investor views to produce
    posterior expected returns and optimal weights.
    """

    def __init__(self, tau: float = 0.05):
        self.tau = tau

    def compute(self, market_weights: np.ndarray, cov_matrix: np.ndarray,
                risk_aversion: float = 2.5, views: Optional[np.ndarray] = None,
                view_confidence: Optional[np.ndarray] = None,
                view_returns: Optional[np.ndarray] = None) -> dict:
        """Compute Black-Litterman posterior returns and optimal weights.

        Args:
            market_weights: Market cap weights.
            cov_matrix: Covariance matrix.
            risk_aversion: Risk aversion parameter.
            views: P matrix (view matrix), shape (K, N).
            view_confidence: Omega diagonal (uncertainty in views).
            view_returns: Q vector (expected view returns).

        Returns:
            Dict with expected returns, posterior covariance, and weights.
        """
        pi = risk_aversion * cov_matrix @ market_weights

        if views is None or view_confidence is None or view_returns is None:
            return {"expected_returns": pi, "weights": market_weights}

        omega = np.diag(view_confidence)
        tau_sigma = self.tau * cov_matrix

        m1 = np.linalg.inv(tau_sigma)
        m2 = views.T @ np.linalg.inv(omega) @ views
        
        # M is the covariance of the posterior estimate of the mean
        m_matrix = np.linalg.inv(m1 + m2)
        # Posterior covariance of asset returns is cov_matrix (Sigma) + M
        posterior_cov = cov_matrix + m_matrix

        m3 = m1 @ pi
        m4 = views.T @ np.linalg.inv(omega) @ view_returns
        posterior_mean = m_matrix @ (m3 + m4)

        # Optimize weights using full posterior return covariance
        weights = np.linalg.inv(risk_aversion * posterior_cov) @ posterior_mean
        weights = np.maximum(weights, 0)
        weights /= np.sum(weights)

        return {"expected_returns": posterior_mean, "posterior_cov": posterior_cov, "weights": weights}


class PortfolioEngine:
    """Main portfolio management engine.

    Integrates all optimization methods and provides a unified
    interface for portfolio construction, rebalancing, and monitoring.
    """

    def __init__(self, config: Optional[PortfolioConfig] = None):
        self.config = config or PortfolioConfig()
        self.mv_optimizer = MeanVarianceOptimizer(self.config)
        self.rp_optimizer = RiskParityOptimizer()
        self.hrp_optimizer = HierarchicalRiskParity()
        self.max_div_optimizer = MaximumDiversificationPortfolio()
        self.min_corr_optimizer = MinimumCorrelationAlgorithm()
        self.cvar_optimizer = CVaRPortfolioOptimization()
        self.cvar_budget = CVaRRiskBudgeting()
        self.kelly_allocator = KellyAllocator()
        self.bl_model = BlackLitterman()
        self.rebalancing = DynamicRebalancing(
            self.config.rebalance_threshold,
            transaction_cost_bps=self.config.transaction_cost_bps,
        )
        self.leverage_optimizer = LeverageOptimizer(max_leverage=self.config.max_leverage)
        self.transaction_cost_model = TransactionCostModel(
            fixed_cost_usd=self.config.fixed_cost_usd,
            proportional_cost_bps=self.config.proportional_cost_bps,
            market_impact_alpha=self.config.market_impact_alpha,
        )

    def optimize_portfolio(self, method: str, expected_returns: np.ndarray,
                           cov_matrix: np.ndarray, **kwargs) -> dict:
        """Optimize portfolio allocation using specified method.

        Args:
            method: Optimization method name.
            expected_returns: Expected returns per asset.
            cov_matrix: Covariance matrix.
            **kwargs: Additional method-specific arguments.

        Returns:
            Dict with optimization results.

        Raises:
            ValueError: If method is unknown.
        """
        if method == "mean_variance":
            return self.mv_optimizer.optimize(expected_returns, cov_matrix, kwargs.get("target_return"))
        elif method == "risk_parity":
            return self.rp_optimizer.optimize(cov_matrix)
        elif method == "hrp":
            return self.hrp_optimizer.optimize(kwargs.get("returns_matrix", np.eye(len(expected_returns))))
        elif method == "max_diversification":
            return self.max_div_optimizer.optimize(cov_matrix)
        elif method == "min_correlation":
            return self.min_corr_optimizer.optimize(kwargs.get("corr_matrix", np.eye(len(expected_returns))))
        elif method == "cvar":
            return self.cvar_optimizer.optimize(kwargs.get("returns_matrix", np.eye(len(expected_returns))))
        elif method == "cvar_budget":
            return self.cvar_budget.optimize(kwargs.get("returns_matrix", np.eye(len(expected_returns))),
                                             kwargs.get("risk_budget"))
        elif method == "black_litterman":
            return self.bl_model.compute(
                kwargs.get("market_weights", np.ones(len(expected_returns)) / len(expected_returns)),
                cov_matrix, views=kwargs.get("views"), view_confidence=kwargs.get("view_confidence"),
                view_returns=kwargs.get("view_returns"),
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def compute_rebalance_trades(self, current_weights: np.ndarray,
                                 target_weights: np.ndarray, total_value: float,
                                 threshold: Optional[float] = None) -> List[dict]:
        """Compute trades needed to rebalance portfolio.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights.
            total_value: Total portfolio value.
            threshold: Minimum weight change to trigger a trade.

        Returns:
            List of trade dicts with asset index, weight change, and notional.
        """
        threshold = threshold or self.config.rebalance_threshold
        trades = []
        for i, (curr, target) in enumerate(zip(current_weights, target_weights)):
            diff = target - curr
            if abs(diff) > threshold:
                trades.append({
                    "asset_index": i, "weight_change": diff,
                    "notional_change": diff * total_value,
                    "action": "buy" if diff > 0 else "sell",
                })
        return trades

    def reconcile(self, expected: PortfolioSnapshot, actual: PortfolioSnapshot) -> dict:
        """Reconcile expected vs actual portfolio state.

        Args:
            expected: Expected portfolio snapshot.
            actual: Actual portfolio snapshot.

        Returns:
            Dict with reconciliation results and discrepancies.
        """
        discrepancies = []
        if abs(expected.total_value - actual.total_value) > 1.0:
            discrepancies.append({
                "field": "total_value",
                "expected": expected.total_value,
                "actual": actual.total_value,
                "diff": actual.total_value - expected.total_value,
            })
        for exp_pos in expected.positions:
            act_pos = next((p for p in actual.positions if p.symbol == exp_pos.symbol), None)
            if act_pos is None:
                discrepancies.append({"field": "missing_position", "symbol": exp_pos.symbol})
            elif abs(exp_pos.quantity - act_pos.quantity) > 1e-8:
                discrepancies.append({
                    "field": "quantity", "symbol": exp_pos.symbol,
                    "expected": exp_pos.quantity, "actual": act_pos.quantity,
                })
        return {
            "is_reconciled": len(discrepancies) == 0,
            "discrepancies": discrepancies,
            "timestamp": datetime.utcnow().isoformat(),
        }
