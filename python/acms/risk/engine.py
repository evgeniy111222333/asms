"""Risk Engine for ACMS."""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from scipy import stats

from acms.core import (
    RiskCheckResult, RiskDecision, Position, Order, Side,
    PortfolioSnapshot, SignalDirection,
)
from acms.risk.config import RiskConfig
from acms.risk.var import ValueAtRisk
from acms.risk.expected_shortfall import ExpectedShortfall
from acms.risk.stress_testing import StressTesting
from acms.risk.liquidity import LiquidityRiskAssessor
from acms.risk.correlation import CorrelationRiskMonitor
from acms.risk.counterparty import CounterpartyRiskScorer
from acms.risk.heatmap import PortfolioHeatMap
from acms.risk.circuit_breaker import CircuitBreaker
from acms.risk.budgeting import RiskBudgeting
from acms.risk.position_sizing import kelly_size, fixed_fractional_size, volatility_regime_size


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

__all__ = ['RiskEngine']
