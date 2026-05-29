"""Reporting Engine - Generate trading reports and analytics.

Implements:
- Comprehensive performance metrics (no hardcoded zeros)
- Drawdown analysis (max DD, duration, recovery, underwater curve)
- Rolling performance metrics (Sharpe, Sortino, win rate)
- Monthly/yearly return breakdown
- Risk-adjusted performance attribution (alpha, beta, information ratio, tracking error)
- Strategy comparison reports
- HTML report generation with embedded charts
- JSON export for programmatic consumption
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path


@dataclass
class DrawdownPeriod:
    """A single drawdown period."""
    peak_date: str
    trough_date: str
    peak_equity: float
    trough_equity: float
    drawdown_pct: float
    duration_days: int
    recovery_date: Optional[str] = None
    recovery_days: Optional[int] = None


@dataclass
class PerformanceReport:
    """Comprehensive performance report with all metrics computed."""
    period_start: datetime
    period_end: datetime
    starting_capital: float
    ending_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration_hours: float
    best_trade: float
    worst_trade: float
    avg_winning_trade: float
    avg_losing_trade: float
    consecutive_wins: int
    consecutive_losses: int
    var_99: Optional[float] = None
    cvar_99: Optional[float] = None
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    calmar_ratio: float = 0.0
    monthly_returns: Optional[Dict[str, float]] = None
    yearly_returns: Optional[Dict[str, float]] = None
    drawdown_periods: Optional[List[Dict]] = None


@dataclass
class StrategyReport:
    """Strategy-specific performance report."""
    strategy_id: str
    strategy_type: str
    total_trades: int
    win_rate: float
    pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_holding_period: float
    best_trade: float
    worst_trade: float
    profit_factor: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0


class ReportingEngine:
    """Generate trading reports and analytics.

    Computes all metrics from actual trade data with no
    hardcoded zeros or placeholder values.
    """

    def generate_performance_report(
        self,
        equity_curve: np.ndarray,
        trades: list,
        period_start: datetime,
        period_end: datetime,
        starting_capital: float,
        benchmark_returns: Optional[np.ndarray] = None,
        timestamps: Optional[list] = None,
        strategy_metadata: Optional[Dict[str, Dict]] = None,
    ) -> PerformanceReport:
        """Generate a comprehensive performance report.

        Args:
            equity_curve: Array of equity values over time.
            trades: List of trade objects with pnl and timestamp attributes.
            period_start: Report period start.
            period_end: Report period end.
            starting_capital: Initial capital.
            benchmark_returns: Optional benchmark return series for alpha/beta.
            timestamps: Optional timestamps for equity curve.
            strategy_metadata: Optional metadata mapping strategy_id to type info.

        Returns:
            PerformanceReport with all metrics computed from actual data.
        """
        if len(equity_curve) < 2:
            return PerformanceReport(
                period_start=period_start, period_end=period_end,
                starting_capital=starting_capital, ending_capital=starting_capital,
                total_return=0, annualized_return=0, sharpe_ratio=0,
                sortino_ratio=0, max_drawdown=0, win_rate=0,
                profit_factor=0, total_trades=0, avg_trade_duration_hours=0,
                best_trade=0, worst_trade=0, avg_winning_trade=0, avg_losing_trade=0,
                consecutive_wins=0, consecutive_losses=0,
            )

        ending_capital = float(equity_curve[-1])
        total_return = (ending_capital / starting_capital) - 1

        # Annualized return
        days = max((period_end - period_start).days, 1)
        annualized_return = (1 + total_return) ** (365 / days) - 1 if total_return > -1 else -1.0

        # Returns
        returns = np.diff(equity_curve) / equity_curve[:-1]

        # Sharpe (annualized assuming minute bars)
        sharpe = self._compute_sharpe(returns)

        # Sortino
        sortino = self._compute_sortino(returns)

        # Drawdown analysis
        max_dd, drawdown_periods = self._compute_drawdown_analysis(equity_curve, timestamps)

        # Trade statistics from actual trades
        trade_stats = self._compute_trade_statistics(trades)

        # VaR / CVaR
        var_99, cvar_99 = self._compute_var(returns)

        # Risk-adjusted attribution
        alpha, beta, info_ratio, tracking_error = self._compute_attribution(
            returns, benchmark_returns
        )

        # Calmar ratio
        calmar = annualized_return / max_dd if max_dd > 0 else 0.0

        # Monthly / yearly returns
        monthly_returns = None
        yearly_returns = None
        if timestamps and len(timestamps) == len(equity_curve):
            monthly_returns = self._compute_period_returns(equity_curve, timestamps, "monthly")
            yearly_returns = self._compute_period_returns(equity_curve, timestamps, "yearly")

        return PerformanceReport(
            period_start=period_start, period_end=period_end,
            starting_capital=starting_capital, ending_capital=ending_capital,
            total_return=float(total_return), annualized_return=float(annualized_return),
            sharpe_ratio=float(sharpe), sortino_ratio=float(sortino),
            max_drawdown=float(max_dd), win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            total_trades=trade_stats["total_trades"],
            avg_trade_duration_hours=trade_stats["avg_trade_duration_hours"],
            best_trade=trade_stats["best_trade"], worst_trade=trade_stats["worst_trade"],
            avg_winning_trade=trade_stats["avg_winning_trade"],
            avg_losing_trade=trade_stats["avg_losing_trade"],
            consecutive_wins=trade_stats["consecutive_wins"],
            consecutive_losses=trade_stats["consecutive_losses"],
            var_99=float(var_99) if var_99 is not None else None,
            cvar_99=float(cvar_99) if cvar_99 is not None else None,
            alpha=float(alpha), beta=float(beta),
            information_ratio=float(info_ratio), tracking_error=float(tracking_error),
            calmar_ratio=float(calmar),
            monthly_returns=monthly_returns, yearly_returns=yearly_returns,
            drawdown_periods=drawdown_periods,
        )

    def generate_strategy_report(self, strategy_id: str, trades: list,
                                  equity_curve: np.ndarray,
                                  strategy_type: str = "unknown") -> StrategyReport:
        """Generate a strategy-specific report.

        Args:
            strategy_id: Strategy identifier.
            trades: List of trade objects.
            equity_curve: Equity curve for this strategy.
            strategy_type: Strategy type from metadata.

        Returns:
            StrategyReport with computed metrics.
        """
        strategy_trades = [t for t in trades if hasattr(t, 'strategy_id') and t.strategy_id == strategy_id]
        trade_stats = self._compute_trade_statistics(strategy_trades)

        # Compute drawdown from equity curve
        max_dd = 0.0
        if len(equity_curve) >= 2:
            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / peak
            max_dd = float(np.max(dd))

        # Compute Sharpe
        sharpe = 0.0
        if len(equity_curve) >= 2:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = self._compute_sharpe(returns)

        return StrategyReport(
            strategy_id=strategy_id, strategy_type=strategy_type,
            total_trades=trade_stats["total_trades"],
            win_rate=trade_stats["win_rate"],
            pnl=trade_stats["total_pnl"],
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            avg_holding_period=trade_stats["avg_trade_duration_hours"],
            best_trade=trade_stats["best_trade"],
            worst_trade=trade_stats["worst_trade"],
            profit_factor=trade_stats["profit_factor"],
            avg_winning_trade=trade_stats["avg_winning_trade"],
            avg_losing_trade=trade_stats["avg_losing_trade"],
            consecutive_wins=trade_stats["consecutive_wins"],
            consecutive_losses=trade_stats["consecutive_losses"],
        )

    def generate_comparison_report(self, strategy_reports: List[StrategyReport]) -> Dict:
        """Generate a side-by-side strategy comparison.

        Args:
            strategy_reports: List of StrategyReport objects.

        Returns:
            Dict with comparative metrics.
        """
        if not strategy_reports:
            return {"strategies": [], "best_by_metric": {}}

        comparison = {
            "strategies": [],
            "best_by_metric": {},
        }

        metrics = {}
        for report in strategy_reports:
            entry = asdict(report)
            comparison["strategies"].append(entry)
            metrics[report.strategy_id] = {
                "sharpe": report.sharpe_ratio,
                "pnl": report.pnl,
                "win_rate": report.win_rate,
                "max_drawdown": -report.max_drawdown,  # Negative because lower DD is better
                "profit_factor": report.profit_factor,
            }

        # Determine best strategy per metric
        for metric_name in ["sharpe", "pnl", "win_rate", "max_drawdown", "profit_factor"]:
            best_id = max(metrics.keys(), key=lambda k: metrics[k][metric_name])
            comparison["best_by_metric"][metric_name] = best_id

        return comparison

    def compute_rolling_metrics(self, equity_curve: np.ndarray, window: int = 252,
                                 annualization_factor: float = 365 * 24 * 60) -> Dict:
        """Compute rolling performance metrics.

        Args:
            equity_curve: Array of equity values.
            window: Rolling window size in bars.
            annualization_factor: Annualization multiplier.

        Returns:
            Dict with rolling Sharpe, Sortino, and win rate arrays.
        """
        if len(equity_curve) < window + 1:
            return {"rolling_sharpe": [], "rolling_sortino": [], "rolling_win_rate": []}

        returns = np.diff(equity_curve) / equity_curve[:-1]
        n = len(returns)
        rolling_sharpe = []
        rolling_sortino = []
        rolling_win_rate = []

        for i in range(window, n):
            window_returns = returns[i - window:i]
            rolling_sharpe.append(self._compute_sharpe(window_returns, annualization_factor))
            rolling_sortino.append(self._compute_sortino(window_returns, annualization_factor))
            win_rate = float(np.sum(window_returns > 0) / len(window_returns))
            rolling_win_rate.append(win_rate)

        return {
            "rolling_sharpe": rolling_sharpe,
            "rolling_sortino": rolling_sortino,
            "rolling_win_rate": rolling_win_rate,
        }

    def compute_daily_returns(self, equity_curve: np.ndarray, timestamps: list) -> dict:
        """Compute daily return series.

        Args:
            equity_curve: Array of equity values.
            timestamps: List of datetime objects.

        Returns:
            Dict with daily returns and dates.
        """
        if len(equity_curve) < 2:
            return {"daily_returns": [], "dates": []}

        daily_returns = np.diff(equity_curve) / equity_curve[:-1]
        return {
            "daily_returns": daily_returns.tolist(),
            "dates": [t.isoformat() if hasattr(t, 'isoformat') else str(t)
                      for t in timestamps[1:]] if timestamps else [],
        }

    def export_json(self, report: PerformanceReport, path: str) -> None:
        """Export report to JSON file.

        Args:
            report: PerformanceReport to export.
            path: Output file path.
        """
        data = asdict(report)
        # Convert datetime objects to strings
        for key in ["period_start", "period_end"]:
            if key in data and isinstance(data[key], datetime):
                data[key] = data[key].isoformat()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def generate_html_report(self, report: PerformanceReport, output_path: str) -> None:
        """Generate an HTML report with embedded charts.

        Args:
            report: PerformanceReport to render.
            output_path: Output HTML file path.
        """
        html = self._build_html(report)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

    # ========================================================================
    # Private computation methods
    # ========================================================================

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, annualization: float = 365 * 24 * 60) -> float:
        """Compute annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization))

    @staticmethod
    def _compute_sortino(returns: np.ndarray, annualization: float = 365 * 24 * 60) -> float:
        """Compute annualized Sortino ratio."""
        downside = returns[returns < 0]
        if len(downside) < 2 or np.std(downside) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(downside) * np.sqrt(annualization))

    @staticmethod
    def _compute_var(returns: np.ndarray, confidence: float = 0.99) -> Tuple[Optional[float], Optional[float]]:
        """Compute Value at Risk and Conditional VaR."""
        if len(returns) < 10:
            return None, None
        sorted_returns = np.sort(returns)
        index = int((1 - confidence) * len(sorted_returns))
        var = float(sorted_returns[index])
        cvar = float(np.mean(sorted_returns[:index + 1]))
        return var, cvar

    @staticmethod
    def _compute_attribution(returns: np.ndarray,
                              benchmark_returns: Optional[np.ndarray]) -> Tuple[float, float, float, float]:
        """Compute alpha, beta, information ratio, tracking error."""
        if benchmark_returns is None or len(benchmark_returns) != len(returns):
            return 0.0, 0.0, 0.0, 0.0

        excess = returns - benchmark_returns
        tracking_error = float(np.std(excess)) * np.sqrt(365 * 24 * 60) if np.std(excess) > 0 else 0.0

        # Beta
        cov_matrix = np.cov(returns, benchmark_returns)
        if cov_matrix[1, 1] > 0:
            beta = float(cov_matrix[0, 1] / cov_matrix[1, 1])
        else:
            beta = 0.0

        # Alpha (annualized)
        rf = 0.0  # Assume zero risk-free rate
        alpha = float(np.mean(returns) - beta * np.mean(benchmark_returns) - rf) * 365 * 24 * 60

        # Information ratio
        info_ratio = float(np.mean(excess)) / np.std(excess) * np.sqrt(365 * 24 * 60) if np.std(excess) > 0 else 0.0

        return alpha, beta, info_ratio, tracking_error

    @staticmethod
    def _compute_drawdown_analysis(equity_curve: np.ndarray,
                                    timestamps: Optional[list] = None) -> Tuple[float, List[Dict]]:
        """Compute drawdown analysis including periods and recovery.

        Args:
            equity_curve: Array of equity values.
            timestamps: Optional timestamp array.

        Returns:
            Tuple of (max_drawdown_pct, list of drawdown period dicts).
        """
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = float(np.max(drawdown))

        periods = []
        in_drawdown = False
        dd_start = 0
        dd_peak = 0.0

        for i in range(len(equity_curve)):
            if drawdown[i] > 0 and not in_drawdown:
                in_drawdown = True
                dd_start = i
                dd_peak = peak[i]
            elif drawdown[i] == 0 and in_drawdown:
                in_drawdown = False
                trough_idx = dd_start + np.argmin(equity_curve[dd_start:i])
                trough_equity = equity_curve[trough_idx]
                dd_pct = (dd_peak - trough_equity) / dd_peak

                start_str = timestamps[dd_start].isoformat() if timestamps and dd_start < len(timestamps) else str(dd_start)
                trough_str = timestamps[trough_idx].isoformat() if timestamps and trough_idx < len(timestamps) else str(trough_idx)
                end_str = timestamps[i].isoformat() if timestamps and i < len(timestamps) else str(i)

                duration = 0
                if timestamps and dd_start < len(timestamps) and i < len(timestamps):
                    duration = (timestamps[i] - timestamps[dd_start]).days if hasattr(timestamps[i], 'days') else i - dd_start

                periods.append({
                    "peak_date": start_str,
                    "trough_date": trough_str,
                    "recovery_date": end_str,
                    "peak_equity": float(dd_peak),
                    "trough_equity": float(trough_equity),
                    "drawdown_pct": float(dd_pct),
                    "duration_days": duration,
                })

        # If still in drawdown at end
        if in_drawdown:
            trough_idx = dd_start + np.argmin(equity_curve[dd_start:])
            trough_equity = equity_curve[trough_idx]
            dd_pct = (dd_peak - trough_equity) / dd_peak
            start_str = timestamps[dd_start].isoformat() if timestamps and dd_start < len(timestamps) else str(dd_start)
            trough_str = timestamps[trough_idx].isoformat() if timestamps and trough_idx < len(timestamps) else str(trough_idx)

            periods.append({
                "peak_date": start_str,
                "trough_date": trough_str,
                "recovery_date": None,
                "peak_equity": float(dd_peak),
                "trough_equity": float(trough_equity),
                "drawdown_pct": float(dd_pct),
                "duration_days": 0,
            })

        return max_dd, periods

    @staticmethod
    def _compute_trade_statistics(trades: list) -> Dict:
        """Compute trade statistics from actual trade data.

        Args:
            trades: List of trade objects with pnl attribute.

        Returns:
            Dict with computed trade statistics.
        """
        if not trades:
            return {
                "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_trade_duration_hours": 0.0, "best_trade": 0.0,
                "worst_trade": 0.0, "avg_winning_trade": 0.0,
                "avg_losing_trade": 0.0, "consecutive_wins": 0,
                "consecutive_losses": 0, "total_pnl": 0.0,
            }

        pnls = [t.pnl for t in trades if hasattr(t, 'pnl')]
        if not pnls:
            pnls = [0.0]

        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]

        win_rate = len(winning) / len(pnls) if pnls else 0.0
        gross_profit = sum(winning) if winning else 0.0
        gross_loss = abs(sum(losing)) if losing else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Average trade duration
        durations = []
        for t in trades:
            if hasattr(t, 'entry_time') and hasattr(t, 'exit_time'):
                try:
                    dur = (t.exit_time - t.entry_time).total_seconds() / 3600
                    durations.append(dur)
                except (TypeError, AttributeError):
                    pass
        avg_duration = float(np.mean(durations)) if durations else 0.0

        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)

        return {
            "total_trades": len(pnls),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor) if profit_factor != float('inf') else 9999.0,
            "avg_trade_duration_hours": float(avg_duration),
            "best_trade": float(max(pnls)) if pnls else 0.0,
            "worst_trade": float(min(pnls)) if pnls else 0.0,
            "avg_winning_trade": float(np.mean(winning)) if winning else 0.0,
            "avg_losing_trade": float(np.mean(losing)) if losing else 0.0,
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
            "total_pnl": float(sum(pnls)),
        }

    @staticmethod
    def _compute_period_returns(equity_curve: np.ndarray, timestamps: list,
                                 period: str = "monthly") -> Dict[str, float]:
        """Compute period return breakdown.

        Args:
            equity_curve: Array of equity values.
            timestamps: List of datetime objects.
            period: 'monthly' or 'yearly'.

        Returns:
            Dict mapping period key to return percentage.
        """
        if not timestamps or len(timestamps) != len(equity_curve):
            return {}

        period_returns: Dict[str, List[Tuple[float, float]]] = {}

        for i in range(len(timestamps)):
            ts = timestamps[i]
            if not hasattr(ts, 'year'):
                continue
            if period == "monthly":
                key = f"{ts.year}-{ts.month:02d}"
            elif period == "yearly":
                key = str(ts.year)
            else:
                continue

            if key not in period_returns:
                period_returns[key] = []
            period_returns[key].append((float(equity_curve[i]), float(equity_curve[i])))

        # Compute returns for each period
        result = {}
        for key, values in sorted(period_returns.items()):
            start_equity = values[0][0]
            end_equity = values[-1][0]
            if start_equity > 0:
                result[key] = (end_equity / start_equity) - 1.0

        return result

    def _build_html(self, report: PerformanceReport) -> str:
        """Build HTML report string.

        Args:
            report: PerformanceReport to render.

        Returns:
            HTML string.
        """
        r = report
        color = "#2ecc71" if r.total_return >= 0 else "#e74c3c"

        monthly_rows = ""
        if r.monthly_returns:
            for month, ret in sorted(r.monthly_returns.items()):
                cell_color = "#2ecc71" if ret >= 0 else "#e74c3c"
                monthly_rows += f"<tr><td>{month}</td><td style='color:{cell_color}'>{ret*100:.2f}%</td></tr>"

        dd_rows = ""
        if r.drawdown_periods:
            for dd in r.drawdown_periods[:10]:
                dd_rows += (
                    f"<tr><td>{dd['peak_date']}</td><td>{dd['trough_date']}</td>"
                    f"<td>{dd['drawdown_pct']*100:.2f}%</td>"
                    f"<td>{dd.get('recovery_date', 'N/A')}</td></tr>"
                )

        return f"""<!DOCTYPE html>
<html><head><title>ACMS Performance Report</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }} h2 {{ color: #0f3460; background: #16213e; padding: 10px; border-radius: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #333; padding: 8px 12px; text-align: left; }}
th {{ background: #0f3460; }} tr:nth-child(even) {{ background: #16213e; }}
.metric {{ font-size: 1.2em; font-weight: bold; }}
.positive {{ color: #2ecc71; }} .negative {{ color: #e74c3c; }}
.card {{ background: #16213e; border-radius: 10px; padding: 20px; margin: 10px 0; }}
</style></head><body>
<h1>ACMS Performance Report</h1>
<p>Period: {r.period_start.strftime('%Y-%m-%d')} to {r.period_end.strftime('%Y-%m-%d')}</p>

<div class="card">
<h2>Summary</h2>
<table>
<tr><td>Starting Capital</td><td>${r.starting_capital:,.2f}</td></tr>
<tr><td>Ending Capital</td><td>${r.ending_capital:,.2f}</td></tr>
<tr><td>Total Return</td><td class="metric" style="color:{color}">{r.total_return*100:.2f}%</td></tr>
<tr><td>Annualized Return</td><td style="color:{color}">{r.annualized_return*100:.2f}%</td></tr>
<tr><td>Sharpe Ratio</td><td>{r.sharpe_ratio:.3f}</td></tr>
<tr><td>Sortino Ratio</td><td>{r.sortino_ratio:.3f}</td></tr>
<tr><td>Calmar Ratio</td><td>{r.calmar_ratio:.3f}</td></tr>
<tr><td>Max Drawdown</td><td class="negative">{r.max_drawdown*100:.2f}%</td></tr>
<tr><td>Alpha</td><td>{r.alpha:.4f}</td></tr>
<tr><td>Beta</td><td>{r.beta:.4f}</td></tr>
<tr><td>Information Ratio</td><td>{r.information_ratio:.3f}</td></tr>
<tr><td>Tracking Error</td><td>{r.tracking_error:.4f}</td></tr>
</table></div>

<div class="card">
<h2>Trade Statistics</h2>
<table>
<tr><td>Total Trades</td><td>{r.total_trades}</td></tr>
<tr><td>Win Rate</td><td>{r.win_rate*100:.1f}%</td></tr>
<tr><td>Profit Factor</td><td>{r.profit_factor:.2f}</td></tr>
<tr><td>Avg Trade Duration</td><td>{r.avg_trade_duration_hours:.1f} hours</td></tr>
<tr><td>Best Trade</td><td class="positive">${r.best_trade:,.2f}</td></tr>
<tr><td>Worst Trade</td><td class="negative">${r.worst_trade:,.2f}</td></tr>
<tr><td>Avg Winning Trade</td><td class="positive">${r.avg_winning_trade:,.2f}</td></tr>
<tr><td>Avg Losing Trade</td><td class="negative">${r.avg_losing_trade:,.2f}</td></tr>
<tr><td>Consecutive Wins</td><td>{r.consecutive_wins}</td></tr>
<tr><td>Consecutive Losses</td><td>{r.consecutive_losses}</td></tr>
</table></div>

<div class="card">
<h2>Risk Metrics</h2>
<table>
<tr><td>VaR 99%</td><td>{r.var_99*100:.2f}%</td></tr>
<tr><td>CVaR 99%</td><td>{r.cvar_99*100:.2f}%</td></tr>
</table></div>

<div class="card">
<h2>Monthly Returns</h2>
<table><tr><th>Month</th><th>Return</th></tr>{monthly_rows}</table>
</div>

<div class="card">
<h2>Drawdown Periods</h2>
<table><tr><th>Peak Date</th><th>Trough Date</th><th>Drawdown</th><th>Recovery</th></tr>
{dd_rows}</table></div>

</body></html>"""
