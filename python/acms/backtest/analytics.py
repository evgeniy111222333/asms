"""Trade-level analytics and rolling metrics."""

from typing import Dict, List, Optional, Any
import numpy as np

from acms.core import Side


class TradeAnalytics:
    """Trade-level analytics: MAE, MFE, ETD.

    MAE (Maximum Adverse Excursion): Maximum unrealized loss during trade.
    MFE (Maximum Favorable Excursion): Maximum unrealized gain during trade.
    ETD (End Trade Drawdown): Difference between MFE and final PnL.
    """

    @staticmethod
    def compute_mae_mfe(entry_price: float, exit_price: float, side: Side,
                        highs_during: np.ndarray, lows_during: np.ndarray,
                        quantity: float) -> Dict[str, float]:
        """Compute MAE, MFE, ETD for a single trade.

        Args:
            entry_price: Trade entry price.
            exit_price: Trade exit price.
            side: Trade side (BUY/SELL).
            highs_during: High prices during the trade.
            lows_during: Low prices during the trade.
            quantity: Trade quantity.

        Returns:
            Dict with mae, mfe, etd.
        """
        if len(highs_during) == 0 or len(lows_during) == 0:
            return {"mae": 0.0, "mfe": 0.0, "etd": 0.0}

        if side == Side.BUY:
            best_price = np.max(highs_during)
            worst_price = np.min(lows_during)
            mfe = (best_price - entry_price) * quantity
            mae = (entry_price - worst_price) * quantity
            final_pnl = (exit_price - entry_price) * quantity
        else:
            best_price = np.min(lows_during)
            worst_price = np.max(highs_during)
            mfe = (entry_price - best_price) * quantity
            mae = (worst_price - entry_price) * quantity
            final_pnl = (entry_price - exit_price) * quantity

        etd = mfe - final_pnl

        return {"mae": float(mae), "mfe": float(mfe), "etd": float(etd)}



class RollingMetrics:
    """Rolling performance metrics computation.

    Computes rolling Sharpe, Sortino, and maximum drawdown
    over a specified window.
    """

    @staticmethod
    def rolling_sharpe(equity_curve: np.ndarray, window: int = 60,
                       annualization_factor: float = 525600) -> np.ndarray:
        """Compute rolling Sharpe ratio.

        Args:
            equity_curve: Equity curve array.
            window: Rolling window size in bars.
            annualization_factor: Bars per year for annualization.

        Returns:
            Array of rolling Sharpe ratios.
        """
        if len(equity_curve) < window + 1:
            return np.array([])

        # Safe returns - guard against zero values
        equity_safe = np.where(equity_curve[:-1] == 0, np.nan, equity_curve[:-1])
        returns = np.diff(equity_curve) / equity_safe
        returns = np.nan_to_num(returns, nan=0.0)

        n = len(returns)
        rolling = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_returns = returns[i - window + 1:i + 1]
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns, ddof=1)
            if std_ret > 0:
                rolling[i] = mean_ret / std_ret * np.sqrt(annualization_factor)

        return rolling

    @staticmethod
    def rolling_sortino(equity_curve: np.ndarray, window: int = 60,
                        annualization_factor: float = 525600) -> np.ndarray:
        """Compute rolling Sortino ratio.

        Args:
            equity_curve: Equity curve array.
            window: Rolling window size in bars.
            annualization_factor: Bars per year for annualization.

        Returns:
            Array of rolling Sortino ratios.
        """
        if len(equity_curve) < window + 1:
            return np.array([])

        # Safe returns - guard against zero values
        equity_safe = np.where(equity_curve[:-1] == 0, np.nan, equity_curve[:-1])
        returns = np.diff(equity_curve) / equity_safe
        returns = np.nan_to_num(returns, nan=0.0)

        n = len(returns)
        rolling = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_returns = returns[i - window + 1:i + 1]
            mean_ret = np.mean(window_returns)
            downside = window_returns[window_returns < 0]
            if len(downside) > 0 and np.std(downside, ddof=1) > 0:
                rolling[i] = mean_ret / np.std(downside, ddof=1) * np.sqrt(annualization_factor)

        return rolling

    @staticmethod
    def rolling_max_drawdown(equity_curve: np.ndarray, window: int = 60) -> np.ndarray:
        """Compute rolling maximum drawdown.

        Args:
            equity_curve: Equity curve array.
            window: Rolling window size in bars.

        Returns:
            Array of rolling max drawdowns.
        """
        if len(equity_curve) < window:
            return np.array([])

        n = len(equity_curve)
        rolling = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_equity = equity_curve[i - window + 1:i + 1]
            peak = np.maximum.accumulate(window_equity)
            # Guard against zero peak
            safe_peak = np.where(peak == 0, np.nan, peak)
            dd = (peak - window_equity) / safe_peak
            dd = np.nan_to_num(dd, nan=0.0)
            rolling[i] = np.max(dd)

        return rolling


__all__ = ["TradeAnalytics", "RollingMetrics"]
