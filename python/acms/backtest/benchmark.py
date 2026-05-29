"""Benchmark comparison and analysis tools."""

import numpy as np
from typing import Optional, List, Dict

from acms.core import Candle, Side
from acms.strategies import Strategy


class BenchmarkComparison:
    """Benchmark comparison for backtest results.

    Computes performance of simple benchmarks:
    - Buy-and-hold: hold the first asset from start to end
    - Equal-weight: equal allocation across all assets
    - Best single asset: whichever asset performed best
    """

    @staticmethod
    def compute_benchmarks(candles: List[Candle],
                           multi_asset_candles: Optional[Dict[str, List[Candle]]] = None) -> Dict:
        """Compute benchmark returns for comparison.

        Args:
            candles: Primary symbol candles.
            multi_asset_candles: Dict of symbol -> candles for multi-asset benchmarks.

        Returns:
            Dict with benchmark returns.
        """
        buy_hold = 0.0
        if len(candles) > 0 and candles[0].close > 0:
            buy_hold = (candles[-1].close / candles[0].close - 1)

        equal_weight = buy_hold  # Default to buy_hold for single asset
        best_single = buy_hold

        if multi_asset_candles and len(multi_asset_candles) > 1:
            asset_returns = {}
            for symbol, sym_candles in multi_asset_candles.items():
                if len(sym_candles) > 0 and sym_candles[0].close > 0:
                    asset_returns[symbol] = sym_candles[-1].close / sym_candles[0].close - 1
                else:
                    asset_returns[symbol] = 0.0

            if asset_returns:
                equal_weight = np.mean(list(asset_returns.values()))
                best_single = max(asset_returns.values())

        return {
            "buy_and_hold_return": float(buy_hold),
            "equal_weight_return": float(equal_weight),
            "best_single_asset_return": float(best_single),
        }



class RegimeDetector:
    """Regime detection for regime-aware backtesting.

    Uses simple volatility and trend-based regime classification
    to label market regimes during backtesting.
    """

    def __init__(self, lookback: int = 100, n_regimes: int = 3):
        self.lookback = lookback
        self.n_regimes = n_regimes

    def detect_regimes(self, closes: np.ndarray) -> np.ndarray:
        """Detect market regimes from price series.

        Classifies each bar into a regime based on rolling
        volatility and trend.

        Args:
            closes: Array of close prices.

        Returns:
            Array of regime labels (0 = low vol, 1 = medium, 2 = high vol/crisis).
        """
        n = len(closes)
        if n < self.lookback:
            return np.zeros(n, dtype=int)

        returns = np.diff(closes) / closes[:-1]
        regimes = np.zeros(n, dtype=int)

        for i in range(self.lookback, n):
            window_returns = returns[i - self.lookback:i]
            vol = np.std(window_returns)
            mean_ret = np.mean(window_returns)

            # Simple 3-regime classification
            vol_threshold_low = np.percentile(np.std(returns[max(0, i-500):i]) if i > 500 else
                                              np.std(returns[:i]), 33) if i > 10 else 0.01
            vol_threshold_high = np.percentile(np.std(returns[max(0, i-500):i]) if i > 500 else
                                               np.std(returns[:i]), 67) if i > 10 else 0.03

            if vol > vol_threshold_high or mean_ret < -3 * vol:
                regimes[i] = 2  # Crisis / high vol
            elif vol > vol_threshold_low:
                regimes[i] = 1  # Normal
            else:
                regimes[i] = 0  # Low vol

        return regimes



class SensitivityAnalysis:
    """Parametric sensitivity analysis for backtests.

    Varies key parameters and shows how backtest results change,
    identifying which parameters have the most impact.
    """

    @staticmethod
    def run(engine: 'BacktestEngine', candles: List[Candle],
            strategy: Strategy, params: Dict[str, List[float]]) -> Dict:
        """Run parametric sensitivity analysis.

        Args:
            engine: BacktestEngine instance.
            candles: Price candles.
            strategy: Strategy to test.
            params: Dict mapping parameter name to list of values to test.

        Returns:
            Dict with sensitivity results per parameter.
        """
        results = {}

        for param_name, values in params.items():
            param_results = []
            original_value = getattr(engine.config, param_name, None)

            for value in values:
                if original_value is not None:
                    setattr(engine.config, param_name, value)

                try:
                    result = engine._run_single(candles, strategy)
                    param_results.append({
                        "value": value,
                        "total_return": result.total_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "total_trades": result.total_trades,
                    })
                except Exception:
                    param_results.append({
                        "value": value,
                        "total_return": float('nan'),
                        "sharpe_ratio": float('nan'),
                        "max_drawdown": float('nan'),
                        "total_trades": 0,
                    })

            if original_value is not None:
                setattr(engine.config, param_name, original_value)

            # Compute sensitivity metrics
            valid_results = [r for r in param_results if not np.isnan(r["total_return"])]
            if len(valid_results) >= 2:
                returns_range = [r["total_return"] for r in valid_results]
                sensitivity = max(returns_range) - min(returns_range)
            else:
                sensitivity = 0.0

            results[param_name] = {
                "results": param_results,
                "sensitivity": float(sensitivity),
            }

        return results


__all__ = ["BenchmarkComparison", "RegimeDetector", "SensitivityAnalysis"]
