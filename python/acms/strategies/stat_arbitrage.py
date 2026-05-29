"""Statistical arbitrage strategy implementation."""

import numpy as np
from typing import Optional, List, Dict
from datetime import datetime

from acms.core import Signal, SignalDirection, Candle, Position
from acms.strategies.base import Strategy


class StatisticalArbitrageStrategy(Strategy):
    """Statistical Arbitrage (pair trading with cointegration + Kalman filter).

    Identifies cointegrated pairs and trades the spread
    when it deviates from equilibrium. Uses a Kalman filter
    to dynamically update the hedge ratio instead of static OLS.
    """

    def __init__(self, symbol: str, symbol2: str, lookback: int = 100,
                 entry_zscore: float = 2.0, exit_zscore: float = 0.5,
                 use_kalman: bool = True):
        super().__init__("stat_arb", symbol)
        self.symbol2 = symbol2
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.use_kalman = use_kalman
        self._hedge_ratio = None
        self._spread_mean = None
        self._spread_std = None
        # Kalman filter state
        self._kf_state = np.array([1.0])  # hedge ratio estimate
        self._kf_cov = np.array([[1.0]])  # state covariance
        self._kf_Q = np.array([[0.001]])  # process noise
        self._kf_R = np.array([[0.01]])   # measurement noise
        self._spread_history: List[float] = []
        self._pair_candles: Dict[str, List[Candle]] = {}

    def update_pair_data(self, symbol: str, candles: List[Candle]) -> None:
        """Update candle data for the pair symbol."""
        self._pair_candles[symbol] = candles

    def _kalman_update(self, y: float, x: float) -> float:
        """Kalman filter update for dynamic hedge ratio estimation."""
        state_pred = self._kf_state.copy()
        cov_pred = self._kf_cov + self._kf_Q
        H = np.array([[x]])
        S = H @ cov_pred @ H.T + self._kf_R
        K = cov_pred @ H.T @ np.linalg.inv(S)
        innovation = np.array([[y]]) - H @ state_pred.reshape(-1, 1)
        self._kf_state = state_pred + (K @ innovation).flatten()
        self._kf_cov = (np.eye(1) - K @ H) @ cov_pred
        return float(self._kf_state[0])

    def compute_spread(self, prices1: np.ndarray, prices2: np.ndarray) -> np.ndarray:
        """Compute the cointegration spread with optional Kalman filter."""
        if len(prices1) != len(prices2) or len(prices1) < 30:
            return np.array([])
        if self.use_kalman:
            spread = np.zeros(len(prices1))
            for i in range(len(prices1)):
                hr = self._kalman_update(prices1[i], prices2[i])
                spread[i] = prices1[i] - hr * prices2[i]
            self._hedge_ratio = self._kf_state[0]
            return spread
        else:
            X = np.column_stack([np.ones(len(prices2)), prices2])
            beta = np.linalg.lstsq(X, prices1, rcond=None)[0]
            self._hedge_ratio = beta[1]
            return prices1 - self._hedge_ratio * prices2

    def cointegration_test(self, prices1: np.ndarray, prices2: np.ndarray,
                           significance: float = 0.05) -> Dict:
        """Run augmented Engle-Granger cointegration test."""
        spread = self.compute_spread(prices1[-self.lookback:], prices2[-self.lookback:])
        if len(spread) < 20:
            return {"is_cointegrated": False, "hedge_ratio": None, "half_life": float('inf')}
        spread_diff = np.diff(spread)
        spread_lag = spread[:-1]
        valid = np.isfinite(spread_diff) & np.isfinite(spread_lag)
        if valid.sum() < 10:
            return {"is_cointegrated": False, "hedge_ratio": None, "half_life": float('inf')}
        X = spread_lag[valid].reshape(-1, 1)
        y = spread_diff[valid]
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        beta_val = beta[0]
        is_cointegrated = beta_val < -0.05
        half_life = -np.log(2) / beta_val if beta_val < 0 else float('inf')
        return {
            "is_cointegrated": is_cointegrated,
            "hedge_ratio": self._hedge_ratio,
            "half_life": half_life,
        }

    def evaluate_pair(self, closes1: np.ndarray, closes2: np.ndarray) -> Optional[Signal]:
        """Evaluate the pair for trading signals."""
        spread = self.compute_spread(closes1[-self.lookback:], closes2[-self.lookback:])
        if len(spread) < 20:
            return None
        self._spread_mean = np.mean(spread)
        self._spread_std = np.std(spread)
        if self._spread_std == 0:
            return None
        z_score = (spread[-1] - self._spread_mean) / self._spread_std
        self._spread_history.append(z_score)
        regime_mult = 1.0
        if len(self._spread_history) > 20:
            recent_std = np.std(self._spread_history[-20:])
            if recent_std > 2 * self._spread_std:
                regime_mult = 0.5
        entry = self.entry_zscore * regime_mult
        if z_score > entry:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.SHORT,
                strength=min(abs(z_score) / 4.0, 1.0), strategy_id=self.strategy_id,
                indicators={"spread_zscore": z_score, "hedge_ratio": self._hedge_ratio,
                            "spread_mean": self._spread_mean, "kalman_enabled": self.use_kalman},
            )
        elif z_score < -entry:
            return Signal(
                id=f"sig_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
                symbol=self.symbol, direction=SignalDirection.LONG,
                strength=min(abs(z_score) / 4.0, 1.0), strategy_id=self.strategy_id,
                indicators={"spread_zscore": z_score, "hedge_ratio": self._hedge_ratio,
                            "spread_mean": self._spread_mean, "kalman_enabled": self.use_kalman},
            )
        return None

    def evaluate(self, candles: List[Candle]) -> Optional[Signal]:
        """Evaluate statistical arbitrage using the primary pair from config."""
        if not candles or len(candles) < 2:
            return None
        pair_candles = self._pair_candles.get(self.symbol2, [])
        if not pair_candles:
            return None
        prices_a = np.array([c.close for c in candles])
        prices_b = np.array([c.close for c in pair_candles])
        min_len = min(len(prices_a), len(prices_b))
        if min_len < self.lookback:
            return None
        prices_a = prices_a[-min_len:]
        prices_b = prices_b[-min_len:]
        return self.evaluate_pair(prices_a, prices_b)

    def should_exit_pair(self, closes1: np.ndarray, closes2: np.ndarray) -> bool:
        """Check if spread has reverted to exit threshold."""
        spread = self.compute_spread(closes1[-self.lookback:], closes2[-self.lookback:])
        if len(spread) < 2 or self._spread_std is None or self._spread_std == 0:
            return False
        z_score = (spread[-1] - self._spread_mean) / self._spread_std
        return abs(z_score) < self.exit_zscore

    def should_exit(self, candles: List[Candle], position: Position) -> bool:
        """Check if statistical arbitrage position should be exited."""
        pair_candles = self._pair_candles.get(self.symbol2, [])
        if not pair_candles or not candles:
            return False
        prices_a = np.array([c.close for c in candles[-self.lookback:]])
        prices_b = np.array([c.close for c in pair_candles[-self.lookback:]])
        min_len = min(len(prices_a), len(prices_b))
        if min_len < self.lookback:
            return False
        return self.should_exit_pair(prices_a[-min_len:], prices_b[-min_len:])


__all__ = [
    "StatisticalArbitrageStrategy",
]
