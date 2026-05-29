"""Feature engineering for ML models."""

import numpy as np
from typing import Optional


class FeatureEngineer:
    """Generate ML features from market data with feature selection.

    Computes a comprehensive set of technical features including:
    - Multi-period returns
    - Rolling volatility
    - Volume features (OBV-like, volume ratio)
    - Price position within range
    - RSI-like momentum
    - Price acceleration
    """

    def __init__(self, window: int = 60):
        self.window = window
        self._selected_features: Optional[np.ndarray] = None

    def compute_features(self, candles_data: dict) -> np.ndarray:
        """Compute feature matrix from OHLCV data.

        Args:
            candles_data: Dict with keys 'close', 'volume', 'high', 'low', 'open'
                          each mapping to numpy arrays.

        Returns:
            Feature matrix of shape (window, n_features) or empty array if
            insufficient data.
        """
        closes = candles_data.get("close", np.array([]))
        volumes = candles_data.get("volume", np.array([]))
        highs = candles_data.get("high", np.array([]))
        lows = candles_data.get("low", np.array([]))
        opens = candles_data.get("open", np.array([]))

        if len(closes) < self.window + 20:
            return np.array([])

        features = []

        # Returns at multiple horizons
        for period in [1, 5, 10, 20]:
            ret = np.diff(closes, period) / closes[:-period]
            features.append(ret[-self.window:])

        # Rolling volatility
        returns = np.diff(closes) / closes[:-1]
        for period in [5, 10, 20]:
            vol = np.array([np.std(returns[max(0, i - period):i]) for i in range(period, len(returns))])
            if len(vol) >= self.window:
                features.append(vol[-self.window:])

        # Volume features
        if len(volumes) > self.window * 2:
            vol_ratio = volumes[-self.window:] / (np.mean(volumes[-self.window * 2:-self.window]) + 1e-10)
            features.append(vol_ratio[-self.window:])

        # Price position in range
        if len(highs) > self.window and len(lows) > self.window:
            range_ = highs[-self.window:] - lows[-self.window:]
            pos = (closes[-self.window:] - lows[-self.window:]) / (range_ + 1e-10)
            features.append(pos)

        # RSI-like momentum
        if len(closes) > self.window + 15:
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.convolve(gains, np.ones(14) / 14, mode='valid')
            avg_loss = np.convolve(losses, np.ones(14) / 14, mode='valid')
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - 100 / (1 + rs)
            if len(rsi) >= self.window:
                features.append(rsi[-self.window:])

        # Price acceleration (second derivative)
        if len(closes) > self.window + 5:
            smooth = np.convolve(closes, np.ones(5) / 5, mode='valid')
            velocity = np.diff(smooth)
            accel = np.diff(velocity)
            if len(accel) >= self.window:
                features.append(accel[-self.window:])

        feature_matrix = np.column_stack([f for f in features if len(f) == self.window])
        return feature_matrix

    def create_labels(self, closes: np.ndarray, horizon: int = 5,
                      method: str = "return_sign") -> np.ndarray:
        """Create prediction labels from price data.

        Args:
            closes: Array of closing prices.
            horizon: Forward-looking period for label computation.
            method: Label method - 'return_sign' (binary), 'return_magnitude' (regression),
                    'regime' (3-class: -1, 0, 1).

        Returns:
            Array of labels.
        """
        if len(closes) < horizon + 1:
            return np.array([])
        forward_returns = (closes[horizon:] - closes[:-horizon]) / closes[:-horizon]
        if method == "return_sign":
            return (forward_returns > 0).astype(int)
        elif method == "return_magnitude":
            return forward_returns
        elif method == "regime":
            labels = np.zeros(len(forward_returns))
            labels[forward_returns < -0.01] = -1
            labels[forward_returns > 0.01] = 1
            return labels
        return forward_returns

    def select_features_mutual_info(self, X: np.ndarray, y: np.ndarray,
                                     n_features: int = 20) -> np.ndarray:
        """Select top features using mutual information scoring.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
            n_features: Number of features to select.

        Returns:
            Boolean mask of selected features.
        """
        if X.shape[0] != len(y) or X.shape[1] == 0:
            return np.ones(X.shape[1], dtype=bool)

        try:
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X, y)
            top_indices = np.argsort(mi)[-min(n_features, len(mi)):]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_indices] = True
            self._selected_features = mask
            return mask
        except ImportError:
            # Fallback: variance-based selection
            variances = np.var(X, axis=0)
            top_indices = np.argsort(variances)[-min(n_features, len(variances)):]
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[top_indices] = True
            return mask

    def select_features_recursive(self, X: np.ndarray, y: np.ndarray,
                                   n_features: int = 20) -> np.ndarray:
        """Recursive feature elimination for feature selection.

        Uses a RandomForest estimator to rank features and recursively
        eliminate the least important ones.

        Args:
            X: Feature matrix.
            y: Target labels.
            n_features: Number of features to select.

        Returns:
            Boolean mask of selected features.
        """
        if X.shape[1] <= n_features:
            return np.ones(X.shape[1], dtype=bool)

        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector = selector.fit(X, y)
            return selector.support_
        except ImportError:
            return self.select_features_mutual_info(X, y, n_features)


__all__ = ["FeatureEngineer"]
