"""ML Module - Machine Learning pipeline for ACMS.

Implements:
- PyTorch neural network models (price prediction, regime classification)
- LightGBM gradient boosting (signal generation)
- RL environment (gymnasium.Env subclass for execution optimization)
- Optuna hyperparameter optimization
- Feature engineering pipeline with selection
- Ensemble methods (stacking, voting)
- Walk-forward validation
- Model monitoring (feature drift, prediction drift)
- Autoencoder for anomaly detection
- Transformer-based price prediction
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from scipy import stats as scipy_stats


@dataclass
class MLConfig:
    """Configuration for ML model training and evaluation."""
    model_dir: str = "/data/acms/models"
    feature_window: int = 60
    prediction_horizon: int = 5
    train_test_split: float = 0.8
    validation_split: float = 0.1
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    optuna_trials: int = 100
    optuna_timeout: int = 3600


# ============================================================================
# Feature Engineering
# ============================================================================

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


# ============================================================================
# Walk-Forward Validation
# ============================================================================

class WalkForwardValidation:
    """Walk-forward validation for time series ML models.

    Properly handles temporal ordering by training on past data
    and testing on future data, avoiding look-ahead bias.
    Supports configurable gap between train and test sets.
    """

    def __init__(self, n_splits: int = 5, train_pct: float = 0.7, gap: int = 0):
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.gap = gap

    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/test splits.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            List of (train_indices, test_indices) tuples preserving temporal order.
        """
        n = len(X)
        splits = []
        test_size = n // (self.n_splits + 1)
        if test_size < 10:
            return [(np.arange(n - 1), np.array([n - 1]))]

        for i in range(self.n_splits):
            test_end = n - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap
            train_start = 0

            if train_end <= 0:
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def validate(self, X: np.ndarray, y: np.ndarray, model_factory: callable,
                 metric: callable = None) -> Dict:
        """Run walk-forward validation.

        Args:
            X: Feature matrix.
            y: Target labels.
            model_factory: Callable that returns a new model instance with fit/predict.
            metric: Callable(y_true, y_pred) returning a score.

        Returns:
            Dict with fold scores and aggregate statistics.
        """
        if metric is None:
            metric = lambda yt, yp: np.mean(yt == yp)

        splits = self.split(X, y)
        scores = []

        for train_idx, test_idx in splits:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            score = metric(y_test, y_pred)
            scores.append(score)

        return {
            "fold_scores": scores,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "std_score": float(np.std(scores)) if scores else 0.0,
            "n_splits": len(scores),
        }


# ============================================================================
# Ensemble Methods
# ============================================================================

class EnsembleModel:
    """Ensemble model combining multiple ML models.

    Supports:
    - voting: Majority or weighted voting from base models
    - stacking: Base model predictions fed into a meta-learner
    """

    def __init__(self, models: List = None, method: str = "voting",
                 weights: Optional[List[float]] = None):
        self.models = models or []
        self.method = method
        self.weights = weights
        self.meta_model = None

    def add_model(self, model: Any) -> None:
        """Add a model to the ensemble.

        Args:
            model: Any object with fit/predict methods.
        """
        self.models.append(model)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Fit all models in the ensemble.

        For stacking, also trains a logistic regression meta-model on
        base model predictions.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Self for chaining.
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        if self.method == "voting":
            for model in self.models:
                model.fit(X, y)
        elif self.method == "stacking":
            for model in self.models:
                model.fit(X, y)
            meta_features = self._generate_meta_features(X)
            try:
                from sklearn.linear_model import LogisticRegression
                self.meta_model = LogisticRegression(max_iter=1000)
                self.meta_model.fit(meta_features, y)
            except ImportError:
                self.meta_model = None
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the ensemble.

        Args:
            X: Feature matrix.

        Returns:
            Array of predictions.
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        if self.method == "voting":
            return self._voting_predict(X)
        elif self.method == "stacking":
            return self._stacking_predict(X)
        return np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using the ensemble (voting only).

        Args:
            X: Feature matrix.

        Returns:
            Array of average probabilities across models.
        """
        probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                probas.append(model.predict_proba(X))
        if not probas:
            raise ValueError("No models support predict_proba")
        return np.mean(probas, axis=0)

    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Majority/weighted voting prediction."""
        predictions = np.array([model.predict(X) for model in self.models])
        if self.weights is not None and len(self.weights) == len(self.models):
            weighted = np.zeros(len(X))
            for i, w in enumerate(self.weights):
                weighted += w * (predictions[i] == 1).astype(float)
            return (weighted > sum(self.weights) / 2).astype(int)
        else:
            return (np.mean(predictions, axis=0) > 0.5).astype(int)

    def _stacking_predict(self, X: np.ndarray) -> np.ndarray:
        """Stacking prediction using meta model."""
        if self.meta_model is None:
            return self._voting_predict(X)
        meta_features = self._generate_meta_features(X)
        return self.meta_model.predict(meta_features)

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta features from base model predictions."""
        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.T


# ============================================================================
# Model Monitoring
# ============================================================================

class ModelMonitor:
    """Model monitoring for drift detection.

    Monitors:
    - Feature drift using Population Stability Index (PSI) and KS test
    - Prediction drift using KS test
    """

    def __init__(self, reference_features: Optional[np.ndarray] = None,
                 reference_predictions: Optional[np.ndarray] = None,
                 drift_threshold: float = 0.05,
                 psi_threshold: float = 0.2):
        self.reference_features = reference_features
        self.reference_predictions = reference_predictions
        self.drift_threshold = drift_threshold
        self.psi_threshold = psi_threshold

    def detect_feature_drift(self, current_features: np.ndarray) -> Dict:
        """Detect feature drift using Kolmogorov-Smirnov test and PSI.

        Args:
            current_features: Current feature matrix.

        Returns:
            Dict with drift detection results per feature including KS test
            and Population Stability Index.
        """
        if self.reference_features is None:
            return {"drift_detected": False, "details": "No reference features"}

        n_features = min(self.reference_features.shape[1], current_features.shape[1])
        drift_results = {}
        any_drift = False

        for j in range(n_features):
            ref = self.reference_features[:, j]
            cur = current_features[:, j]

            # KS test
            ks_stat, p_value = scipy_stats.ks_2samp(ref, cur)
            ks_drifted = p_value < self.drift_threshold

            # PSI
            psi_value = self._compute_psi(ref, cur)
            psi_drifted = psi_value > self.psi_threshold

            drifted = ks_drifted or psi_drifted
            drift_results[f"feature_{j}"] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "ks_drifted": ks_drifted,
                "psi_value": float(psi_value),
                "psi_drifted": psi_drifted,
                "drifted": drifted,
            }
            if drifted:
                any_drift = True

        return {"drift_detected": any_drift, "details": drift_results}

    def detect_prediction_drift(self, current_predictions: np.ndarray) -> Dict:
        """Detect prediction drift using KS test.

        Args:
            current_predictions: Current model predictions.

        Returns:
            Dict with drift detection results.
        """
        if self.reference_predictions is None:
            return {"drift_detected": False, "details": "No reference predictions"}

        ks_stat, p_value = scipy_stats.ks_2samp(self.reference_predictions, current_predictions)
        return {
            "drift_detected": p_value < self.drift_threshold,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
        }

    @staticmethod
    def _compute_psi(expected: np.ndarray, actual: np.ndarray,
                     n_bins: int = 10) -> float:
        """Compute Population Stability Index (PSI).

        Args:
            expected: Reference distribution values.
            actual: Current distribution values.
            n_bins: Number of bins for distribution comparison.

        Returns:
            PSI value. Values > 0.2 indicate significant drift.
        """
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
        actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Avoid division by zero
        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)


# ============================================================================
# Autoencoder for Anomaly Detection
# ============================================================================

class AnomalyDetector:
    """Autoencoder-based anomaly detection for market data.

    Detects anomalous market conditions by training an autoencoder
    on normal market data and flagging high reconstruction error
    as anomalies. Falls back to statistical distance if PyTorch unavailable.
    """

    def __init__(self, encoding_dim: int = 10, threshold_percentile: float = 95.0):
        self.encoding_dim = encoding_dim
        self.threshold_percentile = threshold_percentile
        self.model = None
        self.threshold = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, epochs: int = 50, batch_size: int = 32) -> None:
        """Fit autoencoder on normal data.

        Args:
            X: Feature matrix of normal market data.
            epochs: Training epochs.
            batch_size: Batch size.
        """
        if len(X) < batch_size:
            batch_size = max(1, len(X))

        try:
            import torch
            import torch.nn as nn

            input_dim = X.shape[1]

            class Autoencoder(nn.Module):
                def __init__(self, input_dim: int, encoding_dim: int):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 64), nn.ReLU(),
                        nn.Linear(64, 32), nn.ReLU(),
                        nn.Linear(32, encoding_dim),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(encoding_dim, 32), nn.ReLU(),
                        nn.Linear(32, 64), nn.ReLU(),
                        nn.Linear(64, input_dim),
                    )

                def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                    return self.decoder(self.encoder(x))

            self.model = Autoencoder(input_dim, self.encoding_dim)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            X_tensor = torch.FloatTensor(X)
            dataset = torch.utils.data.TensorDataset(X_tensor)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.model.train()
            for epoch in range(epochs):
                for batch in loader:
                    x = batch[0]
                    output = self.model(x)
                    loss = criterion(output, x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.model.eval()
            with torch.no_grad():
                recon = self.model(X_tensor)
                errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
            self.threshold = float(np.percentile(errors, self.threshold_percentile))

        except ImportError:
            self.model = None
            self._mean = np.mean(X, axis=0)
            self._std = np.std(X, axis=0) + 1e-10
            distances = np.sqrt(np.sum(((X - self._mean) / self._std) ** 2, axis=1))
            self.threshold = float(np.percentile(distances, self.threshold_percentile))

    def detect(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies in new data.

        Args:
            X: Feature matrix to check for anomalies.

        Returns:
            Boolean array where True indicates anomaly.
        """
        if self.threshold is None:
            raise RuntimeError("Detector not fitted yet")

        if self.model is not None:
            import torch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                recon = self.model(X_tensor)
                errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
            return errors > self.threshold
        else:
            if self._mean is None or self._std is None:
                raise RuntimeError("Detector not fitted yet")
            distances = np.sqrt(np.sum(((X - self._mean) / self._std) ** 2, axis=1))
            return distances > self.threshold

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (reconstruction errors or distances).

        Args:
            X: Feature matrix.

        Returns:
            Array of anomaly scores. Higher = more anomalous.
        """
        if self.model is not None:
            import torch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                recon = self.model(X_tensor)
                errors = torch.mean((recon - X_tensor) ** 2, dim=1).numpy()
            return errors
        else:
            if self._mean is None or self._std is None:
                raise RuntimeError("Detector not fitted yet")
            return np.sqrt(np.sum(((X - self._mean) / self._std) ** 2, axis=1))


# ============================================================================
# Transformer-based Price Prediction
# ============================================================================

class TransformerPredictor:
    """Transformer-based price direction prediction.

    Uses multi-head self-attention to capture long-range
    dependencies in price sequences. Outputs 3-class predictions
    (down, neutral, up).
    """

    def __init__(self, input_size: int = 20, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2,
                 output_size: int = 3, dropout: float = 0.1):
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.model = None
        self.is_trained = False

    def build_model(self) -> None:
        """Build the Transformer model architecture."""
        try:
            import torch
            import torch.nn as nn

            class TransformerModel(nn.Module):
                def __init__(self, input_size: int, d_model: int, nhead: int,
                             num_layers: int, output_size: int, dropout: float):
                    super().__init__()
                    self.input_proj = nn.Linear(input_size, d_model)
                    self.pos_encoder = nn.Parameter(torch.randn(1, 500, d_model) * 0.1)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                        dropout=dropout, batch_first=True,
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.fc = nn.Sequential(
                        nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(32, output_size),
                    )

                def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                    x = self.input_proj(x)
                    x = x + self.pos_encoder[:, :x.size(1), :]
                    x = self.transformer(x)
                    x = x.mean(dim=1)
                    return self.fc(x)

            self.model = TransformerModel(
                self.input_size, self.d_model, self.nhead,
                self.num_layers, self.output_size, self.dropout,
            )
        except ImportError:
            raise ImportError("PyTorch is required for TransformerPredictor")

    def train(self, X: np.ndarray, y: np.ndarray, config: Optional[MLConfig] = None) -> Dict:
        """Train the Transformer model.

        Args:
            X: Input sequences of shape (n_samples, seq_len, input_size).
            y: Target labels.
            config: Training configuration.

        Returns:
            Dict with training metrics.
        """
        if self.model is None:
            self.build_model()

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        config = config or MLConfig()
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        epoch_losses = []
        for epoch in range(config.epochs):
            total_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            epoch_losses.append(total_loss / max(n_batches, 1))
        self.is_trained = True
        return {"final_loss": epoch_losses[-1] if epoch_losses else 0.0, "epochs": config.epochs}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions.

        Args:
            X: Input sequences.

        Returns:
            Array of predicted class indices.
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained yet")
        import torch
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.FloatTensor(X))
            return torch.argmax(output, dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input sequences.

        Returns:
            Array of class probabilities.
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained yet")
        import torch
        self.model.eval()
        with torch.no_grad():
            output = self.model(torch.FloatTensor(X))
            return torch.softmax(output, dim=1).numpy()


# ============================================================================
# RL Environment (gymnasium.Env subclass)
# ============================================================================

class TradingEnvironment:
    """RL Trading Environment for execution optimization.

    Implements a gymnasium-compatible environment where:
    - Observation: market data features + inventory + PnL + indicators
    - Action: buy/sell/hold with continuous size parameter
    - Reward: risk-adjusted return (Sharpe-like)
    """

    def __init__(self, candles_data: dict, initial_inventory: float = 100.0,
                 max_steps: int = 100, transaction_cost_bps: float = 10.0,
                 risk_free_rate: float = 0.0):
        self.candles_data = candles_data
        self.initial_inventory = initial_inventory
        self.max_steps = max_steps
        self.transaction_cost_bps = transaction_cost_bps
        self.risk_free_rate = risk_free_rate

        self.current_step = 0
        self.inventory = initial_inventory
        self.avg_price = 0.0
        self.total_cost = 0.0
        self.done = False

        self.closes = candles_data.get("close", np.array([]))
        self.volumes = candles_data.get("volume", np.array([]))
        self.highs = candles_data.get("high", np.array([]))
        self.lows = candles_data.get("low", np.array([]))

    def _create_gym_env(self):
        """Create the gymnasium environment with full observation/action spaces."""
        try:
            import gymnasium as gym
            from gymnasium import spaces

            closes = self.closes
            volumes = self.volumes
            highs = self.highs
            lows = self.lows
            initial_inventory = self.initial_inventory
            max_steps = self.max_steps
            transaction_cost_bps = self.transaction_cost_bps
            risk_free_rate = self.risk_free_rate

            class ACMSTradingEnv(gym.Env):
                """ACMS Trading Environment following gymnasium interface.

                Observation space (7 dimensions):
                - Normalized price (relative to first close)
                - Normalized volume
                - Inventory ratio (remaining/initial)
                - Step progress ratio
                - Running PnL (normalized)
                - ATR-like volatility proxy
                - Price position in high-low range

                Action space (3 dimensions, continuous):
                - action[0]: direction (-1=buy, 0=hold, +1=sell)
                - action[1]: order size fraction [0,1]
                - action[2]: limit price offset [-0.5%, +0.5%]
                """

                metadata = {"render_modes": ["human"]}

                def __init__(self):
                    super().__init__()
                    self.closes = closes
                    self.volumes = volumes
                    self.highs = highs
                    self.lows = lows
                    self.initial_inventory = initial_inventory
                    self.max_steps = max_steps
                    self.transaction_cost_bps = transaction_cost_bps
                    self.risk_free_rate = risk_free_rate
                    self.avg_volume = float(np.mean(volumes)) if len(volumes) > 0 else 1.0

                    # Action: [direction, size_fraction, price_offset]
                    self.action_space = spaces.Box(
                        low=np.array([-1.0, 0.0, -1.0]),
                        high=np.array([1.0, 1.0, 1.0]),
                        dtype=np.float32,
                    )

                    # Observation: 7 features
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32,
                    )

                    self.reset()

                def reset(self, seed=None, options=None):
                    super().reset(seed=seed)
                    self.current_step = 0
                    self.inventory = self.initial_inventory
                    self.total_cost = 0.0
                    self.avg_price = 0.0
                    self.realized_pnl = 0.0
                    self.pnl_history: List[float] = []
                    self.position = 0.0
                    self.position_avg_price = 0.0
                    return self._get_obs(), {}

                def step(self, action):
                    direction = action[0]
                    size_fraction = action[1]
                    price_offset = action[2] * 0.005

                    idx = min(self.current_step, len(self.closes) - 1)
                    price = self.closes[idx]
                    fill_price = price * (1 + price_offset)

                    # Determine order quantity
                    if direction > 0.3:  # Buy
                        quantity = size_fraction * self.inventory * 0.1
                        cost = fill_price * quantity * (1 + self.transaction_cost_bps / 10000)
                        self.total_cost += cost
                        self.inventory -= cost / fill_price
                        # Update position tracking
                        old_pos_value = self.position * self.position_avg_price
                        self.position += quantity
                        if self.position > 0:
                            self.position_avg_price = (old_pos_value + fill_price * quantity) / self.position
                    elif direction < -0.3:  # Sell
                        quantity = min(size_fraction * abs(self.position) * 0.1, abs(self.position))
                        if quantity > 0 and self.position > 0:
                            proceeds = fill_price * quantity * (1 - self.transaction_cost_bps / 10000)
                            self.realized_pnl += (fill_price - self.position_avg_price) * quantity
                            self.position -= quantity
                            self.inventory += proceeds / fill_price

                    self.current_step += 1

                    # Compute unrealized PnL
                    current_price = self.closes[min(self.current_step, len(self.closes) - 1)]
                    unrealized_pnl = self.position * (current_price - self.position_avg_price)
                    total_pnl = self.realized_pnl + unrealized_pnl
                    self.pnl_history.append(total_pnl)

                    # Reward: risk-adjusted return
                    if len(self.pnl_history) > 1:
                        returns = np.diff(self.pnl_history[-min(20, len(self.pnl_history)):])
                        reward = np.mean(returns) - 0.5 * np.var(returns)
                    else:
                        reward = 0.0

                    # Penalty for excessive inventory
                    if self.inventory < self.initial_inventory * 0.1:
                        reward -= 1.0

                    terminated = self.current_step >= self.max_steps or self.inventory <= 0.01
                    truncated = False

                    return self._get_obs(), float(reward), terminated, truncated, {}

                def _get_obs(self):
                    idx = min(self.current_step, len(self.closes) - 1)
                    price_norm = self.closes[idx] / self.closes[0] - 1
                    volume_norm = self.volumes[idx] / (self.avg_volume + 1e-10)
                    inventory_ratio = self.inventory / self.initial_inventory
                    step_ratio = self.current_step / self.max_steps

                    # Running PnL normalized
                    pnl_norm = 0.0
                    if self.initial_inventory > 0:
                        current_price = self.closes[idx]
                        unrealized = self.position * (current_price - self.position_avg_price)
                        pnl_norm = (self.realized_pnl + unrealized) / self.initial_inventory

                    # ATR-like volatility proxy
                    lookback = min(14, idx)
                    if lookback > 1:
                        atr = float(np.mean(self.highs[idx-lookback:idx] - self.lows[idx-lookback:idx]))
                        atr_norm = atr / (self.closes[0] + 1e-10)
                    else:
                        atr_norm = 0.0

                    # Price position in range
                    range_val = self.highs[idx] - self.lows[idx]
                    if range_val > 0:
                        range_pos = (self.closes[idx] - self.lows[idx]) / range_val
                    else:
                        range_pos = 0.5

                    return np.array([
                        price_norm, volume_norm, inventory_ratio, step_ratio,
                        pnl_norm, atr_norm, range_pos,
                    ], dtype=np.float32)

            return ACMSTradingEnv()
        except ImportError:
            raise ImportError("gymnasium is required for RL environment")

    def make_env(self):
        """Create and return the gymnasium environment."""
        return self._create_gym_env()


# ============================================================================
# PyTorch LSTM Price Prediction Model
# ============================================================================

class PricePredictionModel:
    """LSTM-based price direction prediction model.

    Uses a multi-layer LSTM with attention mechanism for
    predicting price direction (down/neutral/up).
    """

    def __init__(self, input_size: int = 20, hidden_size: int = 128,
                 num_layers: int = 2, output_size: int = 3, dropout: float = 0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.model = None
        self.is_trained = False

    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        try:
            import torch
            import torch.nn as nn

            class LSTMModel(nn.Module):
                def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                             output_size: int, dropout: float):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                        batch_first=True, dropout=dropout)
                    self.attention = nn.Linear(hidden_size, 1)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 64), nn.ReLU(),
                        nn.Dropout(dropout), nn.Linear(64, output_size),
                    )

                def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                    lstm_out, _ = self.lstm(x)
                    attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                    context = torch.sum(attn_weights * lstm_out, dim=1)
                    return self.fc(context)

            self.model = LSTMModel(self.input_size, self.hidden_size, self.num_layers,
                                   self.output_size, self.dropout)
        except ImportError:
            raise ImportError("PyTorch is required")

    def train(self, X: np.ndarray, y: np.ndarray, config: Optional[MLConfig] = None) -> Dict:
        """Train the LSTM model.

        Args:
            X: Input sequences of shape (n_samples, seq_len, input_size).
            y: Target labels.
            config: Training configuration.

        Returns:
            Dict with training metrics.
        """
        if self.model is None:
            self.build_model()
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        config = config or MLConfig()
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0.0
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            total_loss = epoch_loss / max(n_batches, 1)
        self.is_trained = True
        return {"final_loss": total_loss, "epochs": config.epochs}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions.

        Args:
            X: Input sequences.

        Returns:
            Array of predicted class indices.
        """
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained yet")
        import torch
        self.model.eval()
        with torch.no_grad():
            return torch.argmax(self.model(torch.FloatTensor(X)), dim=1).numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None or not self.is_trained:
            raise RuntimeError("Model not trained yet")
        import torch
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(torch.FloatTensor(X)), dim=1).numpy()

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: File path to save model state dict.
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: File path to load model state dict from.
        """
        if self.model is None:
            self.build_model()
        import torch
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.is_trained = True


# ============================================================================
# LightGBM Signal Model
# ============================================================================

class LightGBMSignalModel:
    """LightGBM gradient boosting model for signal generation.

    Supports multi-class classification (down/neutral/up) with
    early stopping and feature importance tracking.
    """

    def __init__(self, objective: str = "multiclass", num_classes: int = 3):
        self.objective = objective
        self.num_classes = num_classes
        self.model = None
        self.feature_importance = None

    def train(self, X: np.ndarray, y: np.ndarray, config: Optional[MLConfig] = None) -> Dict:
        """Train the LightGBM model.

        Args:
            X: Feature matrix.
            y: Target labels.
            config: Training configuration.

        Returns:
            Dict with best iteration and feature importance.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required")

        if len(X) < 10:
            raise ValueError("Not enough data to train")

        config = config or MLConfig()
        split = int(len(X) * config.train_test_split)
        if split < 5 or (len(X) - split) < 5:
            split = max(5, len(X) - 5)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        params = {
            "objective": self.objective, "num_class": self.num_classes,
            "metric": "multi_logloss", "boosting_type": "gbdt",
            "num_leaves": 63, "learning_rate": config.learning_rate,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "verbose": -1,
        }
        self.model = lgb.train(
            params, train_data, num_boost_round=config.epochs,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(config.early_stopping_patience)],
        )
        self.feature_importance = self.model.feature_importance(importance_type="gain")
        return {
            "best_iteration": self.model.best_iteration,
            "feature_importance": self.feature_importance.tolist(),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make class predictions.

        Args:
            X: Feature matrix.

        Returns:
            Array of predicted class indices.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        return np.argmax(self.model.predict(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix.

        Returns:
            Array of class probabilities.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM is required")
        self.model = lgb.Booster(model_file=path)
        self.is_trained = True


# ============================================================================
# Optuna Hyperparameter Optimization
# ============================================================================

class HyperparameterOptimizer:
    """Optuna-based hyperparameter optimization.

    Supports LightGBM and simple neural network model types.
    Uses time-series-aware cross-validation.
    """

    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type
        self.best_params: Optional[Dict] = None
        self.study = None

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 n_trials: int = 100, timeout: int = 3600) -> Dict:
        """Run hyperparameter optimization.

        Args:
            X: Feature matrix.
            y: Target labels.
            n_trials: Maximum number of Optuna trials.
            timeout: Maximum optimization time in seconds.

        Returns:
            Dict with best_params, best_value, and n_trials.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna is required")

        if len(X) < 20:
            raise ValueError("Not enough data for optimization")

        def objective(trial):
            if self.model_type == "lightgbm":
                return self._lightgbm_objective(trial, X, y)
            return 0.0

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=n_trials, timeout=timeout)
        self.best_params = self.study.best_params
        return {
            "best_params": self.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
        }

    def _lightgbm_objective(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """LightGBM hyperparameter optimization objective."""
        try:
            import lightgbm as lgb
        except ImportError:
            return 0.0

        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        }
        split = int(len(X) * 0.8)
        if split < 5 or (len(X) - split) < 5:
            return 0.0
        train_data = lgb.Dataset(X[:split], label=y[:split])
        val_data = lgb.Dataset(X[split:], label=y[split:])
        base_params = {
            "objective": "multiclass", "num_class": 3,
            "metric": "multi_logloss", "verbose": -1,
        }
        base_params.update(params)
        try:
            model = lgb.train(base_params, train_data, num_boost_round=100,
                              valid_sets=[val_data], callbacks=[lgb.early_stopping(5)])
            preds = np.argmax(model.predict(X[split:]), axis=1)
            return float(np.mean(preds == y[split:]))
        except Exception:
            return 0.0


# ============================================================================
# RL Execution Optimizer
# ============================================================================

class RLExecutionOptimizer:
    """Reinforcement learning for optimal execution using Stable-Baselines3.

    Provides a complete pipeline for training RL agents on the
    trading environment, with support for PPO, A2C, and DQN algorithms.
    """

    def __init__(self, algorithm: str = "PPO"):
        self.algorithm = algorithm
        self.model = None
        self.env = None

    def create_environment(self, candles_data: dict, initial_inventory: float = 100.0,
                           max_steps: int = 100, transaction_cost_bps: float = 10.0):
        """Create a trading environment for RL.

        Args:
            candles_data: Dict with 'close', 'volume', 'high', 'low' arrays.
            initial_inventory: Starting inventory size.
            max_steps: Maximum number of steps per episode.
            transaction_cost_bps: Transaction cost in basis points.

        Returns:
            gymnasium.Env instance.
        """
        env_factory = TradingEnvironment(
            candles_data, initial_inventory, max_steps, transaction_cost_bps
        )
        self.env = env_factory.make_env()
        return self.env

    def train(self, total_timesteps: int = 100000, **kwargs) -> Dict:
        """Train RL agent.

        Args:
            total_timesteps: Total training timesteps.
            **kwargs: Additional algorithm-specific parameters.

        Returns:
            Dict with training info.
        """
        try:
            from stable_baselines3 import PPO, A2C, DQN
        except ImportError:
            raise ImportError("stable-baselines3 is required")

        if self.env is None:
            raise RuntimeError("Environment not created. Call create_environment() first.")

        algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        algo_class = algo_map.get(self.algorithm, PPO)

        learning_rate = kwargs.get("learning_rate", 3e-4)
        n_steps = kwargs.get("n_steps", 2048)
        batch_size = kwargs.get("batch_size", 64)
        n_epochs = kwargs.get("n_epochs", 10)

        self.model = algo_class(
            "MlpPolicy", self.env,
            learning_rate=learning_rate,
            n_steps=n_steps if self.algorithm == "PPO" else None,
            batch_size=batch_size,
            n_epochs=n_epochs if self.algorithm == "PPO" else None,
            verbose=0,
        )
        self.model.learn(total_timesteps=total_timesteps)
        return {"algorithm": self.algorithm, "total_timesteps": total_timesteps}

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict optimal action for given observation.

        Args:
            observation: Current environment observation.
            deterministic: Whether to use deterministic policy.

        Returns:
            Selected action array.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        """Save trained RL model to disk."""
        if self.model is None:
            raise RuntimeError("No model to save")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load trained RL model from disk."""
        try:
            from stable_baselines3 import PPO, A2C, DQN
        except ImportError:
            raise ImportError("stable-baselines3 is required")
        algo_map = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
        algo_class = algo_map.get(self.algorithm, PPO)
        self.model = algo_class.load(path)


# ============================================================================
# Model Persistence Utilities
# ============================================================================

class ModelRegistry:
    """Registry for managing trained ML models with metadata.

    Tracks model versions, training metrics, and provides
    a unified save/load interface for all model types.
    """

    def __init__(self, model_dir: str = "/data/acms/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._registry: Dict[str, Dict] = {}

    def register(self, name: str, model: Any, metrics: Optional[Dict] = None,
                 model_type: str = "unknown") -> None:
        """Register a trained model.

        Args:
            name: Unique model name.
            model: The trained model object.
            metrics: Training/validation metrics.
            model_type: Type identifier (lstm, lightgbm, transformer, etc.).
        """
        self._registry[name] = {
            "model": model,
            "metrics": metrics or {},
            "model_type": model_type,
            "registered_at": str(np.datetime64('now')),
        }

    def get(self, name: str) -> Optional[Any]:
        """Retrieve a registered model by name."""
        entry = self._registry.get(name)
        return entry["model"] if entry else None

    def list_models(self) -> List[Dict]:
        """List all registered models with metadata."""
        return [
            {"name": k, "model_type": v["model_type"], "metrics": v["metrics"],
             "registered_at": v["registered_at"]}
            for k, v in self._registry.items()
        ]

    def save_model(self, name: str, path: Optional[str] = None) -> str:
        """Save a registered model to disk.

        Args:
            name: Model name in registry.
            path: Optional custom path. Defaults to model_dir/name.

        Returns:
            Path where model was saved.
        """
        entry = self._registry.get(name)
        if not entry:
            raise KeyError(f"Model '{name}' not found in registry")

        save_path = path or str(self.model_dir / name)
        model = entry["model"]

        if hasattr(model, 'save'):
            model.save(save_path)
        else:
            try:
                import torch
                torch.save(model.state_dict() if hasattr(model, 'state_dict') else model, save_path)
            except ImportError:
                import pickle
                with open(save_path, 'wb') as f:
                    pickle.dump(model, f)

        return save_path
