"""Autoencoder-based anomaly detection for market data."""

import numpy as np
from typing import Optional


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


__all__ = ["AnomalyDetector"]
