"""LSTM-based price direction prediction model."""

import numpy as np
from typing import Optional, Dict
from pathlib import Path

from acms.ml.config import MLConfig


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



__all__ = ["PricePredictionModel"]
