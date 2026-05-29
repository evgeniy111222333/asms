"""Transformer-based price direction prediction."""

import numpy as np
from typing import Optional, Dict

from acms.ml.config import MLConfig


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


__all__ = ["TransformerPredictor"]
