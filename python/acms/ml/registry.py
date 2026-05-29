"""Model persistence registry."""

from typing import Optional, List, Dict, Any
from pathlib import Path


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


__all__ = ["ModelRegistry"]
