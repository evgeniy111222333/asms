"""Kelly Criterion allocation."""

import numpy as np


class KellyAllocator:
    """Kelly Criterion allocation across multiple assets."""

    def allocate(self, win_rates: np.ndarray, win_loss_ratios: np.ndarray,
                 capital: float, fraction: float = 0.5) -> dict:
        """Compute Kelly-optimal allocations.

        Args:
            win_rates: Win rate per asset.
            win_loss_ratios: Average win/loss ratio per asset.
            capital: Total capital.
            fraction: Fractional Kelly (0.5 = half-Kelly).

        Returns:
            Dict with weights and allocations.
        """
        n = len(win_rates)
        kelly_f = np.zeros(n)
        for i in range(n):
            if win_loss_ratios[i] > 0:
                kelly_f[i] = win_rates[i] - (1 - win_rates[i]) / win_loss_ratios[i]
                kelly_f[i] = max(kelly_f[i], 0.0)
        kelly_f *= fraction
        total = np.sum(kelly_f)
        if total > 1.0:
            kelly_f /= total
        allocations = kelly_f * capital
        return {"weights": kelly_f, "allocations": allocations}


__all__ = [
    "KellyAllocator",
]
