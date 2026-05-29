"""Signal persistence filter for whipsaw reduction."""

from collections import deque

from acms.core import SignalDirection


class SignalPersistenceFilter:
    """Filters whipsaw signals by requiring persistence.

    A signal must persist for N consecutive bars before
    being considered valid. This reduces false signals in
    choppy markets.
    """

    def __init__(self, persistence_bars: int = 2):
        self.persistence_bars = max(1, persistence_bars)
        self._signal_history: deque = deque(maxlen=persistence_bars + 1)
        self._last_direction = SignalDirection.NEUTRAL
        self._consecutive_count = 0

    def filter(self, direction: SignalDirection, strength: float) -> tuple:
        """Apply persistence filter to a signal.

        Args:
            direction: Current signal direction.
            strength: Current signal strength.

        Returns:
            Tuple of (filtered_direction, filtered_strength).
        """
        self._signal_history.append((direction, strength))
        if direction == self._last_direction:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_direction = direction
        if self._consecutive_count >= self.persistence_bars:
            return direction, strength
        elif self._consecutive_count == 1:
            return direction, strength * 0.3
        else:
            ratio = self._consecutive_count / self.persistence_bars
            return direction, strength * ratio

    def reset(self):
        """Reset filter state."""
        self._signal_history.clear()
        self._last_direction = SignalDirection.NEUTRAL
        self._consecutive_count = 0


__all__ = [
    "SignalPersistenceFilter",
]
