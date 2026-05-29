"""Circuit Breaker for ACMS."""

from typing import Optional
from datetime import datetime


class CircuitBreaker:
    """Circuit breaker for automatic trading pause.

    Triggers on rapid loss threshold, order rate spike, or volatility spike.
    Enforces a cooldown period before trading can resume.
    """

    def __init__(self, loss_threshold_pct: float = 0.03,
                 cooldown_minutes: int = 30,
                 vol_spike_mult: float = 5.0):
        """Initialize circuit breaker.

        Args:
            loss_threshold_pct: Loss percentage that triggers the breaker.
            cooldown_minutes: Minutes before the breaker auto-resets.
            vol_spike_mult: Volatility spike multiplier to trigger.
        """
        self.loss_threshold_pct = loss_threshold_pct
        self.cooldown_minutes = cooldown_minutes
        self.vol_spike_mult = vol_spike_mult
        self.triggered = False
        self.trigger_reason = ""
        self.triggered_at: Optional[datetime] = None

    def check(self, current_pnl_pct: float, current_vol: float,
              normal_vol: float) -> bool:
        """Check if circuit breaker should trigger.

        Args:
            current_pnl_pct: Current P&L as percentage of capital.
            current_vol: Current realized volatility.
            normal_vol: Normal (average) volatility.

        Returns:
            True if circuit breaker is triggered.
        """
        if self.triggered:
            if self.triggered_at:
                elapsed = (datetime.utcnow() - self.triggered_at).total_seconds() / 60
                if elapsed >= self.cooldown_minutes:
                    self.reset()
                    return False
            return True

        if current_pnl_pct < -self.loss_threshold_pct:
            self._trigger(f"Loss exceeds {self.loss_threshold_pct:.1%}: {current_pnl_pct:.2%}")
            return True

        if normal_vol > 0 and current_vol > normal_vol * self.vol_spike_mult:
            self._trigger(f"Volatility spike: {current_vol:.4f} vs normal {normal_vol:.4f}")
            return True

        return False

    def _trigger(self, reason: str):
        """Activate the circuit breaker."""
        self.triggered = True
        self.trigger_reason = reason
        self.triggered_at = datetime.utcnow()

    def reset(self):
        """Reset the circuit breaker."""
        self.triggered = False
        self.trigger_reason = ""
        self.triggered_at = None

__all__ = ['CircuitBreaker']
