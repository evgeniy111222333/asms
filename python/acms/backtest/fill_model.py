"""Execution fill models for realistic order simulation."""

from typing import Dict


class FillModel:
    """Execution fill models for realistic order simulation.

    Supports:
    - Immediate fill: full fill at current price
    - Partial fill: partial execution with configurable fill rate
    - Fill-or-kill (FOK): full fill or no fill
    """

    @staticmethod
    def immediate_fill(quantity: float, price: float) -> Dict:
        """Immediate full fill at specified price.

        Args:
            quantity: Order quantity.
            price: Fill price.

        Returns:
            Dict with fill details.
        """
        return {
            "filled_quantity": quantity,
            "fill_price": price,
            "fill_pct": 1.0,
            "partial": False,
        }

    @staticmethod
    def partial_fill(quantity: float, price: float, fill_pct: float = 0.7,
                     available_depth: float = float('inf')) -> Dict:
        """Partial fill model.

        Fills only a fraction of the order based on available depth
        and configured fill rate.

        Args:
            quantity: Order quantity.
            price: Fill price.
            fill_pct: Maximum fill percentage.
            available_depth: Available order book depth.

        Returns:
            Dict with fill details.
        """
        depth_fill = min(quantity, available_depth)
        effective_fill = min(depth_fill, quantity * fill_pct)
        return {
            "filled_quantity": effective_fill,
            "fill_price": price,
            "fill_pct": float(effective_fill / quantity) if quantity > 0 else 0.0,
            "partial": effective_fill < quantity,
            "unfilled_quantity": quantity - effective_fill,
        }

    @staticmethod
    def fill_or_kill(quantity: float, price: float,
                     available_depth: float = float('inf'),
                     min_fill_pct: float = 0.95) -> Dict:
        """Fill-or-kill model.

        Order is fully filled or not at all, depending on
        available depth.

        Args:
            quantity: Order quantity.
            price: Fill price.
            available_depth: Available order book depth.
            min_fill_pct: Minimum fill percentage to accept.

        Returns:
            Dict with fill details (filled_quantity is 0 or full).
        """
        if available_depth >= quantity * min_fill_pct:
            return {
                "filled_quantity": quantity,
                "fill_price": price,
                "fill_pct": 1.0,
                "partial": False,
                "unfilled_quantity": 0.0,
            }
        return {
            "filled_quantity": 0.0,
            "fill_price": 0.0,
            "fill_pct": 0.0,
            "partial": False,
            "unfilled_quantity": quantity,
        }


__all__ = ["FillModel"]
