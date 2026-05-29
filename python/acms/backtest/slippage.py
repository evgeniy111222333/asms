"""Slippage models for realistic execution simulation."""

import numpy as np

from acms.core import Side


class SlippageModel:
    """Slippage models for realistic execution simulation.

    Implements three slippage models:
    - Percentage: fixed slippage as percentage of price
    - Square-root (Almgren-Chriss): slippage proportional to sqrt(participation rate)
    - Volume-dependent: slippage increases with lower volume
    """

    @staticmethod
    def percentage(price: float, quantity: float, slippage_bps: float, side: Side) -> float:
        """Percentage slippage model.

        fill_price = price * (1 +/- slippage_bps/10000)

        Args:
            price: Market price.
            quantity: Order quantity.
            slippage_bps: Slippage in basis points.
            side: Order side.

        Returns:
            Fill price after slippage.
        """
        direction = 1.0 if side == Side.BUY else -1.0
        return price * (1 + direction * slippage_bps / 10000)

    @staticmethod
    def square_root(price: float, quantity: float, avg_daily_volume: float,
                    slippage_bps: float, side: Side) -> float:
        """Square-root slippage model (Almgren-Chriss inspired).

        Slippage proportional to sqrt(order_size / daily_volume).

        Args:
            price: Market price.
            quantity: Order quantity.
            avg_daily_volume: Average daily volume.
            slippage_bps: Base slippage in basis points.
            side: Order side.

        Returns:
            Fill price after slippage.
        """
        if avg_daily_volume <= 0:
            return price
        participation = quantity / avg_daily_volume
        impact_bps = slippage_bps * np.sqrt(participation)
        direction = 1.0 if side == Side.BUY else -1.0
        return price * (1 + direction * impact_bps / 10000)

    @staticmethod
    def almgren_chriss(price: float, quantity: float, total_volume: float,
                       sigma: float, eta: float = 0.1, side: Side = Side.BUY) -> float:
        """Almgren-Chriss slippage model.

        Models temporary and permanent market impact.

        Args:
            price: Market price.
            quantity: Order quantity.
            total_volume: Total market volume.
            sigma: Volatility.
            eta: Impact coefficient.
            side: Order side.

        Returns:
            Fill price after market impact.
        """
        if total_volume <= 0:
            return price
        participation = quantity / total_volume
        permanent_impact = eta * participation * price
        temporary_impact = eta * participation * np.sqrt(abs(quantity)) * price * 0.001
        total_impact = permanent_impact + temporary_impact
        direction = 1.0 if side == Side.BUY else -1.0
        return price + direction * total_impact

    @staticmethod
    def volume_dependent(price: float, quantity: float, current_volume: float,
                         normal_volume: float, base_slippage_bps: float,
                         side: Side) -> float:
        """Volume-dependent slippage model.

        Slippage increases when current volume is below normal.

        Args:
            price: Market price.
            quantity: Order quantity.
            current_volume: Current market volume.
            normal_volume: Normal average volume.
            base_slippage_bps: Base slippage in basis points.
            side: Order side.

        Returns:
            Fill price after volume-adjusted slippage.
        """
        if normal_volume <= 0:
            return SlippageModel.percentage(price, quantity, base_slippage_bps, side)
        volume_ratio = normal_volume / max(current_volume, 1e-10)
        adjusted_slippage = base_slippage_bps * volume_ratio
        adjusted_slippage = min(adjusted_slippage, base_slippage_bps * 10)  # Cap at 10x
        return SlippageModel.percentage(price, quantity, adjusted_slippage, side)


__all__ = ["SlippageModel"]
