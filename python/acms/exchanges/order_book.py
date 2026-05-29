"""Local Order Book Management for Exchange Adapters."""

from typing import Optional, Dict, List, Tuple
from datetime import datetime


class LocalOrderBook:
    """Local order book manager for depth streaming.

    Maintains a local copy of the order book updated via
    WebSocket depth stream messages.
    """

    def __init__(self, symbol: str, max_depth: int = 50):
        self.symbol = symbol
        self.max_depth = max_depth
        self.bids: Dict[float, float] = {}  # price -> quantity
        self.asks: Dict[float, float] = {}
        self.last_update_id: int = 0
        self.updated_at: Optional[datetime] = None

    def update(self, bids: List[Tuple[float, float]], asks: List[Tuple[float, float]],
               update_id: int = 0) -> None:
        """Update the local order book with new data.

        Args:
            bids: List of (price, quantity) tuples. Quantity 0 removes level.
            asks: List of (price, quantity) tuples.
            update_id: Sequential update identifier.
        """
        if update_id <= self.last_update_id and update_id != 0:
            return

        for price, qty in bids:
            if qty <= 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty

        for price, qty in asks:
            if qty <= 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        self.last_update_id = update_id
        self.updated_at = datetime.utcnow()
        self._trim_depth()

    def _trim_depth(self) -> None:
        """Trim order book to max_depth levels."""
        if len(self.bids) > self.max_depth:
            sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])
            self.bids = dict(sorted_bids[:self.max_depth])
        if len(self.asks) > self.max_depth:
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
            self.asks = dict(sorted_asks[:self.max_depth])

    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return min(self.asks.keys()) if self.asks else None

    def get_spread(self) -> Optional[float]:
        """Get current bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        return None

    def snapshot(self) -> Dict:
        """Get a snapshot of the order book.

        Returns:
            Dict with sorted bids and asks lists.
        """
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
        return {
            "symbol": self.symbol,
            "bids": [(p, q) for p, q in sorted_bids],
            "asks": [(p, q) for p, q in sorted_asks],
            "spread": self.get_spread(),
            "mid_price": self.get_mid_price(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


__all__ = ['LocalOrderBook']
