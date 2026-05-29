"""Exchange Adapters - Real WebSocket/REST connections.

Implements adapters for:
- Binance (REST + WebSocket)
- Bybit (REST + WebSocket with auto-reconnect)
- OKX (REST + WebSocket with auto-reconnect)
- Paper Trading (simulation)

Additional features:
- Rate limiting with token bucket algorithm
- Order book depth streaming with local book management
- Funding rate fetching and streaming
- Multi-exchange arbitrage detection
- Unified error handling and retry logic
"""

from acms.exchanges.rate_limiter import RateLimiter
from acms.exchanges.order_book import LocalOrderBook
from acms.exchanges.base import (
    ExchangeError, RateLimitError, InsufficientFundsError,
    OrderNotFoundError, NetworkError, classify_exchange_error,
    retry_with_backoff, ExchangeCredentials, ExchangeAdapter,
)
from acms.exchanges.binance import BinanceAdapter
from acms.exchanges.bybit import BybitAdapter
from acms.exchanges.okx import OKXAdapter
from acms.exchanges.paper import PaperTradingAdapter
from acms.exchanges.arbitrage import ArbitrageOpportunity, ArbitrageDetector, create_exchange_adapter

__all__ = [
    # Base classes and utilities
    "RateLimiter", "LocalOrderBook", "ExchangeError", "RateLimitError",
    "InsufficientFundsError", "OrderNotFoundError", "NetworkError",
    "classify_exchange_error", "retry_with_backoff",
    "ExchangeCredentials", "ExchangeAdapter",
    # Exchange adapters
    "BinanceAdapter", "BybitAdapter", "OKXAdapter", "PaperTradingAdapter",
    # Arbitrage
    "ArbitrageOpportunity", "ArbitrageDetector", "create_exchange_adapter",
]
