"""Arbitrage Detection for ACMS."""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Any
from datetime import datetime

from acms.exchanges.base import ExchangeAdapter, ExchangeCredentials
from acms.exchanges.binance import BinanceAdapter
from acms.exchanges.bybit import BybitAdapter
from acms.exchanges.okx import OKXAdapter
from acms.exchanges.paper import PaperTradingAdapter


@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity."""
    symbol: str
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_pct: float
    estimated_profit_usd: float
    buy_fee_usd: float
    sell_fee_usd: float
    timestamp: datetime


class ArbitrageDetector:
    """Multi-exchange arbitrage detection engine.

    Monitors price differences across exchanges and identifies
    fee-adjusted profitable opportunities.
    """

    def __init__(self, exchanges: Dict[str, ExchangeAdapter],
                 default_fees_bps: Dict[str, float] = None):
        self.exchanges = exchanges
        self.fees_bps = default_fees_bps or {
            "binance": 10.0, "bybit": 10.0, "okx": 10.0, "paper": 0.0,
        }
        self._latest_prices: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update_price(self, exchange: str, symbol: str, price: float) -> None:
        """Update the latest price for an exchange/symbol pair.

        Args:
            exchange: Exchange name.
            symbol: Trading pair symbol.
            price: Latest price.
        """
        self._latest_prices[symbol][exchange] = price

    async def fetch_all_prices(self, symbol: str) -> Dict[str, float]:
        """Fetch current prices from all exchanges for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Dict mapping exchange name to price.
        """
        prices = {}
        for name, adapter in self.exchanges.items():
            try:
                ob = await adapter.get_order_book(symbol, depth=1)
                if ob.get("asks") and ob.get("bids"):
                    best_ask = ob["asks"][0][0]
                    best_bid = ob["bids"][0][0]
                    mid = (best_ask + best_bid) / 2.0
                    prices[name] = mid
                    self._latest_prices[symbol][name] = mid
            except Exception as e:
                logger.warning("Failed to fetch price from %s: %s", name, e)
        return prices

    def detect_opportunities(self, symbol: str, trade_size: float = 1.0) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities for a symbol.

        Compares prices across exchanges and identifies fee-adjusted
        profitable opportunities.

        Args:
            symbol: Trading pair symbol.
            trade_size: Size in base currency for profit estimation.

        Returns:
            List of ArbitrageOpportunity sorted by estimated profit.
        """
        exchange_prices = self._latest_prices.get(symbol, {})
        if len(exchange_prices) < 2:
            return []

        opportunities = []
        exchange_names = list(exchange_prices.keys())

        for i in range(len(exchange_names)):
            for j in range(len(exchange_names)):
                if i == j:
                    continue
                buy_exchange = exchange_names[i]
                sell_exchange = exchange_names[j]
                buy_price = exchange_prices[buy_exchange]
                sell_price = exchange_prices[sell_exchange]

                if buy_price <= 0 or sell_price <= 0:
                    continue

                spread_pct = (sell_price - buy_price) / buy_price
                buy_fee = buy_price * trade_size * self.fees_bps.get(buy_exchange, 10) / 10000
                sell_fee = sell_price * trade_size * self.fees_bps.get(sell_exchange, 10) / 10000
                gross_profit = (sell_price - buy_price) * trade_size
                net_profit = gross_profit - buy_fee - sell_fee

                if net_profit > 0:
                    opportunities.append(ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=buy_exchange,
                        sell_exchange=sell_exchange,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        spread_pct=spread_pct * 100,
                        estimated_profit_usd=net_profit,
                        buy_fee_usd=buy_fee,
                        sell_fee_usd=sell_fee,
                        timestamp=datetime.utcnow(),
                    ))

        opportunities.sort(key=lambda x: x.estimated_profit_usd, reverse=True)
        return opportunities


# Exchange Factory

def create_exchange_adapter(exchange: str, credentials: Optional[ExchangeCredentials] = None,
                            testnet: bool = False) -> ExchangeAdapter:
    """Factory function to create exchange adapters.

    Args:
        exchange: Exchange name ('binance', 'bybit', 'okx', 'paper').
        credentials: Exchange API credentials.
        testnet: Whether to use testnet endpoints.

    Returns:
        ExchangeAdapter instance.

    Raises:
        ValueError: If exchange name is unknown.
    """
    creds = credentials or ExchangeCredentials()
    if exchange == "binance":
        return BinanceAdapter(creds, testnet)
    elif exchange == "bybit":
        return BybitAdapter(creds, testnet)
    elif exchange == "okx":
        return OKXAdapter(creds, testnet)
    elif exchange == "paper":
        return PaperTradingAdapter()
    else:
        raise ValueError(f"Unknown exchange: {exchange}")

__all__ = ['ArbitrageOpportunity', 'ArbitrageDetector', 'create_exchange_adapter']
