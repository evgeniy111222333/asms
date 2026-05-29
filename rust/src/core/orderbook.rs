//! High-performance L3 Order Book implementation
//!
//! Uses BTreeMap for O(log n) price-level lookups and supports
//! real-time updates via crossbeam channels.

use crate::core::types::*;
use chrono::Utc;
use crossbeam_channel::{Receiver, Sender};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Order book update event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderBookEvent {
    Add {
        price: Decimal,
        quantity: Decimal,
        side: Side,
        order_count: u32,
    },
    Remove {
        price: Decimal,
        side: Side,
    },
    Update {
        price: Decimal,
        quantity: Decimal,
        side: Side,
        order_count: u32,
    },
    Clear,
}

/// High-performance order book
pub struct OrderBook {
    symbol: Symbol,
    exchange: ExchangeId,
    bids: BTreeMap<Decimal, OrderBookLevel>,
    asks: BTreeMap<Decimal, OrderBookLevel>,
    sequence: u64,
    last_update: chrono::DateTime<Utc>,
    event_tx: Sender<OrderBookEvent>,
    event_rx: Receiver<OrderBookEvent>,
    max_depth: usize,
}

impl OrderBook {
    pub fn new(symbol: Symbol, exchange: ExchangeId, max_depth: usize) -> Self {
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        OrderBook {
            symbol,
            exchange,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            sequence: 0,
            last_update: Utc::now(),
            event_tx,
            event_rx,
            max_depth,
        }
    }

    /// Add or update a bid level
    pub fn update_bid(&mut self, price: Decimal, quantity: Decimal, order_count: u32) {
        if quantity == Decimal::ZERO {
            self.bids.remove(&price);
            let _ = self.event_tx.send(OrderBookEvent::Remove {
                price,
                side: Side::Buy,
            });
        } else {
            self.bids.insert(
                price,
                OrderBookLevel {
                    price,
                    quantity,
                    order_count,
                },
            );
            let _ = self.event_tx.send(OrderBookEvent::Update {
                price,
                quantity,
                side: Side::Buy,
                order_count,
            });
        }
        self.sequence += 1;
        self.last_update = Utc::now();
    }

    /// Add or update an ask level
    pub fn update_ask(&mut self, price: Decimal, quantity: Decimal, order_count: u32) {
        if quantity == Decimal::ZERO {
            self.asks.remove(&price);
            let _ = self.event_tx.send(OrderBookEvent::Remove {
                price,
                side: Side::Sell,
            });
        } else {
            self.asks.insert(
                price,
                OrderBookLevel {
                    price,
                    quantity,
                    order_count,
                },
            );
            let _ = self.event_tx.send(OrderBookEvent::Update {
                price,
                quantity,
                side: Side::Sell,
                order_count,
            });
        }
        self.sequence += 1;
        self.last_update = Utc::now();
    }

    /// Apply a full snapshot
    pub fn apply_snapshot(&mut self, bids: Vec<OrderBookLevel>, asks: Vec<OrderBookLevel>) {
        self.bids.clear();
        self.asks.clear();
        for level in bids {
            self.bids.insert(level.price, level);
        }
        for level in asks {
            self.asks.insert(level.price, level);
        }
        self.sequence += 1;
        self.last_update = Utc::now();
        let _ = self.event_tx.send(OrderBookEvent::Clear);
    }

    /// Get top N bid levels (sorted high to low)
    pub fn get_bids(&self, levels: usize) -> Vec<&OrderBookLevel> {
        self.bids.iter().rev().take(levels).map(|(_, v)| v).collect()
    }

    /// Get top N ask levels (sorted low to high)
    pub fn get_asks(&self, levels: usize) -> Vec<&OrderBookLevel> {
        self.asks.iter().take(levels).map(|(_, v)| v).collect()
    }

    /// Get best bid price
    pub fn best_bid(&self) -> Option<Decimal> {
        self.bids.iter().rev().next().map(|(p, _)| *p)
    }

    /// Get best ask price
    pub fn best_ask(&self) -> Option<Decimal> {
        self.asks.iter().next().map(|(p, _)| *p)
    }

    /// Compute midpoint price
    pub fn compute_midpoint(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / rust_decimal_macros::dec!(2)),
            _ => None,
        }
    }

    /// Compute weighted midpoint (bid/ask size weighted)
    pub fn compute_weighted_mid(&self) -> Option<Decimal> {
        let bid = self.bids.iter().rev().next();
        let ask = self.asks.iter().next();
        match (bid, ask) {
            (Some((bp, bl)), Some((ap, al))) => {
                let total = bl.quantity + al.quantity;
                if total == Decimal::ZERO {
                    return Some((*bp + *ap) / rust_decimal_macros::dec!(2));
                }
                Some((*bp * al.quantity + *ap * bl.quantity) / total)
            }
            _ => None,
        }
    }

    /// Compute VWAP from order book depth
    pub fn compute_vwap(&self, side: Side, quantity: Decimal) -> Option<Decimal> {
        let levels: Vec<&OrderBookLevel> = match side {
            Side::Buy => self.asks.iter().map(|(_, v)| v).collect(),
            Side::Sell => self.bids.iter().rev().map(|(_, v)| v).collect(),
        };

        let mut remaining = quantity;
        let mut total_notional = Decimal::ZERO;
        let mut total_qty = Decimal::ZERO;

        for level in levels {
            if remaining <= Decimal::ZERO {
                break;
            }
            let fill_qty = remaining.min(level.quantity);
            total_notional += fill_qty * level.price;
            total_qty += fill_qty;
            remaining -= fill_qty;
        }

        if total_qty > Decimal::ZERO {
            Some(total_notional / total_qty)
        } else {
            None
        }
    }

    /// Compute bid-ask spread
    pub fn compute_spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Compute spread as percentage of midpoint
    pub fn compute_spread_pct(&self) -> Option<Decimal> {
        match (self.compute_spread(), self.compute_midpoint()) {
            (Some(spread), Some(mid)) if mid > Decimal::ZERO => {
                Some(spread / mid * rust_decimal_macros::dec!(100))
            }
            _ => None,
        }
    }

    /// Compute order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    /// Range: [-1, 1], positive = bid-heavy (buying pressure)
    pub fn compute_imbalance(&self, levels: usize) -> Decimal {
        let bid_vol: Decimal = self
            .get_bids(levels)
            .iter()
            .map(|l| l.quantity)
            .sum();
        let ask_vol: Decimal = self
            .get_asks(levels)
            .iter()
            .map(|l| l.quantity)
            .sum();
        let total = bid_vol + ask_vol;
        if total == Decimal::ZERO {
            return Decimal::ZERO;
        }
        (bid_vol - ask_vol) / total
    }

    /// Total bid volume up to N levels
    pub fn total_bid_volume(&self, levels: usize) -> Decimal {
        self.get_bids(levels).iter().map(|l| l.quantity).sum()
    }

    /// Total ask volume up to N levels
    pub fn total_ask_volume(&self, levels: usize) -> Decimal {
        self.get_asks(levels).iter().map(|l| l.quantity).sum()
    }

    /// Create an OrderBookSnapshot
    pub fn snapshot(&self) -> OrderBookSnapshot {
        OrderBookSnapshot {
            symbol: self.symbol.clone(),
            exchange: self.exchange,
            bids: self.get_bids(self.max_depth).into_iter().cloned().collect(),
            asks: self.get_asks(self.max_depth).into_iter().cloned().collect(),
            timestamp: self.last_update,
            sequence: self.sequence,
        }
    }

    /// Subscribe to order book events
    pub fn subscribe(&self) -> Receiver<OrderBookEvent> {
        self.event_rx.clone()
    }

    /// Number of bid levels
    pub fn bid_depth(&self) -> usize {
        self.bids.len()
    }

    /// Number of ask levels
    pub fn ask_depth(&self) -> usize {
        self.asks.len()
    }

    /// Get the symbol
    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_orderbook_basic() {
        let mut ob = OrderBook::new(
            Symbol::new("BTC/USDT"),
            ExchangeId::Binance,
            25,
        );

        ob.update_bid(dec!(50000), dec!(1.5), 3);
        ob.update_bid(dec!(49999), dec!(2.0), 5);
        ob.update_ask(dec!(50001), dec!(1.0), 2);
        ob.update_ask(dec!(50002), dec!(0.5), 1);

        assert_eq!(ob.best_bid(), Some(dec!(50000)));
        assert_eq!(ob.best_ask(), Some(dec!(50001)));
        assert_eq!(ob.compute_midpoint(), Some(dec!(50000.5)));
        assert_eq!(ob.compute_spread(), Some(dec!(1)));
    }

    #[test]
    fn test_weighted_midpoint() {
        let mut ob = OrderBook::new(
            Symbol::new("BTC/USDT"),
            ExchangeId::Binance,
            25,
        );

        ob.update_bid(dec!(100), dec!(10), 1);
        ob.update_ask(dec!(101), dec!(5), 1);

        let wmid = ob.compute_weighted_mid().unwrap();
        // (100*5 + 101*10) / 15 = (500 + 1010) / 15 = 1510/15 = 100.666...
        assert!(wmid > dec!(100) && wmid < dec!(101));
    }

    #[test]
    fn test_imbalance() {
        let mut ob = OrderBook::new(
            Symbol::new("BTC/USDT"),
            ExchangeId::Binance,
            25,
        );

        ob.update_bid(dec!(100), dec!(80), 1);
        ob.update_ask(dec!(101), dec!(20), 1);

        let imbalance = ob.compute_imbalance(5);
        // (80 - 20) / 100 = 0.6
        assert!(imbalance > dec!(0.5));
    }

    #[test]
    fn test_vwap() {
        let mut ob = OrderBook::new(
            Symbol::new("BTC/USDT"),
            ExchangeId::Binance,
            25,
        );

        ob.update_ask(dec!(100), dec!(5), 1);
        ob.update_ask(dec!(101), dec!(5), 1);

        let vwap = ob.compute_vwap(Side::Buy, dec!(6)).unwrap();
        // 5*100 + 1*101 = 601 / 6 = 100.166...
        assert!(vwap > dec!(100) && vwap < dec!(101));
    }
}
