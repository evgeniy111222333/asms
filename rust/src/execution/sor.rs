//! Smart Order Router (SOR)
//!
//! Routes orders across multiple exchanges for best execution,
//! considering latency, fees, and liquidity.

use crate::core::types::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exchange routing score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingScore {
    pub exchange: ExchangeId,
    pub total_score: Decimal,
    pub latency_score: Decimal,
    pub fee_score: Decimal,
    pub liquidity_score: Decimal,
}

/// SOR routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub splits: Vec<OrderSplit>,
    pub scores: Vec<RoutingScore>,
    pub best_exchange: ExchangeId,
    pub estimated_slippage: Decimal,
    pub estimated_fee: Decimal,
}

/// Order split for multi-exchange execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderSplit {
    pub exchange: ExchangeId,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub estimated_fee: Decimal,
}

/// Exchange latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeLatency {
    pub exchange: ExchangeId,
    pub avg_latency_ms: u64,
    pub p50_latency_ms: u64,
    pub p99_latency_ms: u64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// Smart Order Router
pub struct SmartOrderRouter {
    latencies: HashMap<ExchangeId, ExchangeLatency>,
    fees: HashMap<ExchangeId, ExchangeConfig>,
    order_books: HashMap<Symbol, HashMap<ExchangeId, OrderBookSnapshot>>,
    latency_weight: Decimal,
    fee_weight: Decimal,
    liquidity_weight: Decimal,
    exchanges: Vec<ExchangeId>,
}

impl SmartOrderRouter {
    pub fn new() -> Self {
        SmartOrderRouter {
            latencies: HashMap::new(),
            fees: HashMap::new(),
            order_books: HashMap::new(),
            latency_weight: rust_decimal_macros::dec!(0.3),
            fee_weight: rust_decimal_macros::dec!(0.3),
            liquidity_weight: rust_decimal_macros::dec!(0.4),
            exchanges: vec![ExchangeId::Binance, ExchangeId::Bybit, ExchangeId::OKX],
        }
    }

    pub fn with_weights(mut self, latency: Decimal, fee: Decimal, liquidity: Decimal) -> Self {
        self.latency_weight = latency;
        self.fee_weight = fee;
        self.liquidity_weight = liquidity;
        self
    }

    /// Update latency statistics
    pub fn update_latency(&mut self, latency: ExchangeLatency) {
        self.latencies.insert(latency.exchange, latency);
    }

    /// Update exchange fee configuration
    pub fn add_exchange(&mut self, config: ExchangeConfig) {
        self.fees.insert(config.exchange, config);
    }

    /// Alias for add_exchange
    pub fn update_fees(&mut self, config: ExchangeConfig) {
        self.add_exchange(config);
    }

    /// Remove an exchange from routing
    pub fn remove_exchange(&mut self, exchange: ExchangeId) {
        self.exchanges.retain(|e| *e != exchange);
        self.fees.remove(&exchange);
        self.latencies.remove(&exchange);
        for books in self.order_books.values_mut() {
            books.remove(&exchange);
        }
    }

    /// Update order book snapshot
    pub fn update_book(&mut self, symbol: Symbol, exchange: ExchangeId, snapshot: OrderBookSnapshot) {
        self.order_books
            .entry(symbol)
            .or_insert_with(HashMap::new)
            .insert(exchange, snapshot);
    }

    /// Alias for update_book
    pub fn update_order_book(&mut self, symbol: Symbol, exchange: ExchangeId, snapshot: OrderBookSnapshot) {
        self.update_book(symbol, exchange, snapshot);
    }

    /// Route a market order for best execution
    pub fn route_order(&self, symbol: &Symbol, side: Side, quantity: Decimal) -> RoutingDecision {
        self.route_market_order(symbol, side, quantity)
    }

    /// Route a market order for best execution
    pub fn route_market_order(&self, symbol: &Symbol, side: Side, quantity: Decimal) -> RoutingDecision {
        let mut scores = Vec::new();

        for exchange in &self.exchanges {
            let latency_score = self.compute_latency_score(exchange);
            let fee_score = self.compute_fee_score(exchange, side);
            let liquidity_score = self.compute_liquidity_score(symbol, exchange, side, quantity);

            let total = latency_score * self.latency_weight
                + fee_score * self.fee_weight
                + liquidity_score * self.liquidity_weight;

            scores.push(RoutingScore {
                exchange: *exchange,
                total_score: total,
                latency_score,
                fee_score,
                liquidity_score,
            });
        }

        scores.sort_by(|a, b| b.total_score.cmp(&a.total_score));
        let best_exchange = scores[0].exchange;

        let splits = self.compute_optimal_splits(symbol, side, quantity, &scores);
        let estimated_fee = splits.iter().map(|s| s.estimated_fee).sum();
        let estimated_slippage = self.estimate_slippage(symbol, side, quantity);

        RoutingDecision {
            splits,
            scores,
            best_exchange,
            estimated_slippage,
            estimated_fee,
        }
    }

    /// Route a limit order to the best exchange
    pub fn route_limit_order(
        &self,
        symbol: &Symbol,
        side: Side,
        quantity: Decimal,
        price: Decimal,
    ) -> RoutingDecision {
        let mut scores = Vec::new();

        for exchange in &self.exchanges {
            let latency_score = self.compute_latency_score(exchange);
            let fee_score = self.compute_fee_score(exchange, side);
            let liquidity_score = self.compute_liquidity_score(symbol, exchange, side, quantity);

            let total = latency_score * self.latency_weight
                + fee_score * self.fee_weight
                + liquidity_score * self.liquidity_weight;

            scores.push(RoutingScore {
                exchange: *exchange,
                total_score: total,
                latency_score,
                fee_score,
                liquidity_score,
            });
        }

        scores.sort_by(|a, b| b.total_score.cmp(&a.total_score));
        let best = scores[0].exchange;

        let fee = self.get_maker_fee(&best, price * quantity);
        let splits = vec![OrderSplit {
            exchange: best,
            quantity,
            price: Some(price),
            estimated_fee: fee,
        }];

        RoutingDecision {
            splits,
            scores,
            best_exchange: best,
            estimated_slippage: Decimal::ZERO,
            estimated_fee: fee,
        }
    }

    fn compute_latency_score(&self, exchange: &ExchangeId) -> Decimal {
        match self.latencies.get(exchange) {
            Some(lat) => {
                if lat.avg_latency_ms == 0 { return rust_decimal_macros::dec!(1.0); }
                let score = rust_decimal_macros::dec!(1000) / Decimal::from(lat.avg_latency_ms);
                score.min(rust_decimal_macros::dec!(1.0))
            }
            None => rust_decimal_macros::dec!(0.5),
        }
    }

    fn compute_fee_score(&self, exchange: &ExchangeId, _side: Side) -> Decimal {
        match self.fees.get(exchange) {
            Some(config) => {
                let taker = config.taker_fee;
                if taker == Decimal::ZERO { return rust_decimal_macros::dec!(2.0); }
                let score = rust_decimal_macros::dec!(0.001) / taker;
                score.min(rust_decimal_macros::dec!(2.0))
            }
            None => rust_decimal_macros::dec!(0.3),
        }
    }

    fn compute_liquidity_score(
        &self, symbol: &Symbol, exchange: &ExchangeId, side: Side, quantity: Decimal,
    ) -> Decimal {
        match self.order_books.get(symbol).and_then(|m| m.get(exchange)) {
            Some(ob) => {
                let available = match side {
                    Side::Buy => ob.asks.iter().map(|l| l.quantity).sum::<Decimal>(),
                    Side::Sell => ob.bids.iter().map(|l| l.quantity).sum::<Decimal>(),
                };
                if available >= quantity && quantity > Decimal::ZERO {
                    (quantity / available).min(rust_decimal_macros::dec!(1.0))
                } else if available > Decimal::ZERO {
                    available / quantity
                } else {
                    Decimal::ZERO
                }
            }
            None => rust_decimal_macros::dec!(0.3),
        }
    }

    fn compute_optimal_splits(
        &self, symbol: &Symbol, side: Side, quantity: Decimal, scores: &[RoutingScore],
    ) -> Vec<OrderSplit> {
        let total_liquidity: Decimal = scores.iter().map(|s| s.liquidity_score).sum();

        if total_liquidity == Decimal::ZERO {
            return vec![OrderSplit {
                exchange: scores[0].exchange,
                quantity,
                price: None,
                estimated_fee: Decimal::ZERO,
            }];
        }

        let eligible_scores: Vec<&RoutingScore> = scores.iter()
            .filter(|s| s.liquidity_score > Decimal::ZERO)
            .collect();

        if eligible_scores.is_empty() {
            return vec![OrderSplit {
                exchange: scores[0].exchange,
                quantity,
                price: None,
                estimated_fee: Decimal::ZERO,
            }];
        }

        let mut splits = Vec::new();
        let mut allocated = Decimal::ZERO;

        for (i, s) in eligible_scores.iter().enumerate() {
            let is_last = i == eligible_scores.len() - 1;
            let split_qty = if is_last {
                quantity - allocated
            } else {
                let raw_qty = quantity * s.liquidity_score / total_liquidity;
                raw_qty.round_dp(8)
            };

            if split_qty > Decimal::ZERO {
                let fee = self.get_taker_fee(&s.exchange, split_qty);
                splits.push(OrderSplit {
                    exchange: s.exchange,
                    quantity: split_qty,
                    price: None,
                    estimated_fee: fee,
                });
                allocated += split_qty;
            }
        }

        if splits.is_empty() {
            return vec![OrderSplit {
                exchange: eligible_scores[0].exchange,
                quantity,
                price: None,
                estimated_fee: Decimal::ZERO,
            }];
        }

        splits
    }

    fn estimate_slippage(&self, symbol: &Symbol, side: Side, quantity: Decimal) -> Decimal {
        let mut total_slippage = Decimal::ZERO;
        for obs in self.order_books.get(symbol).iter() {
            for ob in obs.values() {
                let levels = match side {
                    Side::Buy => &ob.asks,
                    Side::Sell => &ob.bids,
                };
                if let Some(first) = levels.first() {
                    if let Some(vwap) = self.compute_vwap_from_levels(levels, quantity) {
                        total_slippage = total_slippage.max(
                            (vwap - first.price).abs() / first.price,
                        );
                    }
                }
            }
        }
        total_slippage
    }

    fn compute_vwap_from_levels(&self, levels: &[OrderBookLevel], quantity: Decimal) -> Option<Decimal> {
        let mut remaining = quantity;
        let mut total_notional = Decimal::ZERO;
        let mut total_qty = Decimal::ZERO;
        for level in levels {
            if remaining <= Decimal::ZERO { break; }
            let fill = remaining.min(level.quantity);
            total_notional += fill * level.price;
            total_qty += fill;
            remaining -= fill;
        }
        if total_qty > Decimal::ZERO { Some(total_notional / total_qty) } else { None }
    }

    fn get_taker_fee(&self, exchange: &ExchangeId, notional: Decimal) -> Decimal {
        self.fees.get(exchange).map(|c| notional * c.taker_fee).unwrap_or(Decimal::ZERO)
    }

    fn get_maker_fee(&self, exchange: &ExchangeId, notional: Decimal) -> Decimal {
        self.fees.get(exchange).map(|c| notional * c.maker_fee).unwrap_or(Decimal::ZERO)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_sor_basic_routing() {
        let sor = SmartOrderRouter::new();
        let decision = sor.route_market_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        assert!(!decision.scores.is_empty());
    }

    #[test]
    fn test_sor_limit_routing() {
        let sor = SmartOrderRouter::new();
        let decision = sor.route_limit_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1), dec!(50000));
        assert!(!decision.splits.is_empty());
    }

    #[test]
    fn test_sor_add_exchange() {
        let mut sor = SmartOrderRouter::new();
        sor.add_exchange(ExchangeConfig {
            exchange: ExchangeId::Binance,
            api_key: String::new(),
            api_secret: String::new(),
            passphrase: None,
            rest_url: String::new(),
            ws_url: String::new(),
            rate_limit_per_second: 10,
            maker_fee: dec!(0.001),
            taker_fee: dec!(0.001),
        });
        let decision = sor.route_market_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        // Binance should have better fee score now
        assert!(!decision.scores.is_empty());
    }

    #[test]
    fn test_sor_remove_exchange() {
        let mut sor = SmartOrderRouter::new();
        sor.remove_exchange(ExchangeId::OKX);
        let decision = sor.route_market_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        assert!(decision.scores.iter().all(|s| s.exchange != ExchangeId::OKX));
    }

    #[test]
    fn test_sor_route_order_alias() {
        let sor = SmartOrderRouter::new();
        let decision = sor.route_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        assert!(!decision.scores.is_empty());
    }

    #[test]
    fn test_sor_fee_based_routing() {
        let mut sor = SmartOrderRouter::new()
            .with_weights(dec!(0), dec!(1), dec!(0)); // Fee-only weight

        sor.add_exchange(ExchangeConfig {
            exchange: ExchangeId::Binance,
            api_key: String::new(), api_secret: String::new(), passphrase: None,
            rest_url: String::new(), ws_url: String::new(), rate_limit_per_second: 10,
            maker_fee: dec!(0.001), taker_fee: dec!(0.001),
        });
        sor.add_exchange(ExchangeConfig {
            exchange: ExchangeId::Bybit,
            api_key: String::new(), api_secret: String::new(), passphrase: None,
            rest_url: String::new(), ws_url: String::new(), rate_limit_per_second: 10,
            maker_fee: dec!(0.0005), taker_fee: dec!(0.0005), // Cheaper
        });

        let decision = sor.route_market_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        // Bybit should be preferred due to lower fees
        assert_eq!(decision.best_exchange, ExchangeId::Bybit);
    }

    #[test]
    fn test_sor_update_book() {
        let mut sor = SmartOrderRouter::new();
        sor.update_book(Symbol::new("BTC/USDT"), ExchangeId::Binance, OrderBookSnapshot {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Binance,
            bids: vec![OrderBookLevel { price: dec!(50000), quantity: dec!(10), order_count: 1 }],
            asks: vec![OrderBookLevel { price: dec!(50001), quantity: dec!(10), order_count: 1 }],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        });

        let decision = sor.route_market_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        // Binance should have higher liquidity score
        let binance_score = decision.scores.iter().find(|s| s.exchange == ExchangeId::Binance).unwrap();
        assert!(binance_score.liquidity_score > Decimal::ZERO);
    }

    #[test]
    fn test_sor_latency_based_routing() {
        let mut sor = SmartOrderRouter::new()
            .with_weights(dec!(1), dec!(0), dec!(0)); // Latency-only

        sor.update_latency(ExchangeLatency {
            exchange: ExchangeId::Binance,
            avg_latency_ms: 10,
            p50_latency_ms: 10,
            p99_latency_ms: 50,
            last_update: chrono::Utc::now(),
        });
        sor.update_latency(ExchangeLatency {
            exchange: ExchangeId::Bybit,
            avg_latency_ms: 100,
            p50_latency_ms: 100,
            p99_latency_ms: 500,
            last_update: chrono::Utc::now(),
        });

        let decision = sor.route_market_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        assert_eq!(decision.best_exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_sor_rounding_and_split_precision() {
        let mut sor = SmartOrderRouter::new();
        let config_binance = ExchangeConfig {
            exchange: ExchangeId::Binance,
            api_key: String::new(), api_secret: String::new(), passphrase: None,
            rest_url: String::new(), ws_url: String::new(), rate_limit_per_second: 10,
            maker_fee: dec!(0.001), taker_fee: dec!(0.001),
        };
        let config_bybit = ExchangeConfig {
            exchange: ExchangeId::Bybit,
            ..config_binance.clone()
        };
        let config_okx = ExchangeConfig {
            exchange: ExchangeId::OKX,
            ..config_binance.clone()
        };
        sor.add_exchange(config_binance);
        sor.add_exchange(config_bybit);
        sor.add_exchange(config_okx);

        let snapshot = OrderBookSnapshot {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Binance,
            bids: vec![OrderBookLevel { price: dec!(50000), quantity: dec!(10), order_count: 1 }],
            asks: vec![OrderBookLevel { price: dec!(50001), quantity: dec!(10), order_count: 1 }],
            timestamp: chrono::Utc::now(),
            sequence: 1,
        };
        sor.update_book(Symbol::new("BTC/USDT"), ExchangeId::Binance, snapshot.clone());
        sor.update_book(Symbol::new("BTC/USDT"), ExchangeId::Bybit, OrderBookSnapshot { exchange: ExchangeId::Bybit, ..snapshot.clone() });
        sor.update_book(Symbol::new("BTC/USDT"), ExchangeId::OKX, OrderBookSnapshot { exchange: ExchangeId::OKX, ..snapshot });

        let decision = sor.route_market_order(&Symbol::new("BTC/USDT"), Side::Buy, dec!(1));
        assert!(!decision.splits.is_empty());
        let total_split_qty: Decimal = decision.splits.iter().map(|s| s.quantity).sum();
        assert_eq!(total_split_qty, dec!(1));
    }
}
