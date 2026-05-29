//! Data Normalizer
//!
//! Normalizes tick data from different exchanges to a unified format,
//! handles precision differences, timestamp normalization, and symbol mapping.

use crate::core::types::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exchange-specific tick format (before normalization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTick {
    pub exchange: ExchangeId,
    pub symbol: String, // Exchange-specific symbol format
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: Option<Side>,
    pub trade_id: String,
    pub exchange_timestamp: i64, // Milliseconds from exchange
    pub local_timestamp: i64,   // Milliseconds local
}

/// Exchange precision configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangePrecision {
    pub exchange: ExchangeId,
    pub price_precision: u32,
    pub quantity_precision: u32,
    pub timestamp_is_ms: bool, // true = milliseconds, false = microseconds
}

/// Data normalizer
pub struct DataNormalizer {
    symbol_maps: HashMap<ExchangeId, HashMap<String, Symbol>>,
    precisions: HashMap<ExchangeId, ExchangePrecision>,
    time_offset_ms: HashMap<ExchangeId, i64>, // Offset between exchange and local time
}

impl DataNormalizer {
    pub fn new() -> Self {
        let mut normalizer = DataNormalizer {
            symbol_maps: HashMap::new(),
            precisions: HashMap::new(),
            time_offset_ms: HashMap::new(),
        };

        // Register default symbol mappings
        normalizer.register_exchange_defaults();
        normalizer
    }

    /// Register default exchange configurations
    fn register_exchange_defaults(&mut self) {
        // Binance symbol mapping
        let mut binance_symbols = HashMap::new();
        binance_symbols.insert("BTCUSDT".to_string(), Symbol::new("BTC/USDT"));
        binance_symbols.insert("ETHUSDT".to_string(), Symbol::new("ETH/USDT"));
        binance_symbols.insert("SOLUSDT".to_string(), Symbol::new("SOL/USDT"));
        binance_symbols.insert("BNBUSDT".to_string(), Symbol::new("BNB/USDT"));
        binance_symbols.insert("XRPUSDT".to_string(), Symbol::new("XRP/USDT"));
        binance_symbols.insert("ADAUSDT".to_string(), Symbol::new("ADA/USDT"));
        binance_symbols.insert("DOGEUSDT".to_string(), Symbol::new("DOGE/USDT"));
        binance_symbols.insert("AVAXUSDT".to_string(), Symbol::new("AVAX/USDT"));
        binance_symbols.insert("DOTUSDT".to_string(), Symbol::new("DOT/USDT"));
        binance_symbols.insert("LINKUSDT".to_string(), Symbol::new("LINK/USDT"));

        self.symbol_maps.insert(ExchangeId::Binance, binance_symbols);

        // Bybit symbol mapping
        let mut bybit_symbols = HashMap::new();
        bybit_symbols.insert("BTCUSDT".to_string(), Symbol::new("BTC/USDT"));
        bybit_symbols.insert("ETHUSDT".to_string(), Symbol::new("ETH/USDT"));
        bybit_symbols.insert("SOLUSDT".to_string(), Symbol::new("SOL/USDT"));

        self.symbol_maps.insert(ExchangeId::Bybit, bybit_symbols);

        // OKX symbol mapping (uses instId format: BTC-USDT)
        let mut okx_symbols = HashMap::new();
        okx_symbols.insert("BTC-USDT".to_string(), Symbol::new("BTC/USDT"));
        okx_symbols.insert("ETH-USDT".to_string(), Symbol::new("ETH/USDT"));
        okx_symbols.insert("SOL-USDT".to_string(), Symbol::new("SOL/USDT"));

        self.symbol_maps.insert(ExchangeId::OKX, okx_symbols);

        // Precision configs
        self.precisions.insert(
            ExchangeId::Binance,
            ExchangePrecision {
                exchange: ExchangeId::Binance,
                price_precision: 2,
                quantity_precision: 6,
                timestamp_is_ms: true,
            },
        );
        self.precisions.insert(
            ExchangeId::Bybit,
            ExchangePrecision {
                exchange: ExchangeId::Bybit,
                price_precision: 2,
                quantity_precision: 6,
                timestamp_is_ms: true,
            },
        );
        self.precisions.insert(
            ExchangeId::OKX,
            ExchangePrecision {
                exchange: ExchangeId::OKX,
                price_precision: 2,
                quantity_precision: 6,
                timestamp_is_ms: true,
            },
        );
    }

    /// Add a custom symbol mapping
    pub fn add_symbol_mapping(&mut self, exchange: ExchangeId, exchange_symbol: &str, canonical: Symbol) {
        self.symbol_maps
            .entry(exchange)
            .or_insert_with(HashMap::new)
            .insert(exchange_symbol.to_string(), canonical);
    }

    /// Normalize a raw tick into a unified Tick
    pub fn normalize_tick(&self, raw: &RawTick) -> Result<Tick, String> {
        let symbol = self
            .symbol_maps
            .get(&raw.exchange)
            .and_then(|m| m.get(&raw.symbol))
            .ok_or_else(|| format!("Unknown symbol {} for exchange {}", raw.symbol, raw.exchange))?
            .clone();

        let side = raw.side.unwrap_or(Side::Buy); // Default if not provided

        let precision = self.precisions.get(&raw.exchange);
        let price = match precision {
            Some(p) => raw.price.round_dp(p.price_precision),
            None => raw.price,
        };
        let quantity = match precision {
            Some(p) => raw.quantity.round_dp(p.quantity_precision),
            None => raw.quantity,
        };

        let timestamp_secs = if precision.map(|p| p.timestamp_is_ms).unwrap_or(true) {
            raw.exchange_timestamp / 1000
        } else {
            raw.exchange_timestamp / 1_000_000
        };

        Ok(Tick {
            symbol,
            exchange: raw.exchange,
            price,
            quantity,
            side,
            timestamp: chrono::DateTime::from_timestamp(timestamp_secs, 0)
                .unwrap_or(chrono::Utc::now()),
            trade_id: raw.trade_id.clone(),
        })
    }

    /// Compute time offset between exchange and local time
    pub fn compute_time_offset(&mut self, exchange: ExchangeId, exchange_ts_ms: i64, local_ts_ms: i64) {
        let offset = local_ts_ms - exchange_ts_ms;
        self.time_offset_ms.insert(exchange, offset);
    }

    /// Get time offset for an exchange
    pub fn get_time_offset(&self, exchange: ExchangeId) -> i64 {
        self.time_offset_ms.get(&exchange).copied().unwrap_or(0)
    }

    /// Validate data quality
    pub fn validate_tick(&self, tick: &Tick, prev_tick: Option<&Tick>) -> DataQuality {
        let mut issues = Vec::new();

        // Check for zero price/quantity
        if tick.price <= Decimal::ZERO {
            issues.push("Zero or negative price".to_string());
        }
        if tick.quantity <= Decimal::ZERO {
            issues.push("Zero or negative quantity".to_string());
        }

        // Check for stale data (if > 30 seconds old)
        let age = (chrono::Utc::now() - tick.timestamp).num_seconds();
        if age > 30 {
            issues.push(format!("Stale data: {} seconds old", age));
        }

        // Check for price jumps (> 10% in one tick)
        if let Some(prev) = prev_tick {
            if prev.price > Decimal::ZERO {
                let change = (tick.price - prev.price).abs() / prev.price;
                if change > rust_decimal_macros::dec!(0.10) {
                    issues.push(format!("Price jump: {:.2}%", change * rust_decimal_macros::dec!(100)));
                }
            }
        }

        if issues.is_empty() {
            DataQuality::Valid
        } else {
            DataQuality::Suspect(issues)
        }
    }
}

/// Data quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataQuality {
    Valid,
    Suspect(Vec<String>),
    Invalid(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_normalize_binance_tick() {
        let normalizer = DataNormalizer::new();
        let raw = RawTick {
            exchange: ExchangeId::Binance,
            symbol: "BTCUSDT".to_string(),
            price: dec!(50000.12),
            quantity: dec!(0.5),
            side: Some(Side::Buy),
            trade_id: "12345".to_string(),
            exchange_timestamp: 1700000000000,
            local_timestamp: 1700000000050,
        };

        let tick = normalizer.normalize_tick(&raw).unwrap();
        assert_eq!(tick.symbol.as_str(), "BTC/USDT");
        assert_eq!(tick.exchange, ExchangeId::Binance);
    }

    #[test]
    fn test_normalize_okx_tick() {
        let normalizer = DataNormalizer::new();
        let raw = RawTick {
            exchange: ExchangeId::OKX,
            symbol: "BTC-USDT".to_string(),
            price: dec!(50000),
            quantity: dec!(1),
            side: Some(Side::Sell),
            trade_id: "67890".to_string(),
            exchange_timestamp: 1700000000000,
            local_timestamp: 1700000000030,
        };

        let tick = normalizer.normalize_tick(&raw).unwrap();
        assert_eq!(tick.symbol.as_str(), "BTC/USDT");
    }
}
