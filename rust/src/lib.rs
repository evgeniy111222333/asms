//! ACMS Core - Rust High-Performance Engine
//!
//! This crate provides the performance-critical components of the
//! Algorithmic Crypto Management System:
//! - Core types (Order, Trade, Position, Candle, OrderBook, etc.)
//! - Data Ingestion (WebSocket collectors, normalization, Arrow pipeline)
//! - Execution Engine (order management, SOR, TWAP/VWAP algorithms)
//! - Risk Management hot-path (sub-millisecond pre-trade checks)
//! - PyO3 bridge for zero-copy Python integration

pub mod core;
pub mod execution;
pub mod ingestion;
pub mod risk;

#[cfg(feature = "pyo3_bridge")]
pub mod pyo3_bridge;

pub use core::types::*;
pub use core::orderbook::OrderBook;
pub use core::market_data::CandleAggregator;
pub use core::market_data::RollingVWAP;
pub use core::market_data::VolumeProfile;
pub use execution::engine::ExecutionEngine;
pub use execution::sor::SmartOrderRouter;
pub use execution::twap::TwapExecutor;
pub use execution::vwap_exec::VwapExecutor;
pub use risk::hot_path::RiskHotPath;
pub use risk::circuit_breaker::CircuitBreaker;
pub use risk::circuit_breaker::VolatilityCircuitBreaker;
pub use risk::circuit_breaker::BreakerState;
pub use risk::exposure::ExposureTracker;
pub use ingestion::collector::DataCollector;
pub use ingestion::pipeline::DataPipeline;

#[cfg(test)]
mod comprehensive_tests {
    use super::*;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    #[test]
    fn test_full_order_lifecycle_with_risk() {
        let risk = RiskHotPath::new(RiskLimits::default(), dec!(1000000));
        let engine = ExecutionEngine::new(std::collections::HashMap::new());

        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper,
        );

        let decision = risk.is_allowed(&order, dec!(1000000));
        assert_eq!(decision, RiskDecision::Allow);

        engine.submit_order(&mut order).unwrap();

        let report = engine.process_fill(&order.id, dec!(50000), dec!(0.1), dec!(5), false).unwrap();
        assert_eq!(report.status, OrderStatus::Filled);
        assert!(report.slippage >= Decimal::ZERO);
    }

    #[test]
    fn test_circuit_breaker_and_risk_integration() {
        let config = CircuitBreakerConfig {
            price_move_threshold_pct: dec!(5),
            price_move_window_secs: 60,
            min_volume_threshold: dec!(100),
            max_spread_pct: dec!(1),
            max_realized_vol: dec!(200),
            cooldown_secs: 300,
        };
        let cb = CircuitBreaker::new(config, dec!(100), dec!(1));

        let symbol = Symbol::new("BTC/USDT");
        let now = chrono::Utc::now();
        let (state, _) = cb.check(&symbol, dec!(50000), dec!(49999), dec!(50001), dec!(1000), now);
        assert_eq!(state, BreakerState::Closed);
    }

    #[test]
    fn test_rolling_vwap_with_explicit_timestamps() {
        let mut vwap = RollingVWAP::new(Symbol::new("BTC/USDT"), 300);
        vwap.update_at(dec!(100), dec!(10), chrono::DateTime::from_timestamp(0, 0).unwrap());
        vwap.update_at(dec!(200), dec!(5), chrono::DateTime::from_timestamp(100, 0).unwrap());
        let val = vwap.value().unwrap();
        assert!(val > dec!(133) && val < dec!(134));
    }
}
