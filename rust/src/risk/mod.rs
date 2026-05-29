//! Risk Management Module
//!
//! Sub-millisecond pre-trade risk checks and circuit breakers.

pub mod hot_path;
pub mod circuit_breaker;
pub mod exposure;

pub use hot_path::RiskHotPath;
pub use circuit_breaker::CircuitBreaker;
pub use circuit_breaker::VolatilityCircuitBreaker;
pub use circuit_breaker::BreakerState;
pub use exposure::ExposureTracker;
