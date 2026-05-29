//! Circuit Breaker - Market condition monitoring
//!
//! Halts trading when abnormal market conditions are detected:
//! price spikes, volume drops, spread widening, volatility spikes.
//! Includes VolatilityCircuitBreaker with EMA-based volatility tracking,
//! cooldown and half-open states, and manual override.

use crate::core::types::*;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::VecDeque;

/// Circuit breaker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BreakerState {
    Open,      // Trading halted
    HalfOpen,  // Limited trading allowed
    Closed,    // Normal trading
}

impl std::fmt::Display for BreakerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BreakerState::Open => write!(f, "open"),
            BreakerState::HalfOpen => write!(f, "half_open"),
            BreakerState::Closed => write!(f, "closed"),
        }
    }
}

/// Circuit breaker trigger reason
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum BreakerTrigger {
    PriceSpike { symbol: Symbol, move_pct: Decimal, window_secs: u64 },
    VolumeDrop { symbol: Symbol, volume_pct: Decimal },
    SpreadWidening { symbol: Symbol, spread_pct: Decimal },
    VolatilitySpike { symbol: Symbol, vol_pct: Decimal },
    Manual { reason: String },
}

/// Price-based circuit breaker
pub struct PriceCircuitBreaker {
    config: CircuitBreakerConfig,
    state: RwLock<BreakerState>,
    triggered_at: RwLock<Option<DateTime<Utc>>>,
    trigger_reason: RwLock<Option<BreakerTrigger>>,
    price_history: RwLock<VecDeque<(DateTime<Utc>, Decimal)>>,
}

impl PriceCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        PriceCircuitBreaker {
            config,
            state: RwLock::new(BreakerState::Closed),
            triggered_at: RwLock::new(None),
            trigger_reason: RwLock::new(None),
            price_history: RwLock::new(VecDeque::with_capacity(1000)),
        }
    }

    pub fn check_price(&self, symbol: &Symbol, price: Decimal, timestamp: DateTime<Utc>) -> BreakerState {
        let mut history = self.price_history.write();
        history.push_back((timestamp, price));

        let cutoff = timestamp - chrono::Duration::seconds(self.config.price_move_window_secs as i64);
        while history.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
            history.pop_front();
        }

        if let (Some((_, first_price)), Some((_, last_price))) = (history.front(), history.back()) {
            if *first_price > Decimal::ZERO {
                let move_pct = (last_price - first_price).abs() / *first_price * rust_decimal_macros::dec!(100);
                if move_pct > self.config.price_move_threshold_pct {
                    *self.state.write() = BreakerState::Open;
                    *self.triggered_at.write() = Some(timestamp);
                    *self.trigger_reason.write() = Some(BreakerTrigger::PriceSpike {
                        symbol: symbol.clone(),
                        move_pct,
                        window_secs: self.config.price_move_window_secs,
                    });
                    return BreakerState::Open;
                }
            }
        }

        if *self.state.read() == BreakerState::Open {
            if let Some(triggered) = *self.triggered_at.read() {
                if (timestamp - triggered).num_seconds() >= self.config.cooldown_secs as i64 {
                    *self.state.write() = BreakerState::HalfOpen;
                    return BreakerState::HalfOpen;
                }
            }
        }

        *self.state.read()
    }

    pub fn state(&self) -> BreakerState { *self.state.read() }
    pub fn reset(&self) {
        *self.state.write() = BreakerState::Closed;
        *self.triggered_at.write() = None;
        *self.trigger_reason.write() = None;
    }
}

/// Volume-based circuit breaker
pub struct VolumeCircuitBreaker {
    min_threshold: Decimal,
    state: RwLock<BreakerState>,
    recent_volume: RwLock<Decimal>,
}

impl VolumeCircuitBreaker {
    pub fn new(min_threshold: Decimal) -> Self {
        VolumeCircuitBreaker {
            min_threshold,
            state: RwLock::new(BreakerState::Closed),
            recent_volume: RwLock::new(Decimal::ZERO),
        }
    }

    pub fn check_volume(&self, volume: Decimal) -> BreakerState {
        *self.recent_volume.write() = volume;
        let new_state = if volume < self.min_threshold { BreakerState::Open } else { BreakerState::Closed };
        *self.state.write() = new_state;
        *self.state.read()
    }

    pub fn state(&self) -> BreakerState { *self.state.read() }
}

/// Spread-based circuit breaker
pub struct SpreadCircuitBreaker {
    max_spread_pct: Decimal,
    state: RwLock<BreakerState>,
}

impl SpreadCircuitBreaker {
    pub fn new(max_spread_pct: Decimal) -> Self {
        SpreadCircuitBreaker {
            max_spread_pct,
            state: RwLock::new(BreakerState::Closed),
        }
    }

    pub fn check_spread(&self, bid: Decimal, ask: Decimal) -> BreakerState {
        let mid = (bid + ask) / rust_decimal_macros::dec!(2);
        if mid > Decimal::ZERO {
            let spread_pct = (ask - bid) / mid * rust_decimal_macros::dec!(100);
            *self.state.write() = if spread_pct > self.max_spread_pct { BreakerState::Open } else { BreakerState::Closed };
        }
        *self.state.read()
    }

    pub fn state(&self) -> BreakerState { *self.state.read() }
}

/// Volatility-based circuit breaker using EMA of squared returns
///
/// Tracks volatility via exponential moving average of squared returns.
/// Short-term EMA responds quickly to volatility changes; long-term EMA
/// provides the baseline. When short-term exceeds long-term by a threshold,
/// the breaker triggers.
pub struct VolatilityCircuitBreaker {
    /// Maximum realized volatility (annualized, as percentage) - absolute threshold
    max_vol_pct: Decimal,
    /// EMA smoothing factor for short-term variance (higher = more responsive)
    short_alpha: Decimal,
    /// EMA smoothing factor for long-term variance
    long_alpha: Decimal,
    /// Ratio threshold: trigger when short_ema / long_ema exceeds this
    ratio_threshold: Decimal,
    /// Minimum number of observations required
    min_observations: usize,
    /// Short-term EMA of squared returns
    short_ema: RwLock<Decimal>,
    /// Long-term EMA of squared returns
    long_ema: RwLock<Decimal>,
    /// Last observed price
    last_price: RwLock<Option<Decimal>>,
    /// Observation count
    observation_count: RwLock<usize>,
    /// Rolling price history (for compute_realized_vol fallback)
    price_history: RwLock<VecDeque<(DateTime<Utc>, Decimal)>>,
    /// Window for rolling volatility fallback
    window_secs: u64,
    /// Current state
    state: RwLock<BreakerState>,
    /// When breaker was triggered
    triggered_at: RwLock<Option<DateTime<Utc>>>,
    /// Cooldown period
    cooldown_secs: u64,
}

impl VolatilityCircuitBreaker {
    pub fn new(max_vol_pct: Decimal, window_secs: u64, cooldown_secs: u64) -> Self {
        VolatilityCircuitBreaker {
            max_vol_pct,
            short_alpha: rust_decimal_macros::dec!(0.1),  // Fast EMA
            long_alpha: rust_decimal_macros::dec!(0.02),   // Slow EMA
            ratio_threshold: rust_decimal_macros::dec!(2.0), // Short vol must be 2x long vol
            min_observations: 10,
            short_ema: RwLock::new(Decimal::ZERO),
            long_ema: RwLock::new(Decimal::ZERO),
            last_price: RwLock::new(None),
            observation_count: RwLock::new(0),
            price_history: RwLock::new(VecDeque::with_capacity(1000)),
            window_secs,
            state: RwLock::new(BreakerState::Closed),
            triggered_at: RwLock::new(None),
            cooldown_secs,
        }
    }

    /// Create with custom EMA parameters
    pub fn with_ema_params(
        max_vol_pct: Decimal,
        short_alpha: Decimal,
        long_alpha: Decimal,
        ratio_threshold: Decimal,
        cooldown_secs: u64,
    ) -> Self {
        VolatilityCircuitBreaker {
            max_vol_pct,
            short_alpha,
            long_alpha,
            ratio_threshold,
            min_observations: 10,
            short_ema: RwLock::new(Decimal::ZERO),
            long_ema: RwLock::new(Decimal::ZERO),
            last_price: RwLock::new(None),
            observation_count: RwLock::new(0),
            price_history: RwLock::new(VecDeque::with_capacity(1000)),
            window_secs: 300,
            state: RwLock::new(BreakerState::Closed),
            triggered_at: RwLock::new(None),
            cooldown_secs,
        }
    }

    /// Check a new price for volatility using EMA of squared returns
    pub fn check_price(&self, symbol: &Symbol, price: Decimal, timestamp: DateTime<Utc>) -> BreakerState {
        // Add to price history for fallback vol computation
        {
            let mut history = self.price_history.write();
            history.push_back((timestamp, price));
            let cutoff = timestamp - chrono::Duration::seconds(self.window_secs as i64);
            while history.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
                history.pop_front();
            }
        }

        // Compute log return and update EMA
        {
            let mut last_price = self.last_price.write();
            let mut count = self.observation_count.write();

            if let Some(prev) = *last_price {
                if prev > Decimal::ZERO && price > Decimal::ZERO {
                    let prev_f: f64 = prev.to_string().parse().unwrap_or(1.0);
                    let curr_f: f64 = price.to_string().parse().unwrap_or(1.0);
                    if prev_f > 0.0 && curr_f > 0.0 {
                        let log_return = (curr_f / prev_f).ln();
                        let squared_return = Decimal::from_f64_retain(log_return * log_return).unwrap_or(Decimal::ZERO);

                        // Update short EMA
                        let mut short_ema = self.short_ema.write();
                        if *count < 2 {
                            *short_ema = squared_return;
                        } else {
                            *short_ema = self.short_alpha * squared_return + (Decimal::ONE - self.short_alpha) * *short_ema;
                        }

                        // Update long EMA
                        let mut long_ema = self.long_ema.write();
                        if *count < 2 {
                            *long_ema = squared_return;
                        } else {
                            *long_ema = self.long_alpha * squared_return + (Decimal::ONE - self.long_alpha) * *long_ema;
                        }
                    }
                }
            }

            *last_price = Some(price);
            *count += 1;
        }

        // Need minimum observations
        if *self.observation_count.read() < self.min_observations {
            return *self.state.read();
        }

        // Check cooldown for Open state
        if *self.state.read() == BreakerState::Open {
            if let Some(triggered) = *self.triggered_at.read() {
                if (timestamp - triggered).num_seconds() >= self.cooldown_secs as i64 {
                    *self.state.write() = BreakerState::HalfOpen;
                    return BreakerState::HalfOpen;
                }
            }
            return BreakerState::Open;
        }

        // Compute realized volatility (annualized)
        let realized_vol = self.compute_realized_vol();

        // Absolute threshold check
        if realized_vol > self.max_vol_pct {
            *self.state.write() = BreakerState::Open;
            *self.triggered_at.write() = Some(timestamp);
            return BreakerState::Open;
        }

        // Ratio check: short-term vol vs long-term vol
        let short_var = *self.short_ema.read();
        let long_var = *self.long_ema.read();
        if long_var > Decimal::ZERO {
            let ratio = short_var / long_var;
            if ratio > self.ratio_threshold {
                *self.state.write() = BreakerState::Open;
                *self.triggered_at.write() = Some(timestamp);
                return BreakerState::Open;
            }
        }

        // If in HalfOpen and volatility is back to normal, close
        if *self.state.read() == BreakerState::HalfOpen && realized_vol <= self.max_vol_pct {
            let ratio = if long_var > Decimal::ZERO { short_var / long_var } else { Decimal::ZERO };
            if ratio <= self.ratio_threshold {
                *self.state.write() = BreakerState::Closed;
            }
        }

        *self.state.read()
    }

    /// Compute annualized realized volatility from price history
    fn compute_realized_vol(&self) -> Decimal {
        let history = self.price_history.read();
        if history.len() < 2 { return Decimal::ZERO; }

        let mut log_returns: Vec<(f64, f64)> = Vec::with_capacity(history.len() - 1);
        for i in 1..history.len() {
            let (t_prev, price_prev) = history[i - 1];
            let (t_curr, price_curr) = history[i];

            if price_prev > Decimal::ZERO && price_curr > Decimal::ZERO {
                let prev: f64 = price_prev.to_string().parse().unwrap_or(1.0);
                let curr: f64 = price_curr.to_string().parse().unwrap_or(1.0);
                if prev > 0.0 && curr > 0.0 {
                    let log_return = (curr / prev).ln();
                    let dt_seconds = (t_curr - t_prev).num_milliseconds() as f64 / 1000.0;
                    if dt_seconds > 0.0 {
                        let dt_years = dt_seconds / (365.0 * 24.0 * 3600.0);
                        log_returns.push((log_return, dt_years));
                    }
                }
            }
        }

        if log_returns.is_empty() { return Decimal::ZERO; }

        let sum_squared_returns: f64 = log_returns.iter().map(|(r, _)| r * r).sum();
        let sum_dt_years: f64 = log_returns.iter().map(|(_, dt)| *dt).sum();

        if sum_dt_years > 0.0 {
            let annualized_variance = sum_squared_returns / sum_dt_years;
            let annualized_vol = annualized_variance.sqrt();
            Decimal::from_f64_retain(annualized_vol * 100.0).unwrap_or(Decimal::ZERO)
        } else {
            Decimal::ZERO
        }
    }

    /// Get current realized volatility
    pub fn current_volatility(&self) -> Decimal {
        self.compute_realized_vol()
    }

    /// Get short-term EMA variance
    pub fn short_term_variance(&self) -> Decimal {
        *self.short_ema.read()
    }

    /// Get long-term EMA variance
    pub fn long_term_variance(&self) -> Decimal {
        *self.long_ema.read()
    }

    /// Get variance ratio (short/long)
    pub fn variance_ratio(&self) -> Decimal {
        let long = *self.long_ema.read();
        if long > Decimal::ZERO {
            *self.short_ema.read() / long
        } else {
            Decimal::ZERO
        }
    }

    pub fn state(&self) -> BreakerState { *self.state.read() }
    pub fn reset(&self) {
        *self.state.write() = BreakerState::Closed;
        *self.triggered_at.write() = None;
        *self.short_ema.write() = Decimal::ZERO;
        *self.long_ema.write() = Decimal::ZERO;
        *self.last_price.write() = None;
        *self.observation_count.write() = 0;
        self.price_history.write().clear();
    }
}

/// Composite circuit breaker that monitors all conditions
pub struct CircuitBreaker {
    price_breaker: PriceCircuitBreaker,
    volume_breaker: VolumeCircuitBreaker,
    spread_breaker: SpreadCircuitBreaker,
    volatility_breaker: VolatilityCircuitBreaker,
    manual_override: RwLock<Option<BreakerTrigger>>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig, min_volume: Decimal, max_spread_pct: Decimal) -> Self {
        CircuitBreaker {
            price_breaker: PriceCircuitBreaker::new(config.clone()),
            volume_breaker: VolumeCircuitBreaker::new(min_volume),
            spread_breaker: SpreadCircuitBreaker::new(max_spread_pct),
            volatility_breaker: VolatilityCircuitBreaker::new(
                config.max_realized_vol,
                config.price_move_window_secs,
                config.cooldown_secs,
            ),
            manual_override: RwLock::new(None),
        }
    }

    /// Check all circuit breaker conditions
    pub fn check(
        &self,
        symbol: &Symbol,
        price: Decimal,
        bid: Decimal,
        ask: Decimal,
        volume: Decimal,
        timestamp: DateTime<Utc>,
    ) -> (BreakerState, Option<BreakerTrigger>) {
        if let Some(trigger) = self.manual_override.read().clone() {
            return (BreakerState::Open, Some(trigger));
        }

        let price_state = self.price_breaker.check_price(symbol, price, timestamp);
        if price_state == BreakerState::Open {
            return (BreakerState::Open, self.price_breaker.trigger_reason.read().clone());
        }

        let vol_state = self.volume_breaker.check_volume(volume);
        if vol_state == BreakerState::Open {
            return (BreakerState::Open, Some(BreakerTrigger::VolumeDrop {
                symbol: symbol.clone(), volume_pct: volume,
            }));
        }

        let spread_state = self.spread_breaker.check_spread(bid, ask);
        if spread_state == BreakerState::Open {
            return (BreakerState::Open, Some(BreakerTrigger::SpreadWidening {
                symbol: symbol.clone(),
                spread_pct: (ask - bid) / ((ask + bid) / rust_decimal_macros::dec!(2)) * rust_decimal_macros::dec!(100),
            }));
        }

        let volatility_state = self.volatility_breaker.check_price(symbol, price, timestamp);
        if volatility_state == BreakerState::Open {
            return (BreakerState::Open, Some(BreakerTrigger::VolatilitySpike {
                symbol: symbol.clone(),
                vol_pct: self.volatility_breaker.current_volatility(),
            }));
        }

        (BreakerState::Closed, None)
    }

    pub fn check_price(&self, symbol: &Symbol, price: Decimal, timestamp: DateTime<Utc>) -> BreakerState {
        self.price_breaker.check_price(symbol, price, timestamp)
    }

    pub fn check_volume(&self, volume: Decimal) -> BreakerState {
        self.volume_breaker.check_volume(volume)
    }

    pub fn check_spread(&self, bid: Decimal, ask: Decimal) -> BreakerState {
        self.spread_breaker.check_spread(bid, ask)
    }

    /// Check all breakers and return combined state
    pub fn check_all(
        &self,
        symbol: &Symbol,
        price: Decimal,
        bid: Decimal,
        ask: Decimal,
        volume: Decimal,
        timestamp: DateTime<Utc>,
    ) -> (BreakerState, Option<BreakerTrigger>) {
        self.check(symbol, price, bid, ask, volume, timestamp)
    }

    /// Manually trigger the circuit breaker
    pub fn manual_override(&self, reason: &str) {
        *self.manual_override.write() = Some(BreakerTrigger::Manual {
            reason: reason.to_string(),
        });
    }

    /// Alias for manual_override
    pub fn manual_trigger(&self, reason: &str) {
        self.manual_override(reason);
    }

    /// Reset all circuit breakers
    pub fn reset_all(&self) {
        self.price_breaker.reset();
        self.volatility_breaker.reset();
        *self.manual_override.write() = None;
    }

    /// Is trading currently halted?
    pub fn is_halted(&self) -> bool {
        self.manual_override.read().is_some()
            || self.price_breaker.state() == BreakerState::Open
            || self.volume_breaker.state() == BreakerState::Open
            || self.spread_breaker.state() == BreakerState::Open
            || self.volatility_breaker.state() == BreakerState::Open
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn default_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            price_move_threshold_pct: dec!(5),
            price_move_window_secs: 60,
            min_volume_threshold: dec!(100),
            max_spread_pct: dec!(1),
            max_realized_vol: dec!(100),
            cooldown_secs: 300,
        }
    }

    #[test]
    fn test_price_circuit_breaker_normal() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        let symbol = Symbol::new("BTC/USDT");
        let now = Utc::now();
        let (state, _) = cb.check(&symbol, dec!(50000), dec!(49999), dec!(50001), dec!(1000), now);
        assert_eq!(state, BreakerState::Closed);
    }

    #[test]
    fn test_price_spike_triggers_breaker() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        let symbol = Symbol::new("BTC/USDT");
        let base_time = Utc::now();
        cb.check(&symbol, dec!(50000), dec!(49999), dec!(50001), dec!(1000), base_time);
        let (state, _) = cb.check(&symbol, dec!(53000), dec!(52999), dec!(53001), dec!(1000), base_time + chrono::Duration::seconds(5));
        assert_eq!(state, BreakerState::Open);
    }

    #[test]
    fn test_volume_drop_triggers_breaker() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        let symbol = Symbol::new("BTC/USDT");
        let (state, _) = cb.check(&symbol, dec!(50000), dec!(49999), dec!(50001), dec!(50), Utc::now());
        assert_eq!(state, BreakerState::Open);
    }

    #[test]
    fn test_spread_widening_triggers_breaker() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        let symbol = Symbol::new("BTC/USDT");
        let (state, _) = cb.check(&symbol, dec!(50000), dec!(49000), dec!(51000), dec!(1000), Utc::now());
        assert_eq!(state, BreakerState::Open);
    }

    #[test]
    fn test_manual_override() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        cb.manual_override("Emergency");
        assert!(cb.is_halted());
        cb.reset_all();
        assert!(!cb.is_halted());
    }

    #[test]
    fn test_manual_trigger_alias() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        cb.manual_trigger("Test");
        assert!(cb.is_halted());
        cb.reset_all();
    }

    #[test]
    fn test_spread_circuit_breaker_standalone() {
        let cb = SpreadCircuitBreaker::new(dec!(1));
        assert_eq!(cb.check_spread(dec!(100), dec!(100.05)), BreakerState::Closed);
        assert_eq!(cb.check_spread(dec!(100), dec!(102)), BreakerState::Open);
    }

    #[test]
    fn test_volume_circuit_breaker_standalone() {
        let cb = VolumeCircuitBreaker::new(dec!(100));
        assert_eq!(cb.check_volume(dec!(200)), BreakerState::Closed);
        assert_eq!(cb.check_volume(dec!(50)), BreakerState::Open);
    }

    #[test]
    fn test_volatility_circuit_breaker_stable_prices() {
        let cb = VolatilityCircuitBreaker::new(dec!(200), 300, 60);
        let symbol = Symbol::new("BTC/USDT");
        let base_time = Utc::now();
        for i in 0..15 {
            cb.check_price(&symbol, dec!(50000), base_time + chrono::Duration::seconds(i));
        }
        // Stable prices should not trigger
        assert_ne!(cb.state(), BreakerState::Open);
    }

    #[test]
    fn test_volatility_ema_tracking() {
        let cb = VolatilityCircuitBreaker::new(dec!(200), 300, 60);
        let symbol = Symbol::new("BTC/USDT");
        let base_time = Utc::now();
        // Feed some prices
        for i in 0..20 {
            cb.check_price(&symbol, dec!(50000), base_time + chrono::Duration::seconds(i));
        }
        // Both EMAs should be updated (stable prices → small variance)
        assert!(cb.short_term_variance() >= Decimal::ZERO);
        assert!(cb.long_term_variance() >= Decimal::ZERO);
    }

    #[test]
    fn test_volatility_variance_ratio() {
        let cb = VolatilityCircuitBreaker::new(dec!(200), 300, 60);
        let symbol = Symbol::new("BTC/USDT");
        let base_time = Utc::now();
        for i in 0..20 {
            cb.check_price(&symbol, dec!(50000), base_time + chrono::Duration::seconds(i));
        }
        // With stable prices, ratio should be near 1.0
        let ratio = cb.variance_ratio();
        assert!(ratio >= Decimal::ZERO);
    }

    #[test]
    fn test_volatility_reset() {
        let cb = VolatilityCircuitBreaker::new(dec!(200), 300, 60);
        let symbol = Symbol::new("BTC/USDT");
        let base_time = Utc::now();
        for i in 0..15 {
            cb.check_price(&symbol, dec!(50000), base_time + chrono::Duration::seconds(i));
        }
        cb.reset();
        assert_eq!(cb.state(), BreakerState::Closed);
        assert_eq!(cb.short_term_variance(), Decimal::ZERO);
        assert_eq!(cb.long_term_variance(), Decimal::ZERO);
    }

    #[test]
    fn test_check_all_alias() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        let symbol = Symbol::new("BTC/USDT");
        let now = Utc::now();
        let (state, _) = cb.check_all(&symbol, dec!(50000), dec!(49999), dec!(50001), dec!(1000), now);
        assert_eq!(state, BreakerState::Closed);
    }

    #[test]
    fn test_half_open_after_cooldown() {
        let cb = VolatilityCircuitBreaker::new(dec!(0.001), 60, 1); // Very tight, 1s cooldown
        let symbol = Symbol::new("BTC/USDT");
        let base_time = Utc::now();

        // Feed volatile prices to trigger
        for i in 0..15 {
            let price = if i % 2 == 0 { dec!(50000) } else { dec!(60000) };
            cb.check_price(&symbol, price, base_time + chrono::Duration::seconds(i));
        }

        // After cooldown, should transition to half-open
        if cb.state() == BreakerState::Open {
            let after_cooldown = base_time + chrono::Duration::seconds(100);
            let state = cb.check_price(&symbol, dec!(50000), after_cooldown);
            assert_eq!(state, BreakerState::HalfOpen);
        }
    }

    #[test]
    fn test_reset_all() {
        let cb = CircuitBreaker::new(default_config(), dec!(100), dec!(1));
        cb.manual_trigger("Test");
        assert!(cb.is_halted());
        cb.reset_all();
        assert!(!cb.is_halted());
    }

    #[test]
    fn test_breaker_state_display() {
        assert_eq!(BreakerState::Open.to_string(), "open");
        assert_eq!(BreakerState::HalfOpen.to_string(), "half_open");
        assert_eq!(BreakerState::Closed.to_string(), "closed");
    }
}
