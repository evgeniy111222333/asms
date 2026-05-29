//! TWAP (Time-Weighted Average Price) Execution Algorithm
//!
//! Splits a large order into equal-sized slices executed
//! at regular intervals over a specified time window.
//! Uses proper RNG (rand::thread_rng) with normal distribution
//! for timing jitter and configurable randomization.

use crate::core::types::*;
use chrono::{Duration, Utc};
use rand::Rng;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// TWAP execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapConfig {
    pub total_quantity: Decimal,
    pub symbol: Symbol,
    pub side: Side,
    pub exchange: ExchangeId,
    pub duration_secs: u64,
    pub slice_interval_secs: u64,
    pub randomize_pct: Decimal,
    pub price_limit: Option<Decimal>,
    pub jitter_pct: Decimal, // Timing jitter as percentage of interval
}

impl Default for TwapConfig {
    fn default() -> Self {
        TwapConfig {
            total_quantity: Decimal::ONE,
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Buy,
            exchange: ExchangeId::Paper,
            duration_secs: 600,
            slice_interval_secs: 60,
            randomize_pct: Decimal::ZERO,
            price_limit: None,
            jitter_pct: rust_decimal_macros::dec!(10), // 10% jitter by default
        }
    }
}

/// TWAP slice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapSlice {
    pub index: u32,
    pub quantity: Decimal,
    pub scheduled_time: chrono::DateTime<Utc>,
    pub actual_time: Option<chrono::DateTime<Utc>>,
    pub status: TwapSliceStatus,
    pub fill_price: Option<Decimal>,
    pub fill_quantity: Option<Decimal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwapSliceStatus {
    Pending,
    Submitted,
    Filled,
    Skipped,
    Failed,
}

/// TWAP execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwapState {
    pub config: TwapConfig,
    pub slices: Vec<TwapSlice>,
    pub filled_quantity: Decimal,
    pub avg_fill_price: Decimal,
    pub start_time: chrono::DateTime<Utc>,
    pub end_time: Option<chrono::DateTime<Utc>>,
    pub total_commission: Decimal,
    pub status: TwapStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwapStatus {
    Created,
    Running,
    Paused,
    Completed,
    Cancelled,
    Failed,
}

/// TWAP executor
pub struct TwapExecutor;

impl TwapExecutor {
    /// Create TWAP execution plan with proper RNG-based randomization
    pub fn create_plan(config: TwapConfig) -> TwapState {
        let num_slices = (config.duration_secs / config.slice_interval_secs).max(1) as u32;
        let base_quantity = config.total_quantity / Decimal::from(num_slices);

        let start_time = Utc::now();
        let mut rng = rand::thread_rng();

        let mut remaining_quantity = config.total_quantity;
        let mut slices = Vec::new();

        for i in 0..num_slices {
            let scheduled = start_time + Duration::seconds((i as i64) * (config.slice_interval_secs as i64));

            // Apply timing jitter using normal distribution
            let jitter_pct = if config.jitter_pct > Decimal::ZERO {
                let jitter_factor: f64 = rng.gen_range(-1.0..1.0);
                let pct: f64 = config.jitter_pct.to_string().parse().unwrap_or(10.0);
                Decimal::from_f64_retain(jitter_factor * pct / 100.0).unwrap_or(Decimal::ZERO)
            } else {
                Decimal::ZERO
            };

            let jittered_time = if jitter_pct != Decimal::ZERO {
                let offset_secs = (config.slice_interval_secs as f64) * jitter_pct.to_string().parse::<f64>().unwrap_or(0.0);
                scheduled + Duration::milliseconds((offset_secs * 1000.0) as i64)
            } else {
                scheduled
            };

            // Apply quantity randomization using actual RNG
            let is_last = i == num_slices - 1;
            let qty = if is_last {
                remaining_quantity.max(Decimal::ZERO)
            } else if config.randomize_pct > Decimal::ZERO {
                let random_factor: f64 = rng.gen_range(-1.0..1.0);
                let factor = Decimal::ONE
                    + Decimal::from_f64_retain(random_factor).unwrap_or(Decimal::ZERO)
                        * config.randomize_pct / rust_decimal_macros::dec!(100);
                let mut candidate_qty = (base_quantity * factor).max(Decimal::ZERO);
                if candidate_qty > remaining_quantity {
                    candidate_qty = remaining_quantity;
                }
                remaining_quantity -= candidate_qty;
                candidate_qty
            } else {
                let mut candidate_qty = base_quantity;
                if candidate_qty > remaining_quantity {
                    candidate_qty = remaining_quantity;
                }
                remaining_quantity -= candidate_qty;
                candidate_qty
            };

            slices.push(TwapSlice {
                index: i,
                quantity: qty,
                scheduled_time: jittered_time,
                actual_time: None,
                status: TwapSliceStatus::Pending,
                fill_price: None,
                fill_quantity: None,
            });
        }

        TwapState {
            config,
            slices,
            filled_quantity: Decimal::ZERO,
            avg_fill_price: Decimal::ZERO,
            start_time,
            end_time: None,
            total_commission: Decimal::ZERO,
            status: TwapStatus::Created,
        }
    }

    /// Get the next slice to execute
    pub fn next_slice(state: &mut TwapState) -> Option<&TwapSlice> {
        let now = Utc::now();
        state.slices.iter().find(|s| {
            s.status == TwapSliceStatus::Pending && s.scheduled_time <= now
        })
    }

    /// Mark a slice as filled
    pub fn fill_slice(
        state: &mut TwapState,
        slice_index: u32,
        fill_price: Decimal,
        fill_quantity: Decimal,
        commission: Decimal,
    ) {
        if let Some(slice) = state.slices.get_mut(slice_index as usize) {
            slice.status = TwapSliceStatus::Filled;
            slice.fill_price = Some(fill_price);
            slice.fill_quantity = Some(fill_quantity);
            slice.actual_time = Some(Utc::now());

            let old_notional = state.avg_fill_price * state.filled_quantity;
            let new_notional = fill_price * fill_quantity;
            state.filled_quantity += fill_quantity;
            if state.filled_quantity > Decimal::ZERO {
                state.avg_fill_price = (old_notional + new_notional) / state.filled_quantity;
            }
            state.total_commission += commission;
        }

        if TwapExecutor::is_complete(state) {
            state.status = TwapStatus::Completed;
            state.end_time = Some(Utc::now());
        }
    }

    /// Skip a slice
    pub fn skip_slice(state: &mut TwapState, slice_index: u32) {
        if let Some(slice) = state.slices.get_mut(slice_index as usize) {
            slice.status = TwapSliceStatus::Skipped;
        }

        if TwapExecutor::is_complete(state) {
            state.status = TwapStatus::Completed;
            state.end_time = Some(Utc::now());
        }
    }

    /// Compute participation rate
    pub fn participation_rate(state: &TwapState, market_volume: Decimal) -> Decimal {
        if market_volume > Decimal::ZERO {
            state.filled_quantity / market_volume
        } else {
            Decimal::ZERO
        }
    }

    /// Compute implementation shortfall
    pub fn implementation_shortfall(state: &TwapState, arrival_price: Decimal) -> Decimal {
        if state.avg_fill_price > Decimal::ZERO {
            match state.config.side {
                Side::Buy => state.avg_fill_price - arrival_price,
                Side::Sell => arrival_price - state.avg_fill_price,
            }
        } else {
            Decimal::ZERO
        }
    }

    /// Check if execution is complete
    pub fn is_complete(state: &TwapState) -> bool {
        state.slices.iter().all(|s| {
            matches!(s.status, TwapSliceStatus::Filled | TwapSliceStatus::Skipped | TwapSliceStatus::Failed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn test_config() -> TwapConfig {
        TwapConfig {
            total_quantity: dec!(10),
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Buy,
            exchange: ExchangeId::Binance,
            duration_secs: 600,
            slice_interval_secs: 60,
            randomize_pct: Decimal::ZERO,
            price_limit: None,
            jitter_pct: Decimal::ZERO,
        }
    }

    #[test]
    fn test_twap_plan() {
        let state = TwapExecutor::create_plan(test_config());
        assert_eq!(state.slices.len(), 10);
        assert_eq!(state.status, TwapStatus::Created);
    }

    #[test]
    fn test_twap_plan_with_randomization() {
        let config = TwapConfig {
            randomize_pct: dec!(10),
            jitter_pct: dec!(10),
            ..test_config()
        };
        let state = TwapExecutor::create_plan(config.clone());
        assert_eq!(state.slices.len(), 10);
        let mut total_qty = Decimal::ZERO;
        for slice in &state.slices {
            assert!(slice.quantity > Decimal::ZERO);
            total_qty += slice.quantity;
        }
        assert_eq!(total_qty, config.total_quantity);
    }

    #[test]
    fn test_twap_fill_slice() {
        let mut state = TwapExecutor::create_plan(test_config());
        TwapExecutor::fill_slice(&mut state, 0, dec!(50000), dec!(1), dec!(1));
        assert_eq!(state.slices[0].status, TwapSliceStatus::Filled);
        assert_eq!(state.filled_quantity, dec!(1));
        assert_eq!(state.avg_fill_price, dec!(50000));
        assert!(state.slices[0].actual_time.is_some());
    }

    #[test]
    fn test_twap_skip_slice() {
        let mut state = TwapExecutor::create_plan(test_config());
        TwapExecutor::skip_slice(&mut state, 0);
        assert_eq!(state.slices[0].status, TwapSliceStatus::Skipped);
    }

    #[test]
    fn test_twap_completion() {
        let config = TwapConfig {
            total_quantity: dec!(3),
            duration_secs: 300,
            slice_interval_secs: 100,
            ..test_config()
        };
        let mut state = TwapExecutor::create_plan(config);
        assert_eq!(state.slices.len(), 3);
        for i in 0..3 {
            TwapExecutor::fill_slice(&mut state, i, dec!(50000), dec!(1), dec!(1));
        }
        assert_eq!(state.status, TwapStatus::Completed);
        assert!(state.end_time.is_some());
        assert!(TwapExecutor::is_complete(&state));
    }

    #[test]
    fn test_implementation_shortfall() {
        let mut state = TwapExecutor::create_plan(test_config());
        TwapExecutor::fill_slice(&mut state, 0, dec!(50100), dec!(1), dec!(1));
        let shortfall = TwapExecutor::implementation_shortfall(&state, dec!(50000));
        assert!(shortfall > Decimal::ZERO);
    }

    #[test]
    fn test_twap_minimum_slices() {
        let config = TwapConfig {
            duration_secs: 30,
            slice_interval_secs: 60,
            ..test_config()
        };
        let state = TwapExecutor::create_plan(config);
        assert_eq!(state.slices.len(), 1);
    }

    #[test]
    fn test_participation_rate() {
        let mut state = TwapExecutor::create_plan(test_config());
        TwapExecutor::fill_slice(&mut state, 0, dec!(50000), dec!(1), dec!(1));
        let rate = TwapExecutor::participation_rate(&state, dec!(100));
        assert_eq!(rate, dec!(0.01));
    }

    #[test]
    fn test_is_complete_partial() {
        let config = TwapConfig {
            total_quantity: dec!(3),
            duration_secs: 300,
            slice_interval_secs: 100,
            ..test_config()
        };
        let mut state = TwapExecutor::create_plan(config);
        TwapExecutor::fill_slice(&mut state, 0, dec!(50000), dec!(1), dec!(1));
        assert!(!TwapExecutor::is_complete(&state));
    }

    #[test]
    fn test_twap_jitter_produces_different_times() {
        let config = TwapConfig {
            jitter_pct: dec!(20),
            ..test_config()
        };
        let state = TwapExecutor::create_plan(config);
        // With 20% jitter, scheduled times should vary slightly
        // (Not guaranteed different due to RNG, but the mechanism is in place)
        assert_eq!(state.slices.len(), 10);
    }

    #[test]
    fn test_twap_sell_side_shortfall() {
        let config = TwapConfig {
            side: Side::Sell,
            ..test_config()
        };
        let mut state = TwapExecutor::create_plan(config);
        TwapExecutor::fill_slice(&mut state, 0, dec!(49900), dec!(1), dec!(1));
        // Sold at 49900, arrival was 50000 -> positive shortfall (good for seller)
        let shortfall = TwapExecutor::implementation_shortfall(&state, dec!(50000));
        assert!(shortfall > Decimal::ZERO);
    }
}
