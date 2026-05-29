//! VWAP (Volume-Weighted Average Price) Execution Algorithm
//!
//! Executes orders following the volume distribution profile
//! of the trading day to minimize market impact.

use crate::core::types::*;
use chrono::Utc;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Volume profile for a time bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeBucket {
    pub start_minute: u32,
    pub end_minute: u32,
    pub volume_pct: Decimal,
}

/// VWAP execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapConfig {
    pub total_quantity: Decimal,
    pub symbol: Symbol,
    pub side: Side,
    pub exchange: ExchangeId,
    pub duration_minutes: u32,
    pub max_participation_rate: Decimal,
    pub volume_profile: Vec<VolumeBucket>,
    pub price_limit: Option<Decimal>,
}

impl VwapConfig {
    pub fn default_crypto_profile() -> Vec<VolumeBucket> {
        vec![
            VolumeBucket { start_minute: 0, end_minute: 30, volume_pct: Decimal::from(8) },
            VolumeBucket { start_minute: 30, end_minute: 60, volume_pct: Decimal::from(6) },
            VolumeBucket { start_minute: 60, end_minute: 120, volume_pct: Decimal::from(4) },
            VolumeBucket { start_minute: 120, end_minute: 240, volume_pct: Decimal::from(3) },
            VolumeBucket { start_minute: 240, end_minute: 360, volume_pct: Decimal::from(2) },
            VolumeBucket { start_minute: 360, end_minute: 480, volume_pct: Decimal::from(2) },
            VolumeBucket { start_minute: 480, end_minute: 600, volume_pct: Decimal::from(3) },
            VolumeBucket { start_minute: 600, end_minute: 660, volume_pct: Decimal::from(4) },
            VolumeBucket { start_minute: 660, end_minute: 720, volume_pct: Decimal::from(5) },
            VolumeBucket { start_minute: 720, end_minute: 780, volume_pct: Decimal::from(7) },
            VolumeBucket { start_minute: 780, end_minute: 840, volume_pct: Decimal::from(9) },
            VolumeBucket { start_minute: 840, end_minute: 900, volume_pct: Decimal::from(10) },
            VolumeBucket { start_minute: 900, end_minute: 960, volume_pct: Decimal::from(12) },
            VolumeBucket { start_minute: 960, end_minute: 1020, volume_pct: Decimal::from(10) },
            VolumeBucket { start_minute: 1020, end_minute: 1080, volume_pct: Decimal::from(8) },
            VolumeBucket { start_minute: 1080, end_minute: 1140, volume_pct: Decimal::from(7) },
        ]
    }
}

/// VWAP execution slice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapSlice {
    pub index: u32,
    pub scheduled_time: chrono::DateTime<Utc>,
    pub target_quantity: Decimal,
    pub actual_quantity: Decimal,
    pub fill_price: Option<Decimal>,
    pub market_volume: Decimal,
    pub participation_rate: Decimal,
    pub status: VwapSliceStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VwapSliceStatus {
    Pending,
    Submitted,
    Filled,
    Skipped,
    Failed,
}

/// VWAP execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VwapState {
    pub config: VwapConfig,
    pub slices: Vec<VwapSlice>,
    pub filled_quantity: Decimal,
    pub avg_fill_price: Decimal,
    pub vwap_benchmark: Decimal,
    pub start_time: chrono::DateTime<Utc>,
    pub status: VwapStatus,
    pub total_commission: Decimal,
    pub total_slippage: Decimal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VwapStatus {
    Created,
    Running,
    Paused,
    Completed,
    Cancelled,
}

/// VWAP executor
pub struct VwapExecutor;

impl VwapExecutor {
    /// Create VWAP execution plan from volume profile
    pub fn create_plan(config: VwapConfig) -> VwapState {
        let profile = if config.volume_profile.is_empty() {
            VwapConfig::default_crypto_profile()
        } else {
            config.volume_profile.clone()
        };

        let start_time = Utc::now();
        let bucket_duration = config.duration_minutes / profile.len() as u32;

        let slices: Vec<VwapSlice> = profile
            .iter()
            .enumerate()
            .map(|(i, bucket)| {
                let scheduled = start_time + chrono::Duration::minutes((i as i64) * bucket_duration as i64);
                let target_qty = config.total_quantity * bucket.volume_pct / rust_decimal_macros::dec!(100);
                VwapSlice {
                    index: i as u32,
                    scheduled_time: scheduled,
                    target_quantity: target_qty,
                    actual_quantity: Decimal::ZERO,
                    fill_price: None,
                    market_volume: Decimal::ZERO,
                    participation_rate: Decimal::ZERO,
                    status: VwapSliceStatus::Pending,
                }
            })
            .collect();

        VwapState {
            config,
            slices,
            filled_quantity: Decimal::ZERO,
            avg_fill_price: Decimal::ZERO,
            vwap_benchmark: Decimal::ZERO,
            start_time,
            status: VwapStatus::Created,
            total_commission: Decimal::ZERO,
            total_slippage: Decimal::ZERO,
        }
    }

    /// Compute adaptive quantity for a slice
    pub fn compute_adaptive_quantity(
        state: &VwapState,
        slice_index: u32,
        observed_market_volume: Decimal,
    ) -> Decimal {
        if let Some(slice) = state.slices.get(slice_index as usize) {
            let max_qty = observed_market_volume * state.config.max_participation_rate;
            slice.target_quantity.min(max_qty)
        } else {
            Decimal::ZERO
        }
    }

    /// Fill a VWAP slice
    pub fn fill_slice(
        state: &mut VwapState,
        slice_index: u32,
        fill_price: Decimal,
        fill_quantity: Decimal,
        commission: Decimal,
        market_vwap: Decimal,
    ) {
        if let Some(slice) = state.slices.get_mut(slice_index as usize) {
            slice.status = VwapSliceStatus::Filled;
            slice.fill_price = Some(fill_price);
            slice.actual_quantity = fill_quantity;
            slice.participation_rate = if slice.market_volume > Decimal::ZERO {
                fill_quantity / slice.market_volume
            } else {
                Decimal::ZERO
            };

            let old_notional = state.avg_fill_price * state.filled_quantity;
            let new_notional = fill_price * fill_quantity;
            state.filled_quantity += fill_quantity;
            if state.filled_quantity > Decimal::ZERO {
                state.avg_fill_price = (old_notional + new_notional) / state.filled_quantity;
            }
            state.total_commission += commission;
            state.vwap_benchmark = market_vwap;
            state.total_slippage += (fill_price - market_vwap).abs() * fill_quantity;
        }

        if VwapExecutor::is_complete(state) {
            state.status = VwapStatus::Completed;
        }
    }

    /// Compute VWAP performance vs benchmark
    pub fn performance_vs_benchmark(state: &VwapState) -> Decimal {
        if state.vwap_benchmark > Decimal::ZERO {
            match state.config.side {
                Side::Buy => state.vwap_benchmark - state.avg_fill_price,
                Side::Sell => state.avg_fill_price - state.vwap_benchmark,
            }
        } else {
            Decimal::ZERO
        }
    }

    /// Check if execution is complete
    pub fn is_complete(state: &VwapState) -> bool {
        state.slices.iter().all(|s| {
            matches!(s.status, VwapSliceStatus::Filled | VwapSliceStatus::Skipped | VwapSliceStatus::Failed)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn test_config() -> VwapConfig {
        VwapConfig {
            total_quantity: dec!(100),
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Buy,
            exchange: ExchangeId::Binance,
            duration_minutes: 960,
            max_participation_rate: dec!(0.1),
            volume_profile: VwapConfig::default_crypto_profile(),
            price_limit: None,
        }
    }

    #[test]
    fn test_vwap_plan() {
        let state = VwapExecutor::create_plan(test_config());
        assert!(!state.slices.is_empty());
        let total_target: Decimal = state.slices.iter().map(|s| s.target_quantity).sum();
        assert!(total_target > Decimal::ZERO);
    }

    #[test]
    fn test_vwap_adaptive_quantity() {
        let state = VwapExecutor::create_plan(test_config());
        let adaptive = VwapExecutor::compute_adaptive_quantity(&state, 0, dec!(1000));
        assert!(adaptive <= dec!(100)); // 1000 * 0.1 = 100
    }

    #[test]
    fn test_vwap_performance_vs_benchmark() {
        let mut state = VwapExecutor::create_plan(test_config());
        VwapExecutor::fill_slice(&mut state, 0, dec!(49900), dec!(10), dec!(1), dec!(50000));
        let perf = VwapExecutor::performance_vs_benchmark(&state);
        assert!(perf > Decimal::ZERO);
    }

    #[test]
    fn test_vwap_full_plan_execution() {
        let mut state = VwapExecutor::create_plan(VwapConfig {
            total_quantity: dec!(3),
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Buy,
            exchange: ExchangeId::Paper,
            duration_minutes: 960,
            max_participation_rate: dec!(0.1),
            volume_profile: vec![
                VolumeBucket { start_minute: 0, end_minute: 320, volume_pct: Decimal::from(34) },
                VolumeBucket { start_minute: 320, end_minute: 640, volume_pct: Decimal::from(33) },
                VolumeBucket { start_minute: 640, end_minute: 960, volume_pct: Decimal::from(33) },
            ],
            price_limit: None,
        });

        for i in 0..3 {
            VwapExecutor::fill_slice(&mut state, i, dec!(50000), dec!(1), dec!(1), dec!(50000));
        }

        assert_eq!(state.status, VwapStatus::Completed);
        assert!(VwapExecutor::is_complete(&state));
    }

    #[test]
    fn test_vwap_is_complete_partial() {
        let mut state = VwapExecutor::create_plan(VwapConfig {
            total_quantity: dec!(3),
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Buy,
            exchange: ExchangeId::Paper,
            duration_minutes: 960,
            max_participation_rate: dec!(0.1),
            volume_profile: vec![
                VolumeBucket { start_minute: 0, end_minute: 320, volume_pct: Decimal::from(34) },
                VolumeBucket { start_minute: 320, end_minute: 640, volume_pct: Decimal::from(33) },
                VolumeBucket { start_minute: 640, end_minute: 960, volume_pct: Decimal::from(33) },
            ],
            price_limit: None,
        });

        VwapExecutor::fill_slice(&mut state, 0, dec!(50000), dec!(1), dec!(1), dec!(50000));
        assert!(!VwapExecutor::is_complete(&state));
        assert_ne!(state.status, VwapStatus::Completed);
    }
}
