//! Market data aggregation and processing
//!
//! Candle aggregation from ticks, VWAP computation, volume profiles.
//! RollingVWAP properly expires old data based on window_secs.

use crate::core::types::*;
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// A single time-stamped entry for rolling VWAP computation
#[derive(Debug, Clone)]
struct VwapEntry {
    timestamp: DateTime<Utc>,
    notional: Decimal,
    quantity: Decimal,
}

/// Rolling VWAP tracker that expires old data based on window_secs
#[derive(Debug, Clone)]
pub struct RollingVWAP {
    pub symbol: Symbol,
    entries: VecDeque<VwapEntry>,
    window_secs: u64,
    cached_notional: Decimal,
    cached_volume: Decimal,
}

impl RollingVWAP {
    pub fn new(symbol: Symbol, window_secs: u64) -> Self {
        RollingVWAP {
            symbol,
            entries: VecDeque::new(),
            window_secs,
            cached_notional: Decimal::ZERO,
            cached_volume: Decimal::ZERO,
        }
    }

    /// Update with a new price/quantity observation at the given time
    pub fn update(&mut self, price: Decimal, quantity: Decimal) {
        self.update_at(price, quantity, Utc::now());
    }

    /// Update with an explicit timestamp (useful for replaying historical data)
    pub fn update_at(&mut self, price: Decimal, quantity: Decimal, timestamp: DateTime<Utc>) {
        let notional = price * quantity;
        self.cached_notional += notional;
        self.cached_volume += quantity;
        self.entries.push_back(VwapEntry {
            timestamp,
            notional,
            quantity,
        });
        self.expire_old(timestamp);
    }

    /// Remove entries outside the rolling window
    fn expire_old(&mut self, now: DateTime<Utc>) {
        let cutoff = now - Duration::seconds(self.window_secs as i64);
        while let Some(front) = self.entries.front() {
            if front.timestamp < cutoff {
                let expired = self.entries.pop_front().unwrap();
                self.cached_notional -= expired.notional;
                self.cached_volume -= expired.quantity;
            } else {
                break;
            }
        }
    }

    /// Get the current rolling VWAP value
    pub fn value(&self) -> Option<Decimal> {
        if self.cached_volume > Decimal::ZERO {
            Some(self.cached_notional / self.cached_volume)
        } else {
            None
        }
    }

    /// Get total volume in the current window
    pub fn total_volume(&self) -> Decimal {
        self.cached_volume
    }

    /// Get total notional in the current window
    pub fn total_notional(&self) -> Decimal {
        self.cached_notional
    }

    /// Get number of entries in the current window
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Reset all data
    pub fn reset(&mut self) {
        self.entries.clear();
        self.cached_notional = Decimal::ZERO;
        self.cached_volume = Decimal::ZERO;
    }

    /// Get the configured window in seconds
    pub fn window_secs(&self) -> u64 {
        self.window_secs
    }
}

/// Volume profile bin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeBin {
    pub price: Decimal,
    pub volume: Decimal,
    pub buy_volume: Decimal,
    pub sell_volume: Decimal,
    pub trade_count: u64,
}

/// Volume profile builder
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    pub symbol: Symbol,
    bins: std::collections::BTreeMap<Decimal, VolumeBin>,
    bin_size: Decimal,
    total_volume: Decimal,
    poc_price: Option<Decimal>, // Point of Control
}

impl VolumeProfile {
    pub fn new(symbol: Symbol, bin_size: Decimal) -> Self {
        VolumeProfile {
            symbol,
            bins: std::collections::BTreeMap::new(),
            bin_size,
            total_volume: Decimal::ZERO,
            poc_price: None,
        }
    }

    pub fn add_trade(&mut self, price: Decimal, quantity: Decimal, side: Side) {
        let bin_price = (price / self.bin_size).floor() * self.bin_size;
        let bin = self.bins.entry(bin_price).or_insert_with(|| VolumeBin {
            price: bin_price,
            volume: Decimal::ZERO,
            buy_volume: Decimal::ZERO,
            sell_volume: Decimal::ZERO,
            trade_count: 0,
        });
        bin.volume += quantity;
        bin.trade_count += 1;
        match side {
            Side::Buy => bin.buy_volume += quantity,
            Side::Sell => bin.sell_volume += quantity,
        }
        self.total_volume += quantity;

        // Update POC
        let max_vol = self.bins.values().map(|b| b.volume).max();
        if let Some(max) = max_vol {
            self.poc_price = self
                .bins
                .iter()
                .find(|(_, b)| b.volume == max)
                .map(|(p, _)| *p);
        }
    }

    pub fn get_bins(&self) -> Vec<&VolumeBin> {
        self.bins.values().collect()
    }

    pub fn poc(&self) -> Option<Decimal> {
        self.poc_price
    }

    /// Value area (70% of volume around POC)
    pub fn value_area(&self) -> (Option<Decimal>, Option<Decimal>) {
        if self.total_volume == Decimal::ZERO {
            return (None, None);
        }
        let target = self.total_volume * rust_decimal_macros::dec!(0.70);
        let poc = match self.poc_price {
            Some(p) => p,
            None => return (None, None),
        };

        let mut accumulated = Decimal::ZERO;
        let mut low = poc;
        let mut high = poc;
        let prices: Vec<Decimal> = self.bins.keys().cloned().collect();
        let poc_idx = prices.iter().position(|p| *p == poc).unwrap_or(0);

        let mut lo_idx = poc_idx;
        let mut hi_idx = poc_idx;

        while accumulated < target {
            let lo_vol = if lo_idx > 0 {
                self.bins.get(&prices[lo_idx - 1]).map(|b| b.volume).unwrap_or(Decimal::ZERO)
            } else {
                Decimal::ZERO
            };
            let hi_vol = if hi_idx + 1 < prices.len() {
                self.bins.get(&prices[hi_idx + 1]).map(|b| b.volume).unwrap_or(Decimal::ZERO)
            } else {
                Decimal::ZERO
            };

            if lo_vol >= hi_vol && lo_idx > 0 {
                lo_idx -= 1;
                accumulated += lo_vol;
                low = prices[lo_idx];
            } else if hi_idx + 1 < prices.len() {
                hi_idx += 1;
                accumulated += hi_vol;
                high = prices[hi_idx];
            } else {
                break;
            }
        }

        (Some(low), Some(high))
    }
}

/// Candle aggregator from ticks
pub struct CandleAggregator {
    symbol: Symbol,
    exchange: ExchangeId,
    timeframe: Timeframe,
    current_candle: Option<CandleBuilder>,
    completed_candles: Vec<Candle>,
    max_completed: usize,
}

#[derive(Debug)]
struct CandleBuilder {
    open_time: DateTime<Utc>,
    close_time: DateTime<Utc>,
    open: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: Decimal,
    quote_volume: Decimal,
    trades: u64,
    taker_buy_volume: Decimal,
    taker_buy_quote_volume: Decimal,
}

impl CandleAggregator {
    pub fn new(symbol: Symbol, exchange: ExchangeId, timeframe: Timeframe) -> Self {
        CandleAggregator {
            symbol,
            exchange,
            timeframe,
            current_candle: None,
            completed_candles: Vec::new(),
            max_completed: 1000,
        }
    }

    /// Process a tick and return any completed candle
    pub fn process_tick(&mut self, tick: &Tick) -> Option<Candle> {
        let tf_secs = self.timeframe.duration_secs() as i64;
        let candle_start = tick.timestamp.timestamp() / tf_secs * tf_secs;
        let open_time = DateTime::from_timestamp(candle_start, 0).unwrap_or(tick.timestamp);
        let close_time = open_time + Duration::seconds(tf_secs);

        let completed = match &mut self.current_candle {
            Some(c) if c.open_time.timestamp() != candle_start => {
                let completed = Candle {
                    symbol: self.symbol.clone(),
                    timeframe: self.timeframe,
                    open_time: c.open_time,
                    close_time: c.close_time,
                    open: c.open,
                    high: c.high,
                    low: c.low,
                    close: c.close,
                    volume: c.volume,
                    quote_volume: c.quote_volume,
                    trades: c.trades,
                    taker_buy_volume: c.taker_buy_volume,
                    taker_buy_quote_volume: c.taker_buy_quote_volume,
                };
                self.completed_candles.push(completed.clone());
                if self.completed_candles.len() > self.max_completed {
                    self.completed_candles.remove(0);
                }
                Some(completed)
            }
            _ => None,
        };

        let is_taker_buy = tick.side == Side::Buy;

        match &mut self.current_candle {
            Some(c) if c.open_time.timestamp() == candle_start => {
                if tick.price > c.high {
                    c.high = tick.price;
                }
                if tick.price < c.low {
                    c.low = tick.price;
                }
                c.close = tick.price;
                c.volume += tick.quantity;
                c.quote_volume += tick.price * tick.quantity;
                c.trades += 1;
                if is_taker_buy {
                    c.taker_buy_volume += tick.quantity;
                    c.taker_buy_quote_volume += tick.price * tick.quantity;
                }
            }
            _ => {
                self.current_candle = Some(CandleBuilder {
                    open_time,
                    close_time,
                    open: tick.price,
                    high: tick.price,
                    low: tick.price,
                    close: tick.price,
                    volume: tick.quantity,
                    quote_volume: tick.price * tick.quantity,
                    trades: 1,
                    taker_buy_volume: if is_taker_buy { tick.quantity } else { Decimal::ZERO },
                    taker_buy_quote_volume: if is_taker_buy {
                        tick.price * tick.quantity
                    } else {
                        Decimal::ZERO
                    },
                });
            }
        }

        completed
    }

    /// Get the current incomplete candle
    pub fn current_candle(&self) -> Option<Candle> {
        self.current_candle.as_ref().map(|c| Candle {
            symbol: self.symbol.clone(),
            timeframe: self.timeframe,
            open_time: c.open_time,
            close_time: c.close_time,
            open: c.open,
            high: c.high,
            low: c.low,
            close: c.close,
            volume: c.volume,
            quote_volume: c.quote_volume,
            trades: c.trades,
            taker_buy_volume: c.taker_buy_volume,
            taker_buy_quote_volume: c.taker_buy_quote_volume,
        })
    }

    /// Get completed candles
    pub fn completed_candles(&self) -> &[Candle] {
        &self.completed_candles
    }

    /// Force-close the current candle
    pub fn force_close(&mut self) -> Option<Candle> {
        self.current_candle.take().map(|c| {
            let candle = Candle {
                symbol: self.symbol.clone(),
                timeframe: self.timeframe,
                open_time: c.open_time,
                close_time: c.close_time,
                open: c.open,
                high: c.high,
                low: c.low,
                close: c.close,
                volume: c.volume,
                quote_volume: c.quote_volume,
                trades: c.trades,
                taker_buy_volume: c.taker_buy_volume,
                taker_buy_quote_volume: c.taker_buy_quote_volume,
            };
            self.completed_candles.push(candle.clone());
            candle
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    fn make_tick(price: Decimal, qty: Decimal, side: Side, secs_from_epoch: i64) -> Tick {
        Tick {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Binance,
            price,
            quantity: qty,
            side,
            timestamp: DateTime::from_timestamp(secs_from_epoch, 0).unwrap(),
            trade_id: "1".to_string(),
        }
    }

    #[test]
    fn test_candle_aggregation() {
        let mut agg = CandleAggregator::new(
            Symbol::new("BTC/USDT"),
            ExchangeId::Binance,
            Timeframe::M1,
        );

        agg.process_tick(&make_tick(dec!(100), dec!(1), Side::Buy, 61));
        agg.process_tick(&make_tick(dec!(105), dec!(1), Side::Buy, 62));
        agg.process_tick(&make_tick(dec!(98), dec!(1), Side::Sell, 63));

        let candle = agg.current_candle().unwrap();
        assert_eq!(candle.open, dec!(100));
        assert_eq!(candle.high, dec!(105));
        assert_eq!(candle.low, dec!(98));
        assert_eq!(candle.close, dec!(98));
    }

    #[test]
    fn test_volume_profile() {
        let mut vp = VolumeProfile::new(Symbol::new("BTC/USDT"), dec!(10));
        vp.add_trade(dec!(100), dec!(5), Side::Buy);
        vp.add_trade(dec!(102), dec!(3), Side::Sell);
        vp.add_trade(dec!(100), dec!(7), Side::Buy);
        assert_eq!(vp.poc(), Some(dec!(100)));
    }

    #[test]
    fn test_rolling_vwap_basic() {
        let mut vwap = RollingVWAP::new(Symbol::new("BTC/USDT"), 300);
        vwap.update_at(dec!(100), dec!(10), DateTime::from_timestamp(1000, 0).unwrap());
        vwap.update_at(dec!(102), dec!(5), DateTime::from_timestamp(1010, 0).unwrap());

        let val = vwap.value().unwrap();
        // (100*10 + 102*5) / 15 = 1510/15 = 100.666...
        assert!(val > dec!(100) && val < dec!(101));
    }

    #[test]
    fn test_rolling_vwap_expiry() {
        let mut vwap = RollingVWAP::new(Symbol::new("BTC/USDT"), 10); // 10 second window

        vwap.update_at(dec!(100), dec!(10), DateTime::from_timestamp(0, 0).unwrap());
        assert_eq!(vwap.entry_count(), 1);
        assert!(vwap.value().is_some());

        // Add an entry 20 seconds later - the first should expire
        vwap.update_at(dec!(200), dec!(5), DateTime::from_timestamp(20, 0).unwrap());
        assert_eq!(vwap.entry_count(), 1);
        let val = vwap.value().unwrap();
        assert_eq!(val, dec!(200)); // Only the new entry remains
    }

    #[test]
    fn test_rolling_vwap_partial_expiry() {
        let mut vwap = RollingVWAP::new(Symbol::new("BTC/USDT"), 10);

        vwap.update_at(dec!(100), dec!(10), DateTime::from_timestamp(0, 0).unwrap());
        vwap.update_at(dec!(100), dec!(10), DateTime::from_timestamp(5, 0).unwrap());
        vwap.update_at(dec!(200), dec!(10), DateTime::from_timestamp(15, 0).unwrap());

        // At t=15, the entry at t=0 should expire, t=5 should remain
        assert_eq!(vwap.entry_count(), 2);
        let val = vwap.value().unwrap();
        // (100*10 + 200*10) / 20 = 3000/20 = 150
        assert_eq!(val, dec!(150));
    }

    #[test]
    fn test_rolling_vwap_reset() {
        let mut vwap = RollingVWAP::new(Symbol::new("BTC/USDT"), 300);
        vwap.update(dec!(100), dec!(10));
        assert!(vwap.value().is_some());
        vwap.reset();
        assert!(vwap.value().is_none());
        assert_eq!(vwap.total_volume(), Decimal::ZERO);
    }
}
