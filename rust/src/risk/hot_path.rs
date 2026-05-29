//! Sub-millisecond pre-trade risk checks
//!
//! All checks are designed to complete in <100μs.
//! Uses token bucket rate limiter with atomic operations
//! to avoid lock contention on the hot path.

use crate::core::types::*;
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Kill switch state (lock-free atomic)
pub struct KillSwitch {
    triggered: AtomicBool,
    triggered_at: RwLock<Option<chrono::DateTime<Utc>>>,
    reason: RwLock<Option<String>>,
}

impl KillSwitch {
    pub fn new() -> Self {
        KillSwitch {
            triggered: AtomicBool::new(false),
            triggered_at: RwLock::new(None),
            reason: RwLock::new(None),
        }
    }

    pub fn trigger(&self, reason: &str) {
        self.triggered.store(true, Ordering::SeqCst);
        *self.triggered_at.write() = Some(Utc::now());
        *self.reason.write() = Some(reason.to_string());
    }

    pub fn reset(&self) {
        self.triggered.store(false, Ordering::SeqCst);
        *self.triggered_at.write() = None;
        *self.reason.write() = None;
    }

    pub fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::SeqCst)
    }
}

/// Token bucket rate limiter using only atomic operations.
///
/// Uses a simple token bucket algorithm:
/// - Tokens refill at `refill_rate` per second
/// - Maximum tokens = bucket capacity
/// - Each request consumes one token
/// - If no tokens available, request is throttled
///
/// All state is stored in atomics to avoid lock contention on the hot path.
/// Token counts are stored as fixed-point integers (actual_tokens * 1_000_000)
/// to avoid floating-point while preserving sub-token precision.
pub struct RateLimiter {
    /// Second-level bucket: tokens * 1_000_000 (fixed-point)
    sec_tokens: AtomicU64,
    /// Max tokens for second bucket (fixed-point)
    sec_max: u64,
    /// Refill rate per second (fixed-point)
    sec_refill_rate: u64,
    /// Last refill timestamp (nanoseconds since epoch)
    sec_last_refill_ns: AtomicU64,
    /// Minute-level bucket: tokens * 1_000_000 (fixed-point)
    min_tokens: AtomicU64,
    /// Max tokens for minute bucket (fixed-point)
    min_max: u64,
    /// Last refill timestamp for minute bucket (nanoseconds)
    min_last_refill_ns: AtomicU64,
}

/// Scale factor for fixed-point arithmetic (6 decimal places)
const FP_SCALE: u64 = 1_000_000;

impl RateLimiter {
    pub fn new() -> Self {
        Self::with_limits(10, 100)
    }

    pub fn with_limits(max_per_second: u32, max_per_minute: u32) -> Self {
        let now_ns = Self::current_ns();

        let sec_max = (max_per_second as u64) * FP_SCALE;
        let sec_refill_rate = sec_max; // Full bucket per second

        let min_max = (max_per_minute as u64) * FP_SCALE;
        // Refill rate per nanosecond: min_max / 60_000_000_000
        // To avoid overflow and precision loss, precompute:
        // min_refill_per_ns = min_max * 1_000 / 60_000_000_000 * 1_000
        // Actually let's use: refill per ns = min_max / 60e9
        // We store it as a scaled integer: (min_max / 60_000_000_000) * FP_SCALE
        // But min_max is already in FP_SCALE, so: min_max / 60_000_000_000
        // For 100 orders/min: 100_000_000 / 60_000_000_000 ≈ 0.00167 per ns
        // We need sub-nanosecond precision. Use: min_refill_per_ns as the amount
        // to add per nanosecond, in fixed-point. Compute carefully:
        // per_minute / 60s = per_second. per_second / 1e9 = per_ns.
        // min_refill_per_ns_fp = min_max * FP_SCALE / 60_000_000_000
        // But min_max already has FP_SCALE, so: min_max / 60_000_000_000 * FP_SCALE
        // For 100/min: 100_000_000 * 1_000_000 / 60_000_000_000 = 1666.67
        // That's too small to be precise with integer math per nanosecond.
        // Better approach: compute elapsed nanoseconds and multiply by rate.

        // Instead of per-ns refill, we'll compute: tokens_to_add = elapsed_ns * min_max / 60_000_000_000
        // Store min_max directly and compute on the fly.

        RateLimiter {
            sec_tokens: AtomicU64::new(sec_max), // Start with full bucket
            sec_max,
            sec_refill_rate,
            sec_last_refill_ns: AtomicU64::new(now_ns),
            min_tokens: AtomicU64::new(min_max), // Start with full bucket
            min_max,
            min_last_refill_ns: AtomicU64::new(now_ns),
        }
    }

    /// Get current time in nanoseconds
    fn current_ns() -> u64 {
        Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64
    }

    /// Refill second-level bucket using atomic CAS
    fn refill_second(&self) {
        let now_ns = Self::current_ns();
        let last_ns = self.sec_last_refill_ns.load(Ordering::SeqCst);

        if now_ns > last_ns {
            let elapsed_ns = now_ns - last_ns;
            // tokens_to_add = elapsed_seconds * refill_rate = (elapsed_ns / 1e9) * sec_refill_rate
            let tokens_to_add = if elapsed_ns < 1_000_000_000 {
                // Less than 1 second: fractional refill
                (elapsed_ns * self.sec_refill_rate) / 1_000_000_000
            } else {
                let elapsed_secs = elapsed_ns / 1_000_000_000;
                elapsed_secs * self.sec_refill_rate
            };

            if tokens_to_add > 0 {
                // Try to update the timestamp with CAS
                match self.sec_last_refill_ns.compare_exchange(
                    last_ns,
                    now_ns,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        // We won the CAS, add tokens
                        let mut current = self.sec_tokens.load(Ordering::SeqCst);
                        loop {
                            let new_val = (current + tokens_to_add).min(self.sec_max);
                            match self.sec_tokens.compare_exchange(
                                current,
                                new_val,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                            ) {
                                Ok(_) => break,
                                Err(actual) => current = actual,
                            }
                        }
                    }
                    Err(_) => {
                        // Another thread already refilled, that's fine
                    }
                }
            }
        }
    }

    /// Refill minute-level bucket using atomic CAS
    fn refill_minute(&self) {
        let now_ns = Self::current_ns();
        let last_ns = self.min_last_refill_ns.load(Ordering::SeqCst);

        if now_ns > last_ns {
            let elapsed_ns = now_ns - last_ns;
            // tokens_to_add = (elapsed_ns * min_max) / 60_000_000_000
            // Use u128 to avoid overflow
            let tokens_to_add = ((elapsed_ns as u128) * (self.min_max as u128) / 60_000_000_000_u128) as u64;

            if tokens_to_add > 0 {
                match self.min_last_refill_ns.compare_exchange(
                    last_ns,
                    now_ns,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => {
                        let mut current = self.min_tokens.load(Ordering::SeqCst);
                        loop {
                            let new_val = (current + tokens_to_add).min(self.min_max);
                            match self.min_tokens.compare_exchange(
                                current,
                                new_val,
                                Ordering::SeqCst,
                                Ordering::SeqCst,
                            ) {
                                Ok(_) => break,
                                Err(actual) => current = actual,
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
        }
    }

    /// Check if a request is allowed and consume a token if so (atomic, lock-free)
    pub fn check_and_increment(&self, max_per_second: u32, max_per_minute: u32) -> RiskCheckResult {
        let now = Utc::now();

        // Refill both buckets
        self.refill_second();
        self.refill_minute();

        // Try to consume a second-level token using CAS
        let one_token = FP_SCALE;
        let mut current = self.sec_tokens.load(Ordering::SeqCst);
        loop {
            if current < one_token {
                return RiskCheckResult {
                    decision: RiskDecision::Throttle,
                    check_name: "rate_limit_second".to_string(),
                    reason: format!("Rate limit: per-second tokens exhausted (max: {})", max_per_second),
                    current_value: Decimal::ZERO,
                    limit_value: Decimal::from(max_per_second),
                    timestamp: now,
                };
            }
            match self.sec_tokens.compare_exchange(
                current,
                current - one_token,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }

        // Try to consume a minute-level token using CAS
        let mut current = self.min_tokens.load(Ordering::SeqCst);
        loop {
            if current < one_token {
                // Return the second-level token
                let mut sec_current = self.sec_tokens.load(Ordering::SeqCst);
                loop {
                    let new_val = (sec_current + one_token).min(self.sec_max);
                    match self.sec_tokens.compare_exchange(
                        sec_current,
                        new_val,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => break,
                        Err(actual) => sec_current = actual,
                    }
                }
                return RiskCheckResult {
                    decision: RiskDecision::Throttle,
                    check_name: "rate_limit_minute".to_string(),
                    reason: format!("Rate limit: per-minute tokens exhausted (max: {})", max_per_minute),
                    current_value: Decimal::ZERO,
                    limit_value: Decimal::from(max_per_minute),
                    timestamp: now,
                };
            }
            match self.min_tokens.compare_exchange(
                current,
                current - one_token,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }

        RiskCheckResult {
            decision: RiskDecision::Allow,
            check_name: "rate_limit".to_string(),
            reason: "OK".to_string(),
            current_value: Decimal::ONE,
            limit_value: Decimal::from(max_per_second),
            timestamp: now,
        }
    }

    /// Simple boolean check
    pub fn is_allowed(&self, max_per_second: u32, max_per_minute: u32) -> bool {
        matches!(
            self.check_and_increment(max_per_second, max_per_minute).decision,
            RiskDecision::Allow
        )
    }

    /// Get current available tokens (approximate, for monitoring)
    pub fn available_tokens(&self) -> Decimal {
        self.refill_second();
        let fp = self.sec_tokens.load(Ordering::SeqCst);
        Decimal::from(fp) / Decimal::from(FP_SCALE)
    }
}

/// Drawdown tracker
pub struct DrawdownTracker {
    peak_value: RwLock<Decimal>,
    daily_start_value: RwLock<Decimal>,
    weekly_start_value: RwLock<Decimal>,
    daily_start: RwLock<chrono::DateTime<Utc>>,
    weekly_start: RwLock<chrono::DateTime<Utc>>,
}

impl DrawdownTracker {
    pub fn new(initial_value: Decimal) -> Self {
        let now = Utc::now();
        DrawdownTracker {
            peak_value: RwLock::new(initial_value),
            daily_start_value: RwLock::new(initial_value),
            weekly_start_value: RwLock::new(initial_value),
            daily_start: RwLock::new(now),
            weekly_start: RwLock::new(now),
        }
    }

    pub fn update(&self, current_value: Decimal) {
        let mut peak = self.peak_value.write();
        if current_value > *peak {
            *peak = current_value;
        }
    }

    pub fn current_drawdown(&self, current_value: Decimal) -> Decimal {
        let peak = *self.peak_value.read();
        if peak > Decimal::ZERO {
            (peak - current_value) / peak
        } else {
            Decimal::ZERO
        }
    }

    pub fn daily_drawdown(&self, current_value: Decimal) -> Decimal {
        let daily_start = *self.daily_start_value.read();
        if daily_start > Decimal::ZERO {
            (daily_start - current_value) / daily_start
        } else {
            Decimal::ZERO
        }
    }

    pub fn weekly_drawdown(&self, current_value: Decimal) -> Decimal {
        let weekly_start = *self.weekly_start_value.read();
        if weekly_start > Decimal::ZERO {
            (weekly_start - current_value) / weekly_start
        } else {
            Decimal::ZERO
        }
    }

    pub fn reset_daily(&self, value: Decimal) {
        *self.daily_start_value.write() = value;
        *self.daily_start.write() = Utc::now();
    }

    pub fn reset_weekly(&self, value: Decimal) {
        *self.weekly_start_value.write() = value;
        *self.weekly_start.write() = Utc::now();
    }
}

/// Pre-trade risk checker
pub struct RiskHotPath {
    limits: Arc<RiskLimits>,
    kill_switch: Arc<KillSwitch>,
    rate_limiter: Arc<RateLimiter>,
    drawdown_tracker: Arc<DrawdownTracker>,
    positions: Arc<DashMap<Symbol, Position>>,
    total_exposure: Arc<RwLock<Decimal>>,
}

impl RiskHotPath {
    pub fn new(limits: RiskLimits, initial_portfolio_value: Decimal) -> Self {
        RiskHotPath {
            limits: Arc::new(limits),
            kill_switch: Arc::new(KillSwitch::new()),
            rate_limiter: Arc::new(RateLimiter::with_limits(10, 100)),
            drawdown_tracker: Arc::new(DrawdownTracker::new(initial_portfolio_value)),
            positions: Arc::new(DashMap::new()),
            total_exposure: Arc::new(RwLock::new(Decimal::ZERO)),
        }
    }

    /// Run all pre-trade risk checks (alias for check)
    pub fn check_order(&self, order: &Order, portfolio_value: Decimal) -> Vec<RiskCheckResult> {
        self.check(order, portfolio_value)
    }

    /// Run all pre-trade risk checks. Returns all results.
    pub fn check(&self, order: &Order, portfolio_value: Decimal) -> Vec<RiskCheckResult> {
        let mut results = Vec::with_capacity(8);
        let now = Utc::now();

        // 1. Kill switch check
        if self.kill_switch.is_triggered() {
            results.push(RiskCheckResult {
                decision: RiskDecision::Reject,
                check_name: "kill_switch".to_string(),
                reason: "Kill switch is active - all trading halted".to_string(),
                current_value: Decimal::ONE,
                limit_value: Decimal::ZERO,
                timestamp: now,
            });
            return results;
        }

        // 2. Position limit per symbol
        let current_pos = self
            .positions
            .get(&order.symbol)
            .map(|p| p.value().quantity * p.value().mark_price)
            .unwrap_or(Decimal::ZERO);
        let order_notional = order.quantity * order.price.unwrap_or(Decimal::ZERO);
        let new_pos = current_pos + order_notional;
        results.push(RiskCheckResult {
            decision: if new_pos <= self.limits.max_position_per_symbol { RiskDecision::Allow } else { RiskDecision::Reject },
            check_name: "position_limit_symbol".to_string(),
            reason: format!("Position {} would exceed limit {}", new_pos, self.limits.max_position_per_symbol),
            current_value: new_pos,
            limit_value: self.limits.max_position_per_symbol,
            timestamp: now,
        });

        // 3. Order size check (notional)
        results.push(RiskCheckResult {
            decision: if order_notional <= self.limits.max_order_notional { RiskDecision::Allow } else { RiskDecision::Reject },
            check_name: "order_size_notional".to_string(),
            reason: format!("Order notional {} exceeds max {}", order_notional, self.limits.max_order_notional),
            current_value: order_notional,
            limit_value: self.limits.max_order_notional,
            timestamp: now,
        });

        // 4. Order size check (quantity)
        results.push(RiskCheckResult {
            decision: if order.quantity <= self.limits.max_order_quantity { RiskDecision::Allow } else { RiskDecision::Reject },
            check_name: "order_size_quantity".to_string(),
            reason: format!("Order qty {} exceeds max {}", order.quantity, self.limits.max_order_quantity),
            current_value: order.quantity,
            limit_value: self.limits.max_order_quantity,
            timestamp: now,
        });

        // 5. Rate limit check (token bucket)
        let rate_result = self.rate_limiter.check_and_increment(
            self.limits.max_orders_per_second,
            self.limits.max_orders_per_minute,
        );
        results.push(rate_result);

        // 6. Drawdown check
        let dd = self.drawdown_tracker.current_drawdown(portfolio_value);
        results.push(RiskCheckResult {
            decision: if dd <= self.limits.max_drawdown { RiskDecision::Allow } else { RiskDecision::Reject },
            check_name: "max_drawdown".to_string(),
            reason: format!("Drawdown {:.2}% exceeds max {:.2}%", dd * rust_decimal_macros::dec!(100), self.limits.max_drawdown * rust_decimal_macros::dec!(100)),
            current_value: dd,
            limit_value: self.limits.max_drawdown,
            timestamp: now,
        });

        let daily_dd = self.drawdown_tracker.daily_drawdown(portfolio_value);
        results.push(RiskCheckResult {
            decision: if daily_dd <= self.limits.max_daily_drawdown { RiskDecision::Allow } else { RiskDecision::Reject },
            check_name: "daily_drawdown".to_string(),
            reason: format!("Daily drawdown {:.2}% exceeds max {:.2}%", daily_dd * rust_decimal_macros::dec!(100), self.limits.max_daily_drawdown * rust_decimal_macros::dec!(100)),
            current_value: daily_dd,
            limit_value: self.limits.max_daily_drawdown,
            timestamp: now,
        });

        // 7. Gross exposure check
        let total_exp = *self.total_exposure.read();
        let new_exposure = total_exp + order_notional;
        results.push(RiskCheckResult {
            decision: if new_exposure <= self.limits.max_gross_exposure { RiskDecision::Allow } else { RiskDecision::Reject },
            check_name: "gross_exposure".to_string(),
            reason: format!("Gross exposure {} would exceed {}", new_exposure, self.limits.max_gross_exposure),
            current_value: new_exposure,
            limit_value: self.limits.max_gross_exposure,
            timestamp: now,
        });

        // 8. Concentration check
        if portfolio_value > Decimal::ZERO {
            let concentration = new_pos / portfolio_value;
            results.push(RiskCheckResult {
                decision: if concentration <= self.limits.max_concentration_pct { RiskDecision::Allow } else { RiskDecision::Reject },
                check_name: "concentration".to_string(),
                reason: format!("Concentration {:.1}% exceeds max {:.1}%", concentration * rust_decimal_macros::dec!(100), self.limits.max_concentration_pct * rust_decimal_macros::dec!(100)),
                current_value: concentration,
                limit_value: self.limits.max_concentration_pct,
                timestamp: now,
            });
        }

        // 9. Margin check
        let margin_required = order_notional * self.limits.initial_margin_ratio;
        let margin_available = portfolio_value - total_exp * self.limits.maintenance_margin_ratio;
        results.push(RiskCheckResult {
            decision: if margin_required <= margin_available { RiskDecision::Allow } else { RiskDecision::Reject },
            check_name: "margin".to_string(),
            reason: format!("Required margin {} > available {}", margin_required, margin_available),
            current_value: margin_required,
            limit_value: margin_available,
            timestamp: now,
        });

        results
    }

    /// Quick check: is the order allowed?
    pub fn is_allowed(&self, order: &Order, portfolio_value: Decimal) -> RiskDecision {
        let results = self.check(order, portfolio_value);
        if results.iter().any(|r| r.decision == RiskDecision::Reject) {
            RiskDecision::Reject
        } else if results.iter().any(|r| r.decision == RiskDecision::Throttle) {
            RiskDecision::Throttle
        } else {
            RiskDecision::Allow
        }
    }

    /// Trigger the kill switch
    pub fn trigger_kill_switch(&self, reason: &str) {
        self.kill_switch.trigger(reason);
    }

    /// Reset the kill switch
    pub fn reset_kill_switch(&self) {
        self.kill_switch.reset();
    }

    /// Check if kill switch is active
    pub fn is_kill_switch_active(&self) -> bool {
        self.kill_switch.is_triggered()
    }

    /// Set the kill switch state
    pub fn set_kill_switch(&self, active: bool, reason: &str) {
        if active {
            self.kill_switch.trigger(reason);
        } else {
            self.kill_switch.reset();
        }
    }

    /// Update position tracking
    pub fn update_position(&self, position: Position) {
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Update total exposure
    pub fn update_exposure(&self, exposure: Decimal) {
        *self.total_exposure.write() = exposure;
    }

    /// Update drawdown tracker
    pub fn update_portfolio_value(&self, value: Decimal) {
        self.drawdown_tracker.update(value);
    }

    /// Get risk limits reference
    pub fn get_limits(&self) -> &RiskLimits {
        &self.limits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_kill_switch() {
        let ks = KillSwitch::new();
        assert!(!ks.is_triggered());
        ks.trigger("Test emergency");
        assert!(ks.is_triggered());
        ks.reset();
        assert!(!ks.is_triggered());
    }

    #[test]
    fn test_risk_hot_path_allow() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits, dec!(1000000));
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);
        let decision = risk.is_allowed(&order, dec!(1000000));
        assert_eq!(decision, RiskDecision::Allow);
    }

    #[test]
    fn test_risk_hot_path_kill_switch() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits, dec!(1000000));
        risk.trigger_kill_switch("Emergency stop");
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);
        let decision = risk.is_allowed(&order, dec!(1000000));
        assert_eq!(decision, RiskDecision::Reject);
    }

    #[test]
    fn test_risk_order_size_reject() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits, dec!(1000000));
        let mut order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(100), ExchangeId::Paper);
        order.price = Some(dec!(50000));
        let decision = risk.is_allowed(&order, dec!(1000000));
        assert_eq!(decision, RiskDecision::Reject);
    }

    #[test]
    fn test_rate_limiter_basic() {
        let rl = RateLimiter::with_limits(5, 10);
        for _ in 0..5 {
            let result = rl.check_and_increment(5, 10);
            assert_eq!(result.decision, RiskDecision::Allow);
        }
        let result = rl.check_and_increment(5, 10);
        assert_eq!(result.decision, RiskDecision::Throttle);
    }

    #[test]
    fn test_rate_limiter_refill() {
        let rl = RateLimiter::with_limits(100, 1000);
        for _ in 0..100 {
            rl.check_and_increment(100, 1000);
        }
        assert_eq!(rl.check_and_increment(100, 1000).decision, RiskDecision::Throttle);
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert_eq!(rl.check_and_increment(100, 1000).decision, RiskDecision::Allow);
    }

    #[test]
    fn test_set_kill_switch() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits, dec!(1000000));
        risk.set_kill_switch(true, "Manual trigger");
        assert!(risk.is_kill_switch_active());
        risk.set_kill_switch(false, "Reset");
        assert!(!risk.is_kill_switch_active());
    }

    #[test]
    fn test_drawdown_tracker() {
        let tracker = DrawdownTracker::new(dec!(100000));
        assert_eq!(tracker.current_drawdown(dec!(100000)), Decimal::ZERO);
        tracker.update(dec!(120000));
        let dd = tracker.current_drawdown(dec!(100000));
        assert!(dd > Decimal::ZERO);
        assert!(dd < dec!(1));
    }

    #[test]
    fn test_quantity_limit_reject() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits, dec!(1000000));
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(100), ExchangeId::Paper);
        let decision = risk.is_allowed(&order, dec!(1000000));
        assert_eq!(decision, RiskDecision::Reject);
    }

    #[test]
    fn test_check_order_alias() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits, dec!(1000000));
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);
        let results = risk.check_order(&order, dec!(1000000));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_get_limits() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits.clone(), dec!(1000000));
        assert_eq!(risk.get_limits().max_orders_per_second, limits.max_orders_per_second);
    }

    #[test]
    fn test_position_limit_rejection() {
        let limits = RiskLimits {
            max_position_per_symbol: dec!(100),
            ..RiskLimits::default()
        };
        let risk = RiskHotPath::new(limits, dec!(1000000));
        let mut order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper);
        order.price = Some(dec!(50000));
        let decision = risk.is_allowed(&order, dec!(1000000));
        assert_eq!(decision, RiskDecision::Reject);
    }

    #[test]
    fn test_drawdown_rejection() {
        let limits = RiskLimits {
            max_drawdown: dec!(0.001), // Very tight
            ..RiskLimits::default()
        };
        let risk = RiskHotPath::new(limits, dec!(100000));
        risk.update_portfolio_value(dec!(200000)); // Set high peak
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.001), ExchangeId::Paper);
        // Now portfolio is 100000 but peak was 200000 -> 50% drawdown
        let results = risk.check(&order, dec!(100000));
        let dd_check = results.iter().find(|r| r.check_name == "max_drawdown").unwrap();
        assert_eq!(dd_check.decision, RiskDecision::Reject);
    }

    #[test]
    fn test_all_pre_trade_checks_present() {
        let limits = RiskLimits::default();
        let risk = RiskHotPath::new(limits, dec!(1000000));
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);
        let results = risk.check(&order, dec!(1000000));
        // Should have: position_limit, order_size_notional, order_size_quantity,
        // rate_limit, max_drawdown, daily_drawdown, gross_exposure, margin
        assert!(results.len() >= 7);
    }

    #[test]
    fn test_rate_limiter_minute_limit() {
        let rl = RateLimiter::with_limits(1000, 3); // Very low minute limit
        for _ in 0..3 {
            rl.check_and_increment(1000, 3);
        }
        let result = rl.check_and_increment(1000, 3);
        assert_eq!(result.decision, RiskDecision::Throttle);
        assert_eq!(result.check_name, "rate_limit_minute");
    }
}
