//! Risk check performance benchmarks.
//!
//! Run with: `cargo bench --bench risk_check_bench`

use acms_core::core::types::*;
use acms_core::risk::hot_path::RiskHotPath;
use acms_core::risk::circuit_breaker::CircuitBreaker;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_decimal_macros::dec;

fn bench_kill_switch_check(c: &mut Criterion) {
    let limits = RiskLimits::default();
    let risk = RiskHotPath::new(limits, dec!(1000000));
    risk.trigger_kill_switch("bench");

    let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);

    c.bench_function("kill_switch_check", |b| {
        b.iter(|| black_box(risk.is_allowed(&order, dec!(1000000))));
    });
}

fn bench_position_limit_check(c: &mut Criterion) {
    let limits = RiskLimits::default();
    let risk = RiskHotPath::new(limits, dec!(1000000));

    let mut order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);
    order.price = Some(dec!(50000));

    c.bench_function("position_limit_check", |b| {
        b.iter(|| {
            let results = risk.check(&order, dec!(1000000));
            let pos_check = results.iter().find(|r| r.check_name == "position_limit_symbol");
            black_box(pos_check);
        });
    });
}

fn bench_rate_limit_check(c: &mut Criterion) {
    let limits = RiskLimits {
        max_orders_per_second: 1000,
        max_orders_per_minute: 10000,
        ..RiskLimits::default()
    };
    let risk = RiskHotPath::new(limits, dec!(1000000));

    let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);

    c.bench_function("rate_limit_check", |b| {
        b.iter(|| {
            let results = risk.check(&order, dec!(1000000));
            let rate_check = results.iter().find(|r| r.check_name.starts_with("rate_limit"));
            black_box(rate_check);
        });
    });
}

fn bench_full_pre_trade_check(c: &mut Criterion) {
    let limits = RiskLimits::default();
    let risk = RiskHotPath::new(limits, dec!(1000000));

    let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);

    c.bench_function("full_pre_trade_check", |b| {
        b.iter(|| black_box(risk.check(&order, dec!(1000000))));
    });
}

fn bench_risk_is_allowed(c: &mut Criterion) {
    let limits = RiskLimits::default();
    let risk = RiskHotPath::new(limits, dec!(1000000));

    let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(0.1), ExchangeId::Paper);

    c.bench_function("risk_is_allowed", |b| {
        b.iter(|| black_box(risk.is_allowed(&order, dec!(1000000))));
    });
}

fn bench_circuit_breaker_check(c: &mut Criterion) {
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

    c.bench_function("circuit_breaker_check", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let now = chrono::Utc::now();
            let price = dec!(50000) + rust_decimal::Decimal::from(i % 10);
            i += 1;
            black_box(cb.check(&symbol, price, dec!(49999), dec!(50001), dec!(1000), now));
        });
    });
}

fn bench_circuit_breaker_price_only(c: &mut Criterion) {
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

    c.bench_function("circuit_breaker_price_only", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let now = chrono::Utc::now();
            let price = dec!(50000) + rust_decimal::Decimal::from(i % 10);
            i += 1;
            black_box(cb.check_price(&symbol, price, now));
        });
    });
}

fn bench_kill_switch_toggle(c: &mut Criterion) {
    let limits = RiskLimits::default();
    let risk = RiskHotPath::new(limits, dec!(1000000));

    c.bench_function("kill_switch_toggle", |b| {
        b.iter(|| {
            risk.trigger_kill_switch("bench");
            black_box(risk.is_kill_switch_active());
            risk.reset_kill_switch();
        });
    });
}

criterion_group!(
    benches,
    bench_kill_switch_check,
    bench_position_limit_check,
    bench_rate_limit_check,
    bench_full_pre_trade_check,
    bench_risk_is_allowed,
    bench_circuit_breaker_check,
    bench_circuit_breaker_price_only,
    bench_kill_switch_toggle
);
criterion_main!(benches);
