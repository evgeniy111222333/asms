//! Order book performance benchmarks.
//!
//! Run with: `cargo bench --bench orderbook_bench`

use acms_core::core::orderbook::OrderBook;
use acms_core::core::types::{ExchangeId, Side, Symbol};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

fn bench_single_price_level_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_price_level_update");
    for size in [100, 1000, 10000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let mut book = OrderBook::new(Symbol::new("BTC/USDT"), ExchangeId::Binance, 25);
                for i in 0..size {
                    let price = if i % 2 == 0 {
                        dec!(50000) - Decimal::from(i % 100)
                    } else {
                        dec!(50000) + Decimal::from(i % 100)
                    };
                    if i % 2 == 0 {
                        book.update_bid(price, dec!(1), 1);
                    } else {
                        book.update_ask(price, dec!(1), 1);
                    }
                }
                black_box(&book);
            });
        });
    }
    group.finish();
}

fn bench_snapshot_application(c: &mut Criterion) {
    c.bench_function("snapshot_application", |b| {
        let bids: Vec<acms_core::core::types::OrderBookLevel> = (0..100)
            .map(|i| acms_core::core::types::OrderBookLevel {
                price: dec!(50000) - Decimal::from(i),
                quantity: dec!(1),
                order_count: 1,
            })
            .collect();
        let asks: Vec<acms_core::core::types::OrderBookLevel> = (0..100)
            .map(|i| acms_core::core::types::OrderBookLevel {
                price: dec!(50001) + Decimal::from(i),
                quantity: dec!(1),
                order_count: 1,
            })
            .collect();

        b.iter(|| {
            let mut book = OrderBook::new(Symbol::new("BTC/USDT"), ExchangeId::Binance, 25);
            book.apply_snapshot(bids.clone(), asks.clone());
            black_box(&book);
        });
    });
}

fn bench_midpoint_calculation(c: &mut Criterion) {
    let mut book = OrderBook::new(Symbol::new("BTC/USDT"), ExchangeId::Binance, 25);
    for i in 0..1000 {
        let price = if i % 2 == 0 { dec!(50000) - Decimal::from(i % 100) } else { dec!(50000) + Decimal::from(i % 100) };
        if i % 2 == 0 { book.update_bid(price, dec!(1), 1); } else { book.update_ask(price, dec!(1), 1); }
    }

    c.bench_function("compute_midpoint", |b| {
        b.iter(|| black_box(book.compute_midpoint()));
    });
}

fn bench_vwap_from_depth(c: &mut Criterion) {
    let mut book = OrderBook::new(Symbol::new("BTC/USDT"), ExchangeId::Binance, 25);
    for i in 0..100 {
        book.update_ask(dec!(50000) + Decimal::from(i), dec!(10), 1);
        book.update_bid(dec!(50000) - Decimal::from(i), dec!(10), 1);
    }

    c.bench_function("compute_vwap_from_depth", |b| {
        b.iter(|| black_box(book.compute_vwap(Side::Buy, dec!(10))));
    });
}

fn bench_spread_computation(c: &mut Criterion) {
    let mut book = OrderBook::new(Symbol::new("BTC/USDT"), ExchangeId::Binance, 25);
    for i in 0..100 {
        book.update_bid(dec!(50000) - Decimal::from(i), dec!(1), 1);
        book.update_ask(dec!(50001) + Decimal::from(i), dec!(1), 1);
    }

    c.bench_function("compute_spread", |b| {
        b.iter(|| black_box(book.compute_spread()));
    });

    c.bench_function("compute_spread_pct", |b| {
        b.iter(|| black_box(book.compute_spread_pct()));
    });
}

criterion_group!(
    benches,
    bench_single_price_level_update,
    bench_snapshot_application,
    bench_midpoint_calculation,
    bench_vwap_from_depth,
    bench_spread_computation
);
criterion_main!(benches);
