//! PyO3 Bridge - Main module definition
//!
//! Exposes all Rust components to Python with GIL release
//! for long-running operations.

#[cfg(feature = "pyo3_bridge")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3_bridge")]
use crate::core::types::*;
#[cfg(feature = "pyo3_bridge")]
use crate::core::orderbook::OrderBook;
#[cfg(feature = "pyo3_bridge")]
use crate::execution::engine::ExecutionEngine;
#[cfg(feature = "pyo3_bridge")]
use crate::risk::hot_path::RiskHotPath;
#[cfg(feature = "pyo3_bridge")]
use crate::risk::circuit_breaker::CircuitBreaker;
#[cfg(feature = "pyo3_bridge")]
use crate::ingestion::collector::DataCollector;
#[cfg(feature = "pyo3_bridge")]
use crate::ingestion::pipeline::DataPipeline;
#[cfg(feature = "pyo3_bridge")]
use crate::execution::vwap_exec::VwapExecutor;
#[cfg(feature = "pyo3_bridge")]
use crate::execution::twap::TwapExecutor;
#[cfg(feature = "pyo3_bridge")]
use crate::execution::sor::SmartOrderRouter;
#[cfg(feature = "pyo3_bridge")]
use crate::risk::exposure::ExposureTracker;
#[cfg(feature = "pyo3_bridge")]
use crate::pyo3_bridge::types::*;

// ============================================================================
// PyO3-wrapped OrderBook
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyOrderBook {
    inner: parking_lot::Mutex<OrderBook>,
}

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyOrderBook {
    #[new]
    fn new(symbol: String, exchange: &str, max_depth: usize) -> PyResult<Self> {
        let exchange_id = parse_exchange(exchange)?;
        let ob = OrderBook::new(Symbol::new(symbol), exchange_id, max_depth);
        Ok(PyOrderBook { inner: parking_lot::Mutex::new(ob) })
    }

    fn update_bid(&self, price: f64, quantity: f64, order_count: u32) {
        let price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let quantity = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        self.inner.lock().update_bid(price, quantity, order_count);
    }

    fn update_ask(&self, price: f64, quantity: f64, order_count: u32) {
        let price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let quantity = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        self.inner.lock().update_ask(price, quantity, order_count);
    }

    fn apply_snapshot(&self, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        let bid_levels: Vec<OrderBookLevel> = bids.into_iter()
            .map(|(p, q)| OrderBookLevel { price: Decimal::from_f64_retain(p).unwrap_or(Decimal::ZERO), quantity: Decimal::from_f64_retain(q).unwrap_or(Decimal::ZERO), order_count: 1 })
            .collect();
        let ask_levels: Vec<OrderBookLevel> = asks.into_iter()
            .map(|(p, q)| OrderBookLevel { price: Decimal::from_f64_retain(p).unwrap_or(Decimal::ZERO), quantity: Decimal::from_f64_retain(q).unwrap_or(Decimal::ZERO), order_count: 1 })
            .collect();
        self.inner.lock().apply_snapshot(bid_levels, ask_levels);
    }

    fn midpoint(&self) -> Option<f64> {
        self.inner.lock().compute_midpoint().and_then(|d| d.to_string().parse().ok())
    }

    fn weighted_midpoint(&self) -> Option<f64> {
        self.inner.lock().compute_weighted_mid().and_then(|d| d.to_string().parse().ok())
    }

    fn vwap_from_depth(&self, side: &str, quantity: f64) -> Option<f64> {
        let side = if side == "buy" { Side::Buy } else { Side::Sell };
        let qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        self.inner.lock().compute_vwap(side, qty).and_then(|d| d.to_string().parse().ok())
    }

    fn spread(&self) -> Option<f64> {
        self.inner.lock().compute_spread().and_then(|d| d.to_string().parse().ok())
    }

    fn spread_pct(&self) -> Option<f64> {
        self.inner.lock().compute_spread_pct().and_then(|d| d.to_string().parse().ok())
    }

    fn imbalance(&self, levels: usize) -> f64 {
        self.inner.lock().compute_imbalance(levels).to_string().parse().unwrap_or(0.0)
    }

    fn best_bid(&self) -> Option<f64> {
        self.inner.lock().best_bid().and_then(|d| d.to_string().parse().ok())
    }

    fn best_ask(&self) -> Option<f64> {
        self.inner.lock().best_ask().and_then(|d| d.to_string().parse().ok())
    }

    fn depth(&self) -> usize {
        let ob = self.inner.lock();
        ob.bid_depth() + ob.ask_depth()
    }

    fn snapshot(&self) -> String {
        let snap = self.inner.lock().snapshot();
        serde_json::to_string(&snap).unwrap_or_default()
    }
}

// ============================================================================
// PyO3-wrapped ExecutionEngine
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyExecutionEngine {
    inner: ExecutionEngine,
}

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyExecutionEngine {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyExecutionEngine { inner: ExecutionEngine::new(std::collections::HashMap::new()) })
    }

    fn create_order(&self, symbol: String, side: &str, quantity: f64, exchange: &str) -> PyResult<PyOrder> {
        let side = if side == "buy" { Side::Buy } else { Side::Sell };
        let exchange_id = parse_exchange(exchange)?;
        let qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let order = self.inner.create_market_order(Symbol::new(symbol), side, qty, exchange_id);
        Ok(PyOrder::from(&order))
    }

    fn submit_order(&self, order: &mut PyOrder) -> PyResult<()> {
        let side = match order.side.as_str() {
            "buy" => Side::Buy,
            "sell" => Side::Sell,
            _ => Side::Buy,
        };
        let exchange_id = match parse_exchange(&order.exchange) {
            Ok(e) => e,
            Err(e) => return Err(e),
        };
        let qty = Decimal::from_f64_retain(order.quantity).unwrap_or(Decimal::ZERO);
        let mut rust_order = self.inner.create_market_order(
            Symbol::new(order.symbol.clone()), side, qty, exchange_id,
        );
        // Restore original order ID so it matches
        rust_order.id = OrderId::from_exchange(&order.id);
        if let Some(p) = order.price {
            rust_order.price = Some(Decimal::from_f64_retain(p).unwrap_or(Decimal::ZERO));
        }
        match self.inner.submit_order(&mut rust_order) {
            Ok(()) => {
                // Update the Python order with the submitted state
                order.status = rust_order.status.to_string();
                order.updated_at = rust_order.updated_at.to_rfc3339();
                Ok(())
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    fn process_fill(&self, order_id: String, price: f64, quantity: f64, commission: f64) -> PyResult<PyExecutionReport> {
        let oid = OrderId::from_exchange(&order_id);
        let fill_price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let fill_qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let comm = Decimal::from_f64_retain(commission).unwrap_or(Decimal::ZERO);
        match self.inner.process_fill(&oid, fill_price, fill_qty, comm, false) {
            Ok(report) => Ok(PyExecutionReport::from(&report)),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    fn cancel_order(&self, order_id: String) -> PyResult<()> {
        let oid = OrderId::from_exchange(&order_id);
        match self.inner.cancel_order(&oid) {
            Ok(_) => Ok(()),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    fn get_positions(&self) -> Vec<PyPosition> {
        self.inner.get_positions().values().map(|p| PyPosition::from(p)).collect()
    }

    fn get_open_orders(&self) -> Vec<PyOrder> {
        self.inner.get_open_orders().iter().map(|o| PyOrder::from(o)).collect()
    }

    fn get_order(&self, order_id: String) -> Option<PyOrder> {
        let oid = OrderId::from_exchange(&order_id);
        self.inner.get_order(&oid).map(|o| PyOrder::from(&o))
    }

    fn set_slippage_model(&self, model: &str) {
        let slippage_model = match model {
            "fixed" => SlippageModel::Fixed,
            "percentage" => SlippageModel::Percentage,
            "square_root" => SlippageModel::SquareRoot,
            _ => SlippageModel::Percentage,
        };
        self.inner.set_slippage_model(slippage_model);
    }
}

// ============================================================================
// PyO3-wrapped RiskHotPath
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyRiskHotPath {
    inner: RiskHotPath,
}

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyRiskHotPath {
    #[new]
    fn new(portfolio_value: f64) -> PyResult<Self> {
        let pv = Decimal::from_f64_retain(portfolio_value).unwrap_or(Decimal::ONE);
        Ok(PyRiskHotPath { inner: RiskHotPath::new(RiskLimits::default(), pv) })
    }

    fn check_order(&self, symbol: String, side: &str, quantity: f64, price: Option<f64>, portfolio_value: f64) -> Vec<PyRiskCheckResult> {
        let side = if side == "buy" { Side::Buy } else { Side::Sell };
        let qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let pv = Decimal::from_f64_retain(portfolio_value).unwrap_or(Decimal::ZERO);
        let mut order = Order::new_market(Symbol::new(symbol), side, qty, ExchangeId::Paper);
        if let Some(p) = price { order.price = Some(Decimal::from_f64_retain(p).unwrap_or(Decimal::ZERO)); }
        self.inner.check(&order, pv).iter().map(|r| PyRiskCheckResult::from(r)).collect()
    }

    fn is_kill_switch_active(&self) -> bool {
        self.inner.is_kill_switch_active()
    }

    fn set_kill_switch(&self, active: bool, reason: &str) {
        self.inner.set_kill_switch(active, reason);
    }

    fn get_limits(&self) -> String {
        let limits = self.inner.get_limits();
        serde_json::to_string(limits).unwrap_or_default()
    }
}

// ============================================================================
// PyO3-wrapped CircuitBreakerManager
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyCircuitBreakerManager {
    inner: CircuitBreaker,
}

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyCircuitBreakerManager {
    #[new]
    fn new(price_threshold_pct: f64, window_secs: u64, min_volume: f64, max_spread_pct: f64) -> PyResult<Self> {
        let config = CircuitBreakerConfig {
            price_move_threshold_pct: Decimal::from_f64_retain(price_threshold_pct).unwrap_or(dec!(5)),
            price_move_window_secs: window_secs,
            min_volume_threshold: Decimal::from_f64_retain(min_volume).unwrap_or(dec!(100)),
            max_spread_pct: Decimal::from_f64_retain(max_spread_pct).unwrap_or(dec!(1)),
            max_realized_vol: dec!(200),
            cooldown_secs: 300,
        };
        let min_vol = config.min_volume_threshold;
        let max_spread = config.max_spread_pct;
        Ok(PyCircuitBreakerManager { inner: CircuitBreaker::new(config, min_vol, max_spread) })
    }

    fn check_price(&self, symbol: String, price: f64, timestamp_secs: i64) -> String {
        let price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let ts = chrono::DateTime::from_timestamp(timestamp_secs, 0).unwrap_or(chrono::Utc::now());
        self.inner.check_price(&Symbol::new(symbol), price, ts).to_string()
    }

    fn check_volume(&self, volume: f64) -> String {
        let vol = Decimal::from_f64_retain(volume).unwrap_or(Decimal::ZERO);
        self.inner.check_volume(vol).to_string()
    }

    fn check_spread(&self, bid: f64, ask: f64) -> String {
        let bid = Decimal::from_f64_retain(bid).unwrap_or(Decimal::ZERO);
        let ask = Decimal::from_f64_retain(ask).unwrap_or(Decimal::ZERO);
        self.inner.check_spread(bid, ask).to_string()
    }

    fn check_all(&self, symbol: String, price: f64, bid: f64, ask: f64, volume: f64, timestamp_secs: i64) -> String {
        let sym = Symbol::new(symbol);
        let price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let bid = Decimal::from_f64_retain(bid).unwrap_or(Decimal::ZERO);
        let ask = Decimal::from_f64_retain(ask).unwrap_or(Decimal::ZERO);
        let vol = Decimal::from_f64_retain(volume).unwrap_or(Decimal::ZERO);
        let ts = chrono::DateTime::from_timestamp(timestamp_secs, 0).unwrap_or(chrono::Utc::now());
        let (state, trigger) = self.inner.check_all(&sym, price, bid, ask, vol, ts);
        serde_json::json!({"state": state.to_string(), "trigger": trigger.map(|t| format!("{:?}", t))}).to_string()
    }

    fn manual_override(&self, reason: &str) {
        self.inner.manual_override(reason);
    }
}

// ============================================================================
// PyO3-wrapped DataPipeline
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyDataPipeline {
    inner: parking_lot::Mutex<DataPipeline>,
}

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyDataPipeline {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PyDataPipeline {
            inner: parking_lot::Mutex::new(DataPipeline::new(crate::ingestion::pipeline::PipelineConfig::default())),
        })
    }

    fn process_tick(&self, symbol: String, price: f64, quantity: f64, side: &str) {
        let tick = Tick {
            symbol: Symbol::new(symbol),
            exchange: ExchangeId::Paper,
            price: Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO),
            quantity: Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO),
            side: if side == "buy" { Side::Buy } else { Side::Sell },
            timestamp: chrono::Utc::now(),
            trade_id: uuid::Uuid::new_v4().to_string(),
        };
        self.inner.lock().process_tick(tick);
    }

    fn get_candles(&self) -> Vec<PyCandle> {
        self.inner.lock().get_candles().iter().map(|c| PyCandle::from(c)).collect()
    }

    fn get_quality_stats(&self) -> String {
        let stats = self.inner.lock().get_quality_stats();
        serde_json::to_string(&stats).unwrap_or_default()
    }
}

// ============================================================================
// PyO3-wrapped VwapExecutor
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyVwapExecutor;

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyVwapExecutor {
    #[staticmethod]
    fn create_plan(symbol: String, side: &str, quantity: f64, duration_minutes: u32) -> String {
        let side = if side == "buy" { Side::Buy } else { Side::Sell };
        let qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let config = crate::execution::vwap_exec::VwapConfig {
            total_quantity: qty, symbol: Symbol::new(symbol), side,
            exchange: ExchangeId::Paper, duration_minutes,
            max_participation_rate: dec!(0.1),
            volume_profile: crate::execution::vwap_exec::VwapConfig::default_crypto_profile(),
            price_limit: None,
        };
        let state = VwapExecutor::create_plan(config);
        serde_json::to_string(&state).unwrap_or_default()
    }

    #[staticmethod]
    fn fill_slice(plan_json: &str, slice_index: u32, price: f64, quantity: f64, market_vwap: f64) -> String {
        let mut state: crate::execution::vwap_exec::VwapState = match serde_json::from_str(plan_json) {
            Ok(s) => s, Err(_) => return "{}".to_string(),
        };
        let fill_price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let fill_qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let mvwap = Decimal::from_f64_retain(market_vwap).unwrap_or(Decimal::ZERO);
        VwapExecutor::fill_slice(&mut state, slice_index, fill_price, fill_qty, Decimal::ZERO, mvwap);
        serde_json::to_string(&state).unwrap_or_default()
    }

    #[staticmethod]
    fn performance_vs_benchmark(plan_json: &str) -> f64 {
        let state: crate::execution::vwap_exec::VwapState = match serde_json::from_str(plan_json) {
            Ok(s) => s, Err(_) => return 0.0,
        };
        VwapExecutor::performance_vs_benchmark(&state).to_string().parse().unwrap_or(0.0)
    }

    #[staticmethod]
    fn is_complete(plan_json: &str) -> bool {
        let state: crate::execution::vwap_exec::VwapState = match serde_json::from_str(plan_json) {
            Ok(s) => s, Err(_) => return false,
        };
        VwapExecutor::is_complete(&state)
    }
}

// ============================================================================
// PyO3-wrapped TwapExecutor
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyTwapExecutor;

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyTwapExecutor {
    #[staticmethod]
    fn create_plan(symbol: String, side: &str, quantity: f64, duration_secs: u64, interval_secs: u64, randomize_pct: f64) -> String {
        let side = if side == "buy" { Side::Buy } else { Side::Sell };
        let qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let config = crate::execution::twap::TwapConfig {
            total_quantity: qty, symbol: Symbol::new(symbol), side,
            exchange: ExchangeId::Paper, duration_secs,
            slice_interval_secs: interval_secs,
            randomize_pct: Decimal::from_f64_retain(randomize_pct).unwrap_or(Decimal::ZERO),
            price_limit: None, jitter_pct: dec!(10),
        };
        let state = TwapExecutor::create_plan(config);
        serde_json::to_string(&state).unwrap_or_default()
    }

    #[staticmethod]
    fn next_slice(plan_json: &str) -> Option<u32> {
        let mut state: crate::execution::twap::TwapState = match serde_json::from_str(plan_json) {
            Ok(s) => s, Err(_) => return None,
        };
        TwapExecutor::next_slice(&mut state).map(|s| s.index)
    }

    #[staticmethod]
    fn fill_slice(plan_json: &str, slice_index: u32, price: f64, quantity: f64) -> String {
        let mut state: crate::execution::twap::TwapState = match serde_json::from_str(plan_json) {
            Ok(s) => s, Err(_) => return "{}".to_string(),
        };
        let fill_price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let fill_qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        TwapExecutor::fill_slice(&mut state, slice_index, fill_price, fill_qty, Decimal::ZERO);
        serde_json::to_string(&state).unwrap_or_default()
    }

    #[staticmethod]
    fn participation_rate(plan_json: &str, market_volume: f64) -> f64 {
        let state: crate::execution::twap::TwapState = match serde_json::from_str(plan_json) {
            Ok(s) => s, Err(_) => return 0.0,
        };
        let mv = Decimal::from_f64_retain(market_volume).unwrap_or(Decimal::ZERO);
        TwapExecutor::participation_rate(&state, mv).to_string().parse().unwrap_or(0.0)
    }

    #[staticmethod]
    fn implementation_shortfall(plan_json: &str, arrival_price: f64) -> f64 {
        let state: crate::execution::twap::TwapState = match serde_json::from_str(plan_json) {
            Ok(s) => s, Err(_) => return 0.0,
        };
        let ap = Decimal::from_f64_retain(arrival_price).unwrap_or(Decimal::ZERO);
        TwapExecutor::implementation_shortfall(&state, ap).to_string().parse().unwrap_or(0.0)
    }
}

// ============================================================================
// PyO3-wrapped SmartOrderRouter
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PySmartOrderRouter {
    inner: parking_lot::Mutex<SmartOrderRouter>,
}

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PySmartOrderRouter {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(PySmartOrderRouter { inner: parking_lot::Mutex::new(SmartOrderRouter::new()) })
    }

    fn route_order(&self, symbol: String, side: &str, quantity: f64) -> String {
        let side = if side == "buy" { Side::Buy } else { Side::Sell };
        let qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let decision = self.inner.lock().route_market_order(&Symbol::new(symbol), side, qty);
        serde_json::to_string(&decision).unwrap_or_default()
    }

    fn add_exchange(&self, exchange: &str, maker_fee: f64, taker_fee: f64) {
        let exchange_id = match parse_exchange(exchange) { Ok(e) => e, Err(_) => return };
        let config = ExchangeConfig {
            exchange: exchange_id, api_key: String::new(), api_secret: String::new(),
            passphrase: None, rest_url: String::new(), ws_url: String::new(),
            rate_limit_per_second: 10,
            maker_fee: Decimal::from_f64_retain(maker_fee).unwrap_or(dec!(0.001)),
            taker_fee: Decimal::from_f64_retain(taker_fee).unwrap_or(dec!(0.001)),
        };
        self.inner.lock().add_exchange(config);
    }

    fn remove_exchange(&self, exchange: &str) {
        let exchange_id = match parse_exchange(exchange) { Ok(e) => e, Err(_) => return };
        self.inner.lock().remove_exchange(exchange_id);
    }

    fn update_book(&self, symbol: String, exchange: &str, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) {
        let exchange_id = match parse_exchange(exchange) { Ok(e) => e, Err(_) => return };
        let snap = OrderBookSnapshot {
            symbol: Symbol::new(symbol.clone()),
            exchange: exchange_id,
            bids: bids.into_iter().map(|(p, q)| OrderBookLevel { price: Decimal::from_f64_retain(p).unwrap_or(Decimal::ZERO), quantity: Decimal::from_f64_retain(q).unwrap_or(Decimal::ZERO), order_count: 1 }).collect(),
            asks: asks.into_iter().map(|(p, q)| OrderBookLevel { price: Decimal::from_f64_retain(p).unwrap_or(Decimal::ZERO), quantity: Decimal::from_f64_retain(q).unwrap_or(Decimal::ZERO), order_count: 1 }).collect(),
            timestamp: chrono::Utc::now(),
            sequence: 1,
        };
        self.inner.lock().update_book(Symbol::new(symbol), exchange_id, snap);
    }
}

// ============================================================================
// PyO3-wrapped ExposureTracker
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pyclass]
pub struct PyExposureTracker {
    inner: ExposureTracker,
}

#[cfg(feature = "pyo3_bridge")]
#[pymethods]
impl PyExposureTracker {
    #[new]
    fn new(portfolio_value: f64) -> PyResult<Self> {
        let pv = Decimal::from_f64_retain(portfolio_value).unwrap_or(Decimal::ONE);
        Ok(PyExposureTracker { inner: ExposureTracker::new(pv) })
    }

    fn update_position(&self, symbol: String, side: &str, quantity: f64, price: f64) {
        let side = if side == "buy" { Side::Buy } else { Side::Sell };
        let qty = Decimal::from_f64_retain(quantity).unwrap_or(Decimal::ZERO);
        let mark_price = Decimal::from_f64_retain(price).unwrap_or(Decimal::ZERO);
        let position = Position {
            symbol: Symbol::new(symbol), side, quantity: qty, entry_price: mark_price, mark_price,
            unrealized_pnl: Decimal::ZERO, realized_pnl: Decimal::ZERO, liquidation_price: None,
            leverage: Decimal::ONE, margin: Decimal::ZERO, exchange: ExchangeId::Paper,
            opened_at: chrono::Utc::now(), updated_at: chrono::Utc::now(),
        };
        self.inner.update_position(position);
    }

    fn net_exposure(&self) -> f64 {
        self.inner.net_exposure().to_string().parse().unwrap_or(0.0)
    }

    fn gross_exposure(&self) -> f64 {
        self.inner.gross_exposure().to_string().parse().unwrap_or(0.0)
    }

    fn long_exposure(&self) -> f64 {
        self.inner.long_exposure().to_string().parse().unwrap_or(0.0)
    }

    fn short_exposure(&self) -> f64 {
        self.inner.short_exposure().to_string().parse().unwrap_or(0.0)
    }

    fn delta_adjusted(&self) -> f64 {
        self.inner.delta_adjusted().to_string().parse().unwrap_or(0.0)
    }

    fn beta_adjusted(&self) -> f64 {
        self.inner.beta_adjusted().to_string().parse().unwrap_or(0.0)
    }

    fn concentration_hhi(&self) -> f64 {
        self.inner.concentration_hhi().to_string().parse().unwrap_or(0.0)
    }
}

// ============================================================================
// Helper functions
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
fn parse_exchange(s: &str) -> PyResult<ExchangeId> {
    match s.to_lowercase().as_str() {
        "binance" => Ok(ExchangeId::Binance),
        "bybit" => Ok(ExchangeId::Bybit),
        "okx" => Ok(ExchangeId::OKX),
        "paper" => Ok(ExchangeId::Paper),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown exchange: {}", s))),
    }
}

// ============================================================================
// Module definition
// ============================================================================

#[cfg(feature = "pyo3_bridge")]
#[pymodule]
fn _acms_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrder>()?;
    m.add_class::<PyPosition>()?;
    m.add_class::<PyCandle>()?;
    m.add_class::<PyExecutionReport>()?;
    m.add_class::<PyRiskCheckResult>()?;
    m.add_class::<PyOrderBook>()?;
    m.add_class::<PyExecutionEngine>()?;
    m.add_class::<PyRiskHotPath>()?;
    m.add_class::<PyCircuitBreakerManager>()?;
    m.add_class::<PyDataPipeline>()?;
    m.add_class::<PyVwapExecutor>()?;
    m.add_class::<PyTwapExecutor>()?;
    m.add_class::<PySmartOrderRouter>()?;
    m.add_class::<PyExposureTracker>()?;
    Ok(())
}

/// Initialize the PyO3 module (stub for non-pyo3 builds)
pub fn init_module() {}
