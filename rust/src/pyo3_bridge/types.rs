//! PyO3 Bridge - Type conversions
//!
//! Converts Rust types to Python-compatible types with proper
//! #[pyclass] and #[pymethods] decorators.

use crate::core::types::*;

/// Python-compatible order representation
#[derive(Debug, Clone)]
#[cfg(feature = "pyo3_bridge")]
#[pyo3::pyclass]
pub struct PyOrder {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub client_order_id: Option<String>,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String,
    #[pyo3(get)]
    pub order_type: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub time_in_force: String,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub price: Option<f64>,
    #[pyo3(get)]
    pub stop_price: Option<f64>,
    #[pyo3(get)]
    pub filled_quantity: f64,
    #[pyo3(get)]
    pub average_fill_price: f64,
    #[pyo3(get)]
    pub exchange: String,
    #[pyo3(get)]
    pub commission: f64,
    #[pyo3(get)]
    pub created_at: String,
    #[pyo3(get)]
    pub updated_at: String,
}

#[cfg(feature = "pyo3_bridge")]
impl From<&Order> for PyOrder {
    fn from(order: &Order) -> Self {
        PyOrder {
            id: order.id.as_str().to_string(),
            client_order_id: order.client_order_id.clone(),
            symbol: order.symbol.to_string(),
            side: order.side.to_string(),
            order_type: order.order_type.to_string(),
            status: order.status.to_string(),
            time_in_force: order.time_in_force.to_string(),
            quantity: order.quantity.to_string().parse().unwrap_or(0.0),
            price: order.price.map(|p| p.to_string().parse().unwrap_or(0.0)),
            stop_price: order.stop_price.map(|p| p.to_string().parse().unwrap_or(0.0)),
            filled_quantity: order.filled_quantity.to_string().parse().unwrap_or(0.0),
            average_fill_price: order.average_fill_price.to_string().parse().unwrap_or(0.0),
            exchange: order.exchange.to_string(),
            commission: order.commission.to_string().parse().unwrap_or(0.0),
            created_at: order.created_at.to_rfc3339(),
            updated_at: order.updated_at.to_rfc3339(),
        }
    }
}

/// Python-compatible position representation
#[derive(Debug, Clone)]
#[cfg(feature = "pyo3_bridge")]
#[pyo3::pyclass]
pub struct PyPosition {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub entry_price: f64,
    #[pyo3(get)]
    pub mark_price: f64,
    #[pyo3(get)]
    pub unrealized_pnl: f64,
    #[pyo3(get)]
    pub realized_pnl: f64,
    #[pyo3(get)]
    pub leverage: f64,
    #[pyo3(get)]
    pub exchange: String,
}

#[cfg(feature = "pyo3_bridge")]
impl From<&Position> for PyPosition {
    fn from(pos: &Position) -> Self {
        PyPosition {
            symbol: pos.symbol.to_string(),
            side: pos.side.to_string(),
            quantity: pos.quantity.to_string().parse().unwrap_or(0.0),
            entry_price: pos.entry_price.to_string().parse().unwrap_or(0.0),
            mark_price: pos.mark_price.to_string().parse().unwrap_or(0.0),
            unrealized_pnl: pos.unrealized_pnl.to_string().parse().unwrap_or(0.0),
            realized_pnl: pos.realized_pnl.to_string().parse().unwrap_or(0.0),
            leverage: pos.leverage.to_string().parse().unwrap_or(1.0),
            exchange: pos.exchange.to_string(),
        }
    }
}

/// Python-compatible candle representation
#[derive(Debug, Clone)]
#[cfg(feature = "pyo3_bridge")]
#[pyo3::pyclass]
pub struct PyCandle {
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub timeframe: String,
    #[pyo3(get)]
    pub open_time: String,
    #[pyo3(get)]
    pub open: f64,
    #[pyo3(get)]
    pub high: f64,
    #[pyo3(get)]
    pub low: f64,
    #[pyo3(get)]
    pub close: f64,
    #[pyo3(get)]
    pub volume: f64,
    #[pyo3(get)]
    pub quote_volume: f64,
    #[pyo3(get)]
    pub trades: u64,
}

#[cfg(feature = "pyo3_bridge")]
impl From<&Candle> for PyCandle {
    fn from(c: &Candle) -> Self {
        PyCandle {
            symbol: c.symbol.to_string(),
            timeframe: c.timeframe.to_string(),
            open_time: c.open_time.to_rfc3339(),
            open: c.open.to_string().parse().unwrap_or(0.0),
            high: c.high.to_string().parse().unwrap_or(0.0),
            low: c.low.to_string().parse().unwrap_or(0.0),
            close: c.close.to_string().parse().unwrap_or(0.0),
            volume: c.volume.to_string().parse().unwrap_or(0.0),
            quote_volume: c.quote_volume.to_string().parse().unwrap_or(0.0),
            trades: c.trades,
        }
    }
}

/// Python-compatible execution report
#[derive(Debug, Clone)]
#[cfg(feature = "pyo3_bridge")]
#[pyo3::pyclass]
pub struct PyExecutionReport {
    #[pyo3(get)]
    pub order_id: String,
    #[pyo3(get)]
    pub symbol: String,
    #[pyo3(get)]
    pub side: String,
    #[pyo3(get)]
    pub order_type: String,
    #[pyo3(get)]
    pub status: String,
    #[pyo3(get)]
    pub quantity: f64,
    #[pyo3(get)]
    pub filled_quantity: f64,
    #[pyo3(get)]
    pub average_price: f64,
    #[pyo3(get)]
    pub commission: f64,
    #[pyo3(get)]
    pub slippage: f64,
    #[pyo3(get)]
    pub latency_us: u64,
    #[pyo3(get)]
    pub exchange: String,
    #[pyo3(get)]
    pub timestamp: String,
}

#[cfg(feature = "pyo3_bridge")]
impl From<&ExecutionReport> for PyExecutionReport {
    fn from(r: &ExecutionReport) -> Self {
        PyExecutionReport {
            order_id: r.order_id.as_str().to_string(),
            symbol: r.symbol.to_string(),
            side: r.side.to_string(),
            order_type: r.order_type.to_string(),
            status: r.status.to_string(),
            quantity: r.quantity.to_string().parse().unwrap_or(0.0),
            filled_quantity: r.filled_quantity.to_string().parse().unwrap_or(0.0),
            average_price: r.average_price.to_string().parse().unwrap_or(0.0),
            commission: r.commission.to_string().parse().unwrap_or(0.0),
            slippage: r.slippage.to_string().parse().unwrap_or(0.0),
            latency_us: r.latency_us,
            exchange: r.exchange.to_string(),
            timestamp: r.timestamp.to_rfc3339(),
        }
    }
}

/// Python-compatible risk check result
#[derive(Debug, Clone)]
#[cfg(feature = "pyo3_bridge")]
#[pyo3::pyclass]
pub struct PyRiskCheckResult {
    #[pyo3(get)]
    pub decision: String,
    #[pyo3(get)]
    pub check_name: String,
    #[pyo3(get)]
    pub reason: String,
    #[pyo3(get)]
    pub current_value: f64,
    #[pyo3(get)]
    pub limit_value: f64,
}

#[cfg(feature = "pyo3_bridge")]
impl From<&RiskCheckResult> for PyRiskCheckResult {
    fn from(r: &RiskCheckResult) -> Self {
        PyRiskCheckResult {
            decision: r.decision.to_string(),
            check_name: r.check_name.clone(),
            reason: r.reason.clone(),
            current_value: r.current_value.to_string().parse().unwrap_or(0.0),
            limit_value: r.limit_value.to_string().parse().unwrap_or(0.0),
        }
    }
}

/// Stub types for when pyo3_bridge feature is not enabled
#[cfg(not(feature = "pyo3_bridge"))]
pub struct PyOrder;
#[cfg(not(feature = "pyo3_bridge"))]
pub struct PyPosition;
#[cfg(not(feature = "pyo3_bridge"))]
pub struct PyCandle;
#[cfg(not(feature = "pyo3_bridge"))]
pub struct PyExecutionReport;
#[cfg(not(feature = "pyo3_bridge"))]
pub struct PyRiskCheckResult;
