//! Core types for ACMS
//!
//! All price/quantity values use rust_decimal for exact arithmetic.
//! Timestamps use chrono::DateTime<Utc>.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Enums
// ============================================================================

/// Trading side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Buy,
    Sell,
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "buy"),
            Side::Sell => write!(f, "sell"),
        }
    }
}

/// Order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
    TrailingStop,
    Iceberg,
    TWAP,
    VWAP,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "market"),
            OrderType::Limit => write!(f, "limit"),
            OrderType::Stop => write!(f, "stop"),
            OrderType::StopLimit => write!(f, "stop_limit"),
            OrderType::TrailingStop => write!(f, "trailing_stop"),
            OrderType::Iceberg => write!(f, "iceberg"),
            OrderType::TWAP => write!(f, "twap"),
            OrderType::VWAP => write!(f, "vwap"),
        }
    }
}

/// Order status in lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    Created,
    Validated,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

impl fmt::Display for OrderStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderStatus::Created => write!(f, "created"),
            OrderStatus::Validated => write!(f, "validated"),
            OrderStatus::Submitted => write!(f, "submitted"),
            OrderStatus::PartiallyFilled => write!(f, "partially_filled"),
            OrderStatus::Filled => write!(f, "filled"),
            OrderStatus::Cancelled => write!(f, "cancelled"),
            OrderStatus::Rejected => write!(f, "rejected"),
            OrderStatus::Expired => write!(f, "expired"),
        }
    }
}

/// Time-in-force
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeInForce {
    GTC,  // Good Till Cancel
    IOC,  // Immediate Or Cancel
    FOK,  // Fill Or Kill
    GTD,  // Good Till Date
    DAY,  // Day order
}

impl fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeInForce::GTC => write!(f, "gtc"),
            TimeInForce::IOC => write!(f, "ioc"),
            TimeInForce::FOK => write!(f, "fok"),
            TimeInForce::GTD => write!(f, "gtd"),
            TimeInForce::DAY => write!(f, "day"),
        }
    }
}

/// Exchange identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExchangeId {
    Binance,
    Bybit,
    OKX,
    Paper,
}

impl fmt::Display for ExchangeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExchangeId::Binance => write!(f, "binance"),
            ExchangeId::Bybit => write!(f, "bybit"),
            ExchangeId::OKX => write!(f, "okx"),
            ExchangeId::Paper => write!(f, "paper"),
        }
    }
}

/// Candle timeframe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    S1,
    S5,
    S15,
    S30,
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1,
    W1,
}

impl Timeframe {
    pub fn duration_secs(&self) -> u64 {
        match self {
            Timeframe::S1 => 1,
            Timeframe::S5 => 5,
            Timeframe::S15 => 15,
            Timeframe::S30 => 30,
            Timeframe::M1 => 60,
            Timeframe::M5 => 300,
            Timeframe::M15 => 900,
            Timeframe::M30 => 1800,
            Timeframe::H1 => 3600,
            Timeframe::H4 => 14400,
            Timeframe::D1 => 86400,
            Timeframe::W1 => 604800,
        }
    }
}

impl fmt::Display for Timeframe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Risk check result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskDecision {
    Allow,
    Reject,
    Throttle,
}

impl fmt::Display for RiskDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiskDecision::Allow => write!(f, "allow"),
            RiskDecision::Reject => write!(f, "reject"),
            RiskDecision::Throttle => write!(f, "throttle"),
        }
    }
}

/// Signal direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignalDirection {
    Long,
    Short,
    Neutral,
}

/// Kill switch state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KillSwitchState {
    Active,
    Triggered,
    Cooldown,
}

/// Slippage model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SlippageModel {
    /// Fixed slippage: base_bps only
    Fixed,
    /// Percentage/participation: slippage = base_bps + qty/depth * impact_bps
    Percentage,
    /// Square-root (Almgren-Chriss): slippage = sigma * sqrt(Q/V) * participation_rate
    SquareRoot,
}

impl fmt::Display for SlippageModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SlippageModel::Fixed => write!(f, "fixed"),
            SlippageModel::Percentage => write!(f, "percentage"),
            SlippageModel::SquareRoot => write!(f, "square_root"),
        }
    }
}

impl Default for SlippageModel {
    fn default() -> Self {
        SlippageModel::Percentage
    }
}

// ============================================================================
// Core Structs
// ============================================================================

/// Trading symbol (e.g., "BTC/USDT")
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol(String);

impl Symbol {
    pub fn new(s: impl Into<String>) -> Self {
        Symbol(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn base(&self) -> &str {
        self.0.split('/').next().unwrap_or(&self.0)
    }

    pub fn quote(&self) -> Option<&str> {
        self.0.split('/').nth(1)
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique order identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrderId(String);

impl OrderId {
    pub fn generate() -> Self {
        OrderId(uuid::Uuid::new_v4().to_string())
    }

    pub fn from_exchange(exchange_id: &str) -> Self {
        OrderId(exchange_id.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for OrderId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A trading order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: OrderId,
    pub client_order_id: Option<String>,
    pub symbol: Symbol,
    pub side: Side,
    pub order_type: OrderType,
    pub status: OrderStatus,
    pub time_in_force: TimeInForce,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub stop_price: Option<Decimal>,
    pub trailing_offset: Option<Decimal>,
    pub iceberg_qty: Option<Decimal>,
    pub filled_quantity: Decimal,
    pub average_fill_price: Decimal,
    pub exchange: ExchangeId,
    pub exchange_order_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub submitted_at: Option<DateTime<Utc>>,
    pub commission: Decimal,
    pub commission_asset: Option<String>,
    pub strategy_id: Option<String>,
    pub signal_id: Option<String>,
    pub tags: Vec<String>,
}

impl Order {
    pub fn new_market(symbol: Symbol, side: Side, quantity: Decimal, exchange: ExchangeId) -> Self {
        let now = Utc::now();
        Order {
            id: OrderId::generate(),
            client_order_id: None,
            symbol,
            side,
            order_type: OrderType::Market,
            status: OrderStatus::Created,
            time_in_force: TimeInForce::IOC,
            quantity,
            price: None,
            stop_price: None,
            trailing_offset: None,
            iceberg_qty: None,
            filled_quantity: Decimal::ZERO,
            average_fill_price: Decimal::ZERO,
            exchange,
            exchange_order_id: None,
            created_at: now,
            updated_at: now,
            commission: Decimal::ZERO,
            commission_asset: None,
            strategy_id: None,
            signal_id: None,
            tags: Vec::new(),
            submitted_at: None,
        }
    }

    pub fn new_limit(
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        time_in_force: TimeInForce,
        exchange: ExchangeId,
    ) -> Self {
        let now = Utc::now();
        Order {
            id: OrderId::generate(),
            client_order_id: None,
            symbol,
            side,
            order_type: OrderType::Limit,
            status: OrderStatus::Created,
            time_in_force,
            quantity,
            price: Some(price),
            stop_price: None,
            trailing_offset: None,
            iceberg_qty: None,
            filled_quantity: Decimal::ZERO,
            average_fill_price: Decimal::ZERO,
            exchange,
            exchange_order_id: None,
            created_at: now,
            updated_at: now,
            commission: Decimal::ZERO,
            commission_asset: None,
            strategy_id: None,
            signal_id: None,
            tags: Vec::new(),
            submitted_at: None,
        }
    }

    pub fn remaining_quantity(&self) -> Decimal {
        self.quantity - self.filled_quantity
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Created
                | OrderStatus::Validated
                | OrderStatus::Submitted
                | OrderStatus::PartiallyFilled
        )
    }

    pub fn notional_value(&self) -> Option<Decimal> {
        self.price.map(|p| p * self.quantity)
    }
}

/// A fill / trade execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: String,
    pub order_id: OrderId,
    pub symbol: Symbol,
    pub side: Side,
    pub quantity: Decimal,
    pub price: Decimal,
    pub commission: Decimal,
    pub commission_asset: String,
    pub exchange: ExchangeId,
    pub exchange_trade_id: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub is_maker: bool,
    pub slippage: Decimal,
}

/// Current position in a symbol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: Symbol,
    pub side: Side,
    pub quantity: Decimal,
    pub entry_price: Decimal,
    pub mark_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub liquidation_price: Option<Decimal>,
    pub leverage: Decimal,
    pub margin: Decimal,
    pub exchange: ExchangeId,
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Position {
    pub fn notional_value(&self) -> Decimal {
        self.quantity * self.mark_price
    }

    pub fn signed_quantity(&self) -> Decimal {
        match self.side {
            Side::Buy => self.quantity,
            Side::Sell => -self.quantity,
        }
    }

    pub fn update_mark_price(&mut self, price: Decimal) {
        self.mark_price = price;
        self.unrealized_pnl = (price - self.entry_price) * self.quantity
            * match self.side {
                Side::Buy => Decimal::ONE,
                Side::Sell => -Decimal::ONE,
            };
        self.updated_at = Utc::now();
    }
}

/// OHLCV Candle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub symbol: Symbol,
    pub exchange: ExchangeId,
    pub timeframe: Timeframe,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub open: Decimal,
    pub high: Decimal,
    pub low: Decimal,
    pub close: Decimal,
    pub volume: Decimal,
    pub quote_volume: Decimal,
    pub trades: u64,
    pub taker_buy_volume: Decimal,
    pub taker_buy_quote_volume: Decimal,
}

impl Candle {
    pub fn typical_price(&self) -> Decimal {
        (self.high + self.low + self.close) / rust_decimal_macros::dec!(3)
    }

    pub fn range(&self) -> Decimal {
        self.high - self.low
    }

    pub fn body(&self) -> Decimal {
        (self.close - self.open).abs()
    }

    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }

    pub fn upper_wick(&self) -> Decimal {
        self.high - self.open.max(self.close)
    }

    pub fn lower_wick(&self) -> Decimal {
        self.open.min(self.close) - self.low
    }
}

/// Candle with attached technical indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleWithIndicators {
    pub candle: Candle,
    pub vwap: Option<Decimal>,
    pub sma_20: Option<Decimal>,
    pub ema_12: Option<Decimal>,
    pub ema_26: Option<Decimal>,
    pub rsi_14: Option<Decimal>,
    pub bollinger_upper: Option<Decimal>,
    pub bollinger_lower: Option<Decimal>,
    pub atr_14: Option<Decimal>,
    pub volume_sma_20: Option<Decimal>,
}

/// Order book level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: Decimal,
    pub quantity: Decimal,
    pub order_count: u32,
}

/// L2 Order Book snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSnapshot {
    pub symbol: Symbol,
    pub exchange: ExchangeId,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp: DateTime<Utc>,
    pub sequence: u64,
}

/// Market tick
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tick {
    pub symbol: Symbol,
    pub exchange: ExchangeId,
    pub price: Decimal,
    pub quantity: Decimal,
    pub side: Side,
    pub timestamp: DateTime<Utc>,
    pub trade_id: String,
}

/// Trading signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signal {
    pub id: String,
    pub symbol: Symbol,
    pub direction: SignalDirection,
    pub strength: Decimal,
    pub strategy_id: String,
    pub indicators: Vec<(String, Decimal)>,
    pub timestamp: DateTime<Utc>,
    pub metadata: serde_json::Value,
}

/// Risk check result with details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckResult {
    pub decision: RiskDecision,
    pub check_name: String,
    pub reason: String,
    pub current_value: Decimal,
    pub limit_value: Decimal,
    pub timestamp: DateTime<Utc>,
}

/// Detailed risk check information for auditing/diagnostics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCheckDetail {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: Decimal,
    pub price: Option<Decimal>,
    pub checks: Vec<RiskCheckResult>,
    pub final_decision: RiskDecision,
    pub check_duration_us: u64,
    pub timestamp: DateTime<Utc>,
}

/// Portfolio snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioSnapshot {
    pub timestamp: DateTime<Utc>,
    pub total_value: Decimal,
    pub available_balance: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub positions: Vec<Position>,
    pub margin_used: Decimal,
    pub margin_available: Decimal,
    pub leverage: Decimal,
}

/// Execution report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionReport {
    pub order_id: OrderId,
    pub symbol: Symbol,
    pub side: Side,
    pub order_type: OrderType,
    pub status: OrderStatus,
    pub quantity: Decimal,
    pub filled_quantity: Decimal,
    pub average_price: Decimal,
    pub commission: Decimal,
    pub slippage: Decimal,
    pub latency_us: u64,
    pub exchange: ExchangeId,
    pub timestamp: DateTime<Utc>,
}

/// Exchange-specific symbol mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMapping {
    pub canonical: Symbol,
    pub binance: String,
    pub bybit: String,
    pub okx: String,
}

impl SymbolMapping {
    pub fn for_exchange(&self, exchange: ExchangeId) -> &str {
        match exchange {
            ExchangeId::Binance => &self.binance,
            ExchangeId::Bybit => &self.bybit,
            ExchangeId::OKX => &self.okx,
            ExchangeId::Paper => &self.binance,
        }
    }
}

/// Risk limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskLimits {
    pub max_position_per_symbol: Decimal,
    pub max_position_per_side: Decimal,
    pub max_total_position: Decimal,
    pub max_order_notional: Decimal,
    pub max_order_quantity: Decimal,
    pub max_orders_per_second: u32,
    pub max_orders_per_minute: u32,
    pub max_daily_drawdown: Decimal,
    pub max_weekly_drawdown: Decimal,
    pub max_drawdown: Decimal,
    pub max_net_exposure: Decimal,
    pub max_gross_exposure: Decimal,
    pub max_concentration_pct: Decimal,
    pub initial_margin_ratio: Decimal,
    pub maintenance_margin_ratio: Decimal,
}

impl Default for RiskLimits {
    fn default() -> Self {
        RiskLimits {
            max_position_per_symbol: rust_decimal_macros::dec!(100000),
            max_position_per_side: rust_decimal_macros::dec!(500000),
            max_total_position: rust_decimal_macros::dec!(1000000),
            max_order_notional: rust_decimal_macros::dec!(50000),
            max_order_quantity: rust_decimal_macros::dec!(10),
            max_orders_per_second: 10,
            max_orders_per_minute: 100,
            max_daily_drawdown: rust_decimal_macros::dec!(0.05),
            max_weekly_drawdown: rust_decimal_macros::dec!(0.10),
            max_drawdown: rust_decimal_macros::dec!(0.20),
            max_net_exposure: rust_decimal_macros::dec!(500000),
            max_gross_exposure: rust_decimal_macros::dec!(1000000),
            max_concentration_pct: rust_decimal_macros::dec!(0.25),
            initial_margin_ratio: rust_decimal_macros::dec!(0.10),
            maintenance_margin_ratio: rust_decimal_macros::dec!(0.05),
        }
    }
}

/// Exchange configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfig {
    pub exchange: ExchangeId,
    pub api_key: String,
    pub api_secret: String,
    pub passphrase: Option<String>,
    pub rest_url: String,
    pub ws_url: String,
    pub rate_limit_per_second: u32,
    pub maker_fee: Decimal,
    pub taker_fee: Decimal,
}

/// Circuit breaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub price_move_threshold_pct: Decimal,
    pub price_move_window_secs: u64,
    pub min_volume_threshold: Decimal,
    pub max_spread_pct: Decimal,
    pub max_realized_vol: Decimal,
    pub cooldown_secs: u64,
}

/// Slippage model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    /// Which slippage model to use
    pub model: SlippageModel,
    /// Base slippage as fraction of price (e.g., 0.0001 = 1bp)
    pub base_slippage_bps: Decimal,
    /// Slippage per unit of participation rate (order_qty / available_liquidity)
    pub participation_slope: Decimal,
    /// Maximum slippage as fraction of price
    pub max_slippage_pct: Decimal,
    /// Annualized volatility for square-root model
    pub sigma: Decimal,
    /// Average daily volume for square-root model
    pub avg_daily_volume: Decimal,
    /// Impact bps for percentage model: slippage = base_bps + qty/depth * impact_bps
    pub impact_bps: Decimal,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        SlippageConfig {
            model: SlippageModel::Percentage,
            base_slippage_bps: rust_decimal_macros::dec!(1),     // 1 basis point base
            participation_slope: rust_decimal_macros::dec!(0.1),  // 10% of participation rate
            max_slippage_pct: rust_decimal_macros::dec!(1),       // 1% max
            sigma: rust_decimal_macros::dec!(0.60),               // 60% annualized vol (typical crypto)
            avg_daily_volume: rust_decimal_macros::dec!(1000000), // 1M daily volume
            impact_bps: rust_decimal_macros::dec!(50),            // 50bps impact per unit of depth
        }
    }
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub default_exchange: ExchangeId,
    pub slippage_model: SlippageModel,
    pub slippage_config: SlippageConfig,
    pub max_retries: u32,
    pub retry_delay_ms: u64,
    pub enable_position_tracking: bool,
    pub enable_fee_tracking: bool,
    pub max_open_orders: usize,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        ExecutionConfig {
            default_exchange: ExchangeId::Paper,
            slippage_model: SlippageModel::Percentage,
            slippage_config: SlippageConfig::default(),
            max_retries: 3,
            retry_delay_ms: 100,
            enable_position_tracking: true,
            enable_fee_tracking: true,
            max_open_orders: 1000,
        }
    }
}

/// Convert symbol to exchange-specific format for WebSocket streams
pub fn symbol_to_ws_symbol(symbol: &Symbol, exchange: ExchangeId) -> String {
    match exchange {
        ExchangeId::Binance => {
            format!("{}{}", symbol.base().to_lowercase(), symbol.quote().unwrap_or("usdt").to_lowercase())
        }
        ExchangeId::Bybit => {
            format!("{}{}", symbol.base(), symbol.quote().unwrap_or("USDT"))
        }
        ExchangeId::OKX => {
            format!("{}-{}", symbol.base(), symbol.quote().unwrap_or("USDT"))
        }
        ExchangeId::Paper => {
            format!("{}{}", symbol.base().to_lowercase(), symbol.quote().unwrap_or("usdt").to_lowercase())
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_symbol_base_quote() {
        let sym = Symbol::new("BTC/USDT");
        assert_eq!(sym.base(), "BTC");
        assert_eq!(sym.quote(), Some("USDT"));
    }

    #[test]
    fn test_symbol_no_slash() {
        let sym = Symbol::new("BTC");
        assert_eq!(sym.base(), "BTC");
        assert_eq!(sym.quote(), None);
    }

    #[test]
    fn test_order_id_generate() {
        let id1 = OrderId::generate();
        let id2 = OrderId::generate();
        assert_ne!(id1.as_str(), id2.as_str());
    }

    #[test]
    fn test_order_remaining_quantity() {
        let order = Order {
            filled_quantity: dec!(3),
            quantity: dec!(10),
            ..Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(10), ExchangeId::Paper)
        };
        assert_eq!(order.remaining_quantity(), dec!(7));
    }

    #[test]
    fn test_order_is_active() {
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper);
        assert!(order.is_active());
    }

    #[test]
    fn test_order_notional_value() {
        let order = Order::new_limit(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(2),
            dec!(50000),
            TimeInForce::GTC,
            ExchangeId::Paper,
        );
        assert_eq!(order.notional_value(), Some(dec!(100000)));
    }

    #[test]
    fn test_position_update_mark_price() {
        let mut pos = Position {
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Buy,
            quantity: dec!(1),
            entry_price: dec!(50000),
            mark_price: dec!(50000),
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            liquidation_price: None,
            leverage: Decimal::ONE,
            margin: dec!(50000),
            exchange: ExchangeId::Paper,
            opened_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        pos.update_mark_price(dec!(51000));
        assert_eq!(pos.unrealized_pnl, dec!(1000));
    }

    #[test]
    fn test_position_signed_quantity() {
        let mut pos = Position {
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Sell,
            quantity: dec!(1),
            entry_price: dec!(50000),
            mark_price: dec!(50000),
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            liquidation_price: None,
            leverage: Decimal::ONE,
            margin: dec!(50000),
            exchange: ExchangeId::Paper,
            opened_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        assert_eq!(pos.signed_quantity(), dec!(-1));
        pos.side = Side::Buy;
        assert_eq!(pos.signed_quantity(), dec!(1));
    }

    #[test]
    fn test_candle_methods() {
        let now = chrono::Utc::now();
        let candle = Candle {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Paper,
            timeframe: Timeframe::M1,
            open_time: now,
            close_time: now + chrono::Duration::seconds(60),
            open: dec!(100),
            high: dec!(110),
            low: dec!(95),
            close: dec!(105),
            volume: dec!(1000),
            quote_volume: dec!(100000),
            trades: 500,
            taker_buy_volume: dec!(500),
            taker_buy_quote_volume: dec!(50000),
        };
        assert!(candle.is_bullish());
        assert_eq!(candle.range(), dec!(15));
        assert_eq!(candle.body(), dec!(5));
        assert_eq!(candle.upper_wick(), dec!(5));
        assert_eq!(candle.lower_wick(), dec!(5));
    }

    #[test]
    fn test_timeframe_duration() {
        assert_eq!(Timeframe::M1.duration_secs(), 60);
        assert_eq!(Timeframe::H1.duration_secs(), 3600);
        assert_eq!(Timeframe::D1.duration_secs(), 86400);
    }

    #[test]
    fn test_exchange_id_display() {
        assert_eq!(ExchangeId::Binance.to_string(), "binance");
        assert_eq!(ExchangeId::OKX.to_string(), "okx");
    }

    #[test]
    fn test_side_display() {
        assert_eq!(Side::Buy.to_string(), "buy");
        assert_eq!(Side::Sell.to_string(), "sell");
    }

    #[test]
    fn test_risk_decision_display() {
        assert_eq!(RiskDecision::Allow.to_string(), "allow");
        assert_eq!(RiskDecision::Reject.to_string(), "reject");
    }

    #[test]
    fn test_symbol_mapping() {
        let mapping = SymbolMapping {
            canonical: Symbol::new("BTC/USDT"),
            binance: "BTCUSDT".to_string(),
            bybit: "BTCUSDT".to_string(),
            okx: "BTC-USDT".to_string(),
        };
        assert_eq!(mapping.for_exchange(ExchangeId::Binance), "BTCUSDT");
        assert_eq!(mapping.for_exchange(ExchangeId::OKX), "BTC-USDT");
    }

    #[test]
    fn test_risk_limits_default() {
        let limits = RiskLimits::default();
        assert!(limits.max_position_per_symbol > Decimal::ZERO);
        assert!(limits.max_orders_per_second > 0);
    }

    #[test]
    fn test_slippage_config_default() {
        let config = SlippageConfig::default();
        assert!(config.base_slippage_bps > Decimal::ZERO);
        assert!(config.participation_slope > Decimal::ZERO);
        assert!(config.max_slippage_pct > Decimal::ZERO);
        assert_eq!(config.model, SlippageModel::Percentage);
    }

    #[test]
    fn test_slippage_model_display() {
        assert_eq!(SlippageModel::Fixed.to_string(), "fixed");
        assert_eq!(SlippageModel::Percentage.to_string(), "percentage");
        assert_eq!(SlippageModel::SquareRoot.to_string(), "square_root");
    }

    #[test]
    fn test_slippage_model_default() {
        assert_eq!(SlippageModel::default(), SlippageModel::Percentage);
    }

    #[test]
    fn test_execution_config_default() {
        let config = ExecutionConfig::default();
        assert_eq!(config.default_exchange, ExchangeId::Paper);
        assert_eq!(config.max_retries, 3);
        assert!(config.enable_position_tracking);
    }

    #[test]
    fn test_candle_with_indicators() {
        let now = chrono::Utc::now();
        let candle = Candle {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Paper,
            timeframe: Timeframe::M1,
            open_time: now,
            close_time: now + chrono::Duration::seconds(60),
            open: dec!(100),
            high: dec!(110),
            low: dec!(95),
            close: dec!(105),
            volume: dec!(1000),
            quote_volume: dec!(100000),
            trades: 500,
            taker_buy_volume: dec!(500),
            taker_buy_quote_volume: dec!(50000),
        };
        let cwi = CandleWithIndicators {
            candle,
            vwap: Some(dec!(103)),
            sma_20: Some(dec!(102)),
            ema_12: Some(dec!(104)),
            ema_26: Some(dec!(101)),
            rsi_14: Some(dec!(55)),
            bollinger_upper: Some(dec!(115)),
            bollinger_lower: Some(dec!(90)),
            atr_14: Some(dec!(8)),
            volume_sma_20: Some(dec!(950)),
        };
        assert_eq!(cwi.vwap, Some(dec!(103)));
        assert_eq!(cwi.rsi_14, Some(dec!(55)));
    }

    #[test]
    fn test_risk_check_detail() {
        let detail = RiskCheckDetail {
            order_id: "test-123".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: "buy".to_string(),
            quantity: dec!(1),
            price: Some(dec!(50000)),
            checks: vec![],
            final_decision: RiskDecision::Allow,
            check_duration_us: 50,
            timestamp: chrono::Utc::now(),
        };
        assert_eq!(detail.final_decision, RiskDecision::Allow);
        assert_eq!(detail.check_duration_us, 50);
    }

    #[test]
    fn test_position_short_unrealized_pnl() {
        let mut pos = Position {
            symbol: Symbol::new("BTC/USDT"),
            side: Side::Sell,
            quantity: dec!(1),
            entry_price: dec!(50000),
            mark_price: dec!(50000),
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            liquidation_price: None,
            leverage: Decimal::ONE,
            margin: dec!(50000),
            exchange: ExchangeId::Paper,
            opened_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };
        pos.update_mark_price(dec!(49000));
        assert_eq!(pos.unrealized_pnl, dec!(1000)); // short profits when price drops
    }

    #[test]
    fn test_order_status_transitions() {
        let order = Order::new_market(Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper);
        assert_eq!(order.status, OrderStatus::Created);
        assert!(order.is_active());
    }

    #[test]
    fn test_candle_typical_price() {
        let now = chrono::Utc::now();
        let candle = Candle {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Paper,
            timeframe: Timeframe::M1,
            open_time: now,
            close_time: now + chrono::Duration::seconds(60),
            open: dec!(100),
            high: dec!(110),
            low: dec!(90),
            close: dec!(100),
            volume: dec!(1000),
            quote_volume: dec!(100000),
            trades: 500,
            taker_buy_volume: dec!(500),
            taker_buy_quote_volume: dec!(50000),
        };
        // (110 + 90 + 100) / 3 = 100
        assert_eq!(candle.typical_price(), dec!(100));
    }
}
