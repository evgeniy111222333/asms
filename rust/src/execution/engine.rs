//! Execution Engine - Full order lifecycle management
//!
//! Handles order creation, validation, submission, fill tracking,
//! position management, and execution reporting with multiple
//! realistic slippage models: Percentage, SquareRoot (Almgren-Chriss), Fixed.

use crate::core::orderbook::OrderBook;
use crate::core::types::*;
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;

/// Fill record for partial fill tracking
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FillRecord {
    pub trade_id: String,
    pub order_id: OrderId,
    pub price: Decimal,
    pub quantity: Decimal,
    pub commission: Decimal,
    pub timestamp: chrono::DateTime<Utc>,
    pub is_maker: bool,
}

/// Position tracker per symbol
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PositionTracker {
    pub positions: HashMap<Symbol, Position>,
}

impl PositionTracker {
    pub fn new() -> Self {
        PositionTracker {
            positions: HashMap::new(),
        }
    }

    pub fn apply_fill(&mut self, fill: &FillRecord, symbol: &Symbol, side: Side, exchange: ExchangeId) {
        let position = self.positions.entry(symbol.clone()).or_insert_with(|| Position {
            symbol: symbol.clone(),
            side: if side == Side::Buy { Side::Buy } else { Side::Sell },
            quantity: Decimal::ZERO,
            entry_price: Decimal::ZERO,
            mark_price: fill.price,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            liquidation_price: None,
            leverage: Decimal::ONE,
            margin: Decimal::ZERO,
            exchange,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
        });

        let signed_qty = match side {
            Side::Buy => fill.quantity,
            Side::Sell => -fill.quantity,
        };

        let current_signed = match position.side {
            Side::Buy => position.quantity,
            Side::Sell => -position.quantity,
        };

        let new_signed = current_signed + signed_qty;

        if new_signed == Decimal::ZERO {
            // Position closed
            let pnl_sign = if position.side == Side::Buy { Decimal::ONE } else { -Decimal::ONE };
            position.realized_pnl += (fill.price - position.entry_price) * position.quantity * pnl_sign;
            position.quantity = Decimal::ZERO;
            position.entry_price = Decimal::ZERO;
        } else if (new_signed > Decimal::ZERO) != (current_signed > Decimal::ZERO) && current_signed != Decimal::ZERO {
            // Position flipped - realize PnL on old position
            let closing_qty = position.quantity.min(fill.quantity);
            let pnl_sign = if position.side == Side::Buy { Decimal::ONE } else { -Decimal::ONE };
            position.realized_pnl += (fill.price - position.entry_price) * closing_qty * pnl_sign;
            position.quantity = new_signed.abs();
            position.entry_price = fill.price;
            position.side = if new_signed > Decimal::ZERO { Side::Buy } else { Side::Sell };
        } else if new_signed.abs() < current_signed.abs() {
            // Position reduced - realize PnL on reduced portion, keep entry price same
            let pnl_sign = if position.side == Side::Buy { Decimal::ONE } else { -Decimal::ONE };
            position.realized_pnl += (fill.price - position.entry_price) * fill.quantity * pnl_sign;
            position.quantity = new_signed.abs();
        } else {
            // Adding to position - update entry price (weighted average)
            let old_abs = position.quantity;
            let new_abs = new_signed.abs();
            let total_cost = position.entry_price * old_abs + fill.price * fill.quantity;
            let total_qty = old_abs + fill.quantity;
            position.entry_price = total_cost / total_qty;
            position.quantity = new_abs;
            position.side = if new_signed > Decimal::ZERO { Side::Buy } else { Side::Sell };
        }

        position.mark_price = fill.price;
        position.updated_at = Utc::now();
    }
}

/// Execution engine with full order lifecycle and realistic slippage
pub struct ExecutionEngine {
    /// Active orders by ID
    orders: Arc<DashMap<OrderId, Order>>,
    /// Fill history
    fills: Arc<RwLock<Vec<FillRecord>>>,
    /// Position tracker
    positions: Arc<RwLock<PositionTracker>>,
    /// Execution reports
    reports: Arc<RwLock<Vec<ExecutionReport>>>,
    /// Exchange configurations
    exchange_configs: Arc<HashMap<ExchangeId, ExchangeConfig>>,
    /// Rate limiters per exchange
    order_timestamps: Arc<DashMap<ExchangeId, Vec<chrono::DateTime<Utc>>>>,
    /// Order books per symbol for slippage computation
    order_books: Arc<DashMap<Symbol, Arc<parking_lot::RwLock<OrderBook>>>>,
    /// Slippage configuration (RwLock for runtime mutation)
    slippage_config: Arc<parking_lot::RwLock<SlippageConfig>>,
    /// Open orders list for get_open_orders
    open_order_ids: Arc<RwLock<Vec<OrderId>>>,
}

impl ExecutionEngine {
    pub fn new(exchange_configs: HashMap<ExchangeId, ExchangeConfig>) -> Self {
        ExecutionEngine {
            orders: Arc::new(DashMap::new()),
            fills: Arc::new(RwLock::new(Vec::new())),
            positions: Arc::new(RwLock::new(PositionTracker::new())),
            reports: Arc::new(RwLock::new(Vec::new())),
            exchange_configs: Arc::new(exchange_configs),
            order_timestamps: Arc::new(DashMap::new()),
            order_books: Arc::new(DashMap::new()),
            slippage_config: Arc::new(parking_lot::RwLock::new(SlippageConfig::default())),
            open_order_ids: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create engine with custom slippage config
    pub fn with_slippage_config(self, config: SlippageConfig) -> Self {
        *self.slippage_config.write() = config;
        self
    }

    /// Set the slippage model at runtime
    pub fn set_slippage_model(&self, model: SlippageModel) {
        self.slippage_config.write().model = model;
    }

    /// Get the current slippage config
    pub fn get_slippage_config(&self) -> SlippageConfig {
        self.slippage_config.read().clone()
    }

    /// Register an order book for a symbol (for slippage computation)
    pub fn register_order_book(&self, symbol: Symbol, order_book: Arc<parking_lot::RwLock<OrderBook>>) {
        self.order_books.insert(symbol, order_book);
    }

    /// Create a new market order
    pub fn create_order(
        &self,
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        exchange: ExchangeId,
    ) -> Order {
        Order::new_market(symbol, side, quantity, exchange)
    }

    /// Create a new market order
    pub fn create_market_order(
        &self,
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        exchange: ExchangeId,
    ) -> Order {
        Order::new_market(symbol, side, quantity, exchange)
    }

    /// Create a new limit order
    pub fn create_limit_order(
        &self,
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        time_in_force: TimeInForce,
        exchange: ExchangeId,
    ) -> Order {
        Order::new_limit(symbol, side, quantity, price, time_in_force, exchange)
    }

    /// Create a stop order
    pub fn create_stop_order(
        &self,
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        stop_price: Decimal,
        exchange: ExchangeId,
    ) -> Order {
        let mut order = Order::new_market(symbol, side, quantity, exchange);
        order.order_type = OrderType::Stop;
        order.stop_price = Some(stop_price);
        order
    }

    /// Create a trailing stop order
    pub fn create_trailing_stop_order(
        &self,
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        trailing_offset: Decimal,
        exchange: ExchangeId,
    ) -> Order {
        let mut order = Order::new_market(symbol, side, quantity, exchange);
        order.order_type = OrderType::TrailingStop;
        order.trailing_offset = Some(trailing_offset);
        order
    }

    /// Create an iceberg order
    pub fn create_iceberg_order(
        &self,
        symbol: Symbol,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        visible_qty: Decimal,
        exchange: ExchangeId,
    ) -> Order {
        let mut order = Order::new_limit(symbol, side, quantity, price, TimeInForce::GTC, exchange);
        order.order_type = OrderType::Iceberg;
        order.iceberg_qty = Some(visible_qty);
        order
    }

    /// Submit an order (mark as submitted)
    pub fn submit_order(&self, order: &mut Order) -> Result<(), String> {
        if !matches!(order.status, OrderStatus::Created | OrderStatus::Validated) {
            return Err(format!("Cannot submit order in state {:?}", order.status));
        }
        order.status = OrderStatus::Submitted;
        order.updated_at = Utc::now();
        order.submitted_at = Some(Utc::now());
        self.orders.insert(order.id.clone(), order.clone());
        self.open_order_ids.write().push(order.id.clone());
        Ok(())
    }

    /// Compute slippage based on configured model and order book depth
    pub fn compute_slippage(&self, order: &Order, fill_price: Decimal) -> Decimal {
        if fill_price == Decimal::ZERO {
            return Decimal::ZERO;
        }

        let config = self.slippage_config.read().clone();

        match config.model {
            SlippageModel::Fixed => {
                Self::compute_fixed_slippage(&config, order, fill_price)
            }
            SlippageModel::Percentage => {
                Self::compute_percentage_slippage(&config, order, fill_price, &self.order_books)
            }
            SlippageModel::SquareRoot => {
                Self::compute_sqrt_slippage(&config, order, fill_price, &self.order_books)
            }
        }
    }

    /// Fixed slippage: base_bps of fill price
    fn compute_fixed_slippage(config: &SlippageConfig, order: &Order, fill_price: Decimal) -> Decimal {
        let base_bps = config.base_slippage_bps / rust_decimal_macros::dec!(10000);
        match order.order_type {
            OrderType::Market => fill_price * base_bps,
            OrderType::Stop | OrderType::TrailingStop => fill_price * base_bps * rust_decimal_macros::dec!(2),
            _ => Decimal::ZERO,
        }
    }

    /// Percentage slippage: slippage = base_bps + quantity / depth * impact_bps
    fn compute_percentage_slippage(
        config: &SlippageConfig,
        order: &Order,
        fill_price: Decimal,
        order_books: &DashMap<Symbol, Arc<parking_lot::RwLock<OrderBook>>>,
    ) -> Decimal {
        // Try to compute from real order book data
        if let Some(ob_arc) = order_books.get(&order.symbol) {
            let ob = ob_arc.read();
            let levels: Vec<&OrderBookLevel> = match order.side {
                Side::Buy => ob.get_asks(20),
                Side::Sell => ob.get_bids(20),
            };

            let best_price = match order.side {
                Side::Buy => ob.best_ask(),
                Side::Sell => ob.best_bid(),
            };

            if let Some(best) = best_price {
                // Available liquidity at top levels
                let available_liquidity: Decimal = levels.iter().map(|l| l.quantity * l.price).sum();

                if available_liquidity > Decimal::ZERO {
                    let order_notional = order.quantity * fill_price;
                    let participation_rate = order_notional / available_liquidity;

                    // slippage = base_bps + participation_rate * impact_bps
                    let base_slippage = config.base_slippage_bps
                        / rust_decimal_macros::dec!(10000);
                    let participation_slippage = participation_rate
                        * config.impact_bps / rust_decimal_macros::dec!(10000);
                    let total_slippage_pct = (base_slippage + participation_slippage)
                        .min(config.max_slippage_pct / rust_decimal_macros::dec!(100));

                    return match order.side {
                        Side::Buy => fill_price - best,
                        Side::Sell => best - fill_price,
                    }.abs().max(fill_price * total_slippage_pct);
                }
            }
        }

        // Fallback: simple participation-based model
        let base_bps = config.base_slippage_bps / rust_decimal_macros::dec!(10000);
        let impact = config.impact_bps / rust_decimal_macros::dec!(10000);
        match order.order_type {
            OrderType::Market => fill_price * (base_bps + impact),
            OrderType::Stop | OrderType::TrailingStop => fill_price * (base_bps + impact) * rust_decimal_macros::dec!(2),
            _ => fill_price * base_bps,
        }
    }

    /// Square-root slippage (Almgren-Chriss): slippage = sigma * sqrt(Q / V) * participation_rate
    fn compute_sqrt_slippage(
        config: &SlippageConfig,
        order: &Order,
        fill_price: Decimal,
        order_books: &DashMap<Symbol, Arc<parking_lot::RwLock<OrderBook>>>,
    ) -> Decimal {
        let sigma = config.sigma;
        let daily_vol = config.avg_daily_volume;

        // Q = order quantity in notional terms
        let q = order.quantity * fill_price;

        // V = average daily volume; compute participation
        let participation = if daily_vol > Decimal::ZERO {
            q / daily_vol
        } else {
            rust_decimal_macros::dec!(0.01)
        };

        // Use actual order book depth if available
        let depth_participation = if let Some(ob_arc) = order_books.get(&order.symbol) {
            let ob = ob_arc.read();
            let levels: Vec<&OrderBookLevel> = match order.side {
                Side::Buy => ob.get_asks(20),
                Side::Sell => ob.get_bids(20),
            };
            let available: Decimal = levels.iter().map(|l| l.quantity * l.price).sum();
            if available > Decimal::ZERO { q / available } else { participation }
        } else {
            participation
        };

        // sigma * sqrt(Q/V) * participation_rate
        // sqrt is computed via f64 since rust_decimal doesn't have sqrt
        let q_over_v = depth_participation.to_string().parse::<f64>().unwrap_or(0.01);
        let sqrt_part = q_over_v.sqrt();
        let slippage_fraction = sigma
            * Decimal::from_f64_retain(sqrt_part).unwrap_or(Decimal::ZERO)
            * depth_participation.min(rust_decimal_macros::dec!(1));

        let total_pct = slippage_fraction
            .min(config.max_slippage_pct / rust_decimal_macros::dec!(100))
            .max(config.base_slippage_bps / rust_decimal_macros::dec!(10000));

        fill_price * total_pct
    }

    /// Process a fill for an order
    pub fn process_fill(
        &self,
        order_id: &OrderId,
        fill_price: Decimal,
        fill_quantity: Decimal,
        commission: Decimal,
        is_maker: bool,
    ) -> Result<ExecutionReport, String> {
        let mut order = self
            .orders
            .get_mut(order_id)
            .ok_or_else(|| format!("Order {} not found", order_id.as_str()))?;

        let start = order.submitted_at.unwrap_or(order.created_at);
        let now = Utc::now();

        // Compute slippage before mutating order
        let slippage = self.compute_slippage(&order.clone(), fill_price);

        // Update fill tracking
        let old_filled = order.filled_quantity;
        order.filled_quantity += fill_quantity;
        if order.filled_quantity > order.quantity {
            order.filled_quantity = order.quantity;
        }

        // Update average fill price
        if order.average_fill_price == Decimal::ZERO {
            order.average_fill_price = fill_price;
        } else {
            order.average_fill_price = (order.average_fill_price * old_filled + fill_price * fill_quantity)
                / order.filled_quantity;
        }

        order.commission += commission;
        order.updated_at = now;

        // Update status
        if order.filled_quantity >= order.quantity {
            order.status = OrderStatus::Filled;
        } else {
            order.status = OrderStatus::PartiallyFilled;
        }

        // Create fill record
        let fill = FillRecord {
            trade_id: uuid::Uuid::new_v4().to_string(),
            order_id: order_id.clone(),
            price: fill_price,
            quantity: fill_quantity,
            commission,
            timestamp: now,
            is_maker,
        };

        // Update positions
        self.positions
            .write()
            .apply_fill(&fill, &order.symbol, order.side, order.exchange);

        // Save fill
        self.fills.write().push(fill);

        let latency_us = (now - start).num_microseconds().unwrap_or(0) as u64;

        let report = ExecutionReport {
            order_id: order_id.clone(),
            symbol: order.symbol.clone(),
            side: order.side,
            order_type: order.order_type,
            status: order.status,
            quantity: order.quantity,
            filled_quantity: order.filled_quantity,
            average_price: order.average_fill_price,
            commission: order.commission,
            slippage,
            latency_us,
            exchange: order.exchange,
            timestamp: now,
        };

        // Remove from open orders if filled
        if order.status == OrderStatus::Filled || order.status == OrderStatus::Cancelled {
            self.open_order_ids.write().retain(|id| id != order_id);
        }

        self.reports.write().push(report.clone());
        Ok(report)
    }

    /// Cancel an order
    pub fn cancel_order(&self, order_id: &OrderId) -> Result<Order, String> {
        let mut order = self
            .orders
            .get_mut(order_id)
            .ok_or_else(|| format!("Order {} not found", order_id.as_str()))?;

        if !order.is_active() {
            return Err(format!("Cannot cancel order in state {:?}", order.status));
        }

        order.status = OrderStatus::Cancelled;
        order.updated_at = Utc::now();
        self.open_order_ids.write().retain(|id| id != order_id);
        Ok(order.clone())
    }

    /// Get order by ID
    pub fn get_order(&self, order_id: &OrderId) -> Option<Order> {
        self.orders.get(order_id).map(|r| r.value().clone())
    }

    /// Get all active (open) orders
    pub fn get_open_orders(&self) -> Vec<Order> {
        let open_ids = self.open_order_ids.read().clone();
        open_ids.iter()
            .filter_map(|id| self.orders.get(id).map(|r| r.value().clone()))
            .filter(|o| o.is_active())
            .collect()
    }

    /// Get all active orders (alias)
    pub fn active_orders(&self) -> Vec<Order> {
        self.get_open_orders()
    }

    /// Get orders by symbol
    pub fn orders_by_symbol(&self, symbol: &Symbol) -> Vec<Order> {
        self.orders
            .iter()
            .filter(|r| &r.value().symbol == symbol)
            .map(|r| r.value().clone())
            .collect()
    }

    /// Get all positions
    pub fn get_positions(&self) -> HashMap<Symbol, Position> {
        self.positions.read().positions.clone()
    }

    /// Get execution reports
    pub fn get_reports(&self) -> Vec<ExecutionReport> {
        self.reports.read().clone()
    }

    /// Compute total unrealized PnL
    pub fn total_unrealized_pnl(&self) -> Decimal {
        self.positions
            .read()
            .positions
            .values()
            .map(|p| p.unrealized_pnl)
            .sum()
    }

    /// Compute total realized PnL
    pub fn total_realized_pnl(&self) -> Decimal {
        self.positions
            .read()
            .positions
            .values()
            .map(|p| p.realized_pnl)
            .sum()
    }

    /// Compute fee for an order on a given exchange
    pub fn compute_fee(&self, exchange: ExchangeId, notional: Decimal, is_maker: bool) -> Decimal {
        if let Some(config) = self.exchange_configs.get(&exchange) {
            let fee_rate = if is_maker { config.maker_fee } else { config.taker_fee };
            notional * fee_rate
        } else {
            Decimal::ZERO
        }
    }

    /// Check rate limit for an exchange
    pub fn check_rate_limit(&self, exchange: ExchangeId) -> bool {
        let now = Utc::now();
        let max_per_second = self
            .exchange_configs
            .get(&exchange)
            .map(|c| c.rate_limit_per_second)
            .unwrap_or(10);

        if let Some(mut timestamps) = self.order_timestamps.get_mut(&exchange) {
            let one_sec_ago = now - chrono::Duration::seconds(1);
            timestamps.retain(|t| *t > one_sec_ago);
            timestamps.len() < max_per_second as usize
        } else {
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_order_lifecycle() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            ExchangeId::Paper,
        );

        assert_eq!(order.status, OrderStatus::Created);
        engine.submit_order(&mut order).unwrap();
        assert_eq!(order.status, OrderStatus::Submitted);

        let report = engine
            .process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false)
            .unwrap();
        assert_eq!(report.status, OrderStatus::Filled);
        assert_eq!(report.average_price, dec!(50000));
    }

    #[test]
    fn test_partial_fill() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_limit_order(
            Symbol::new("ETH/USDT"),
            Side::Buy,
            dec!(10),
            dec!(3000),
            TimeInForce::GTC,
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();

        let r1 = engine
            .process_fill(&order.id, dec!(3000), dec!(5), dec!(1.5), true)
            .unwrap();
        assert_eq!(r1.status, OrderStatus::PartiallyFilled);

        let r2 = engine
            .process_fill(&order.id, dec!(3001), dec!(5), dec!(1.5), false)
            .unwrap();
        assert_eq!(r2.status, OrderStatus::Filled);
    }

    #[test]
    fn test_cancel_order() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_limit_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            dec!(50000),
            TimeInForce::GTC,
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        let cancelled = engine.cancel_order(&order.id).unwrap();
        assert_eq!(cancelled.status, OrderStatus::Cancelled);
    }

    #[test]
    fn test_cancel_filled_order_fails() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        engine.process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false).unwrap();
        let result = engine.cancel_order(&order.id);
        assert!(result.is_err());
    }

    #[test]
    fn test_submit_wrong_state_fails() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        let result = engine.submit_order(&mut order);
        assert!(result.is_err());
    }

    #[test]
    fn test_slippage_with_order_book() {
        let engine = ExecutionEngine::new(HashMap::new());

        let ob = Arc::new(parking_lot::RwLock::new(OrderBook::new(
            Symbol::new("BTC/USDT"),
            ExchangeId::Binance,
            25,
        )));
        {
            let mut ob = ob.write();
            ob.update_ask(dec!(50000), dec!(5), 1);
            ob.update_ask(dec!(50001), dec!(10), 1);
            ob.update_ask(dec!(50002), dec!(20), 1);
            ob.update_bid(dec!(49999), dec!(5), 1);
        }
        engine.register_order_book(Symbol::new("BTC/USDT"), ob);

        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            ExchangeId::Binance,
        );
        engine.submit_order(&mut order).unwrap();

        let report = engine
            .process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false)
            .unwrap();
        assert!(report.slippage >= Decimal::ZERO);
    }

    #[test]
    fn test_slippage_without_order_book() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();

        let report = engine
            .process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false)
            .unwrap();
        assert!(report.slippage > Decimal::ZERO);
    }

    #[test]
    fn test_position_tracking() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        engine.process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false).unwrap();

        let positions = engine.get_positions();
        assert!(positions.contains_key(&Symbol::new("BTC/USDT")));
    }

    #[test]
    fn test_get_positions() {
        let engine = ExecutionEngine::new(HashMap::new());
        let positions = engine.get_positions();
        assert!(positions.is_empty());
    }

    #[test]
    fn test_create_stop_order() {
        let engine = ExecutionEngine::new(HashMap::new());
        let order = engine.create_stop_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            dec!(51000),
            ExchangeId::Paper,
        );
        assert_eq!(order.order_type, OrderType::Stop);
        assert_eq!(order.stop_price, Some(dec!(51000)));
    }

    #[test]
    fn test_create_trailing_stop_order() {
        let engine = ExecutionEngine::new(HashMap::new());
        let order = engine.create_trailing_stop_order(
            Symbol::new("BTC/USDT"),
            Side::Sell,
            dec!(1),
            dec!(500),
            ExchangeId::Paper,
        );
        assert_eq!(order.order_type, OrderType::TrailingStop);
        assert_eq!(order.trailing_offset, Some(dec!(500)));
    }

    #[test]
    fn test_create_iceberg_order() {
        let engine = ExecutionEngine::new(HashMap::new());
        let order = engine.create_iceberg_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(10),
            dec!(50000),
            dec!(1),
            ExchangeId::Paper,
        );
        assert_eq!(order.order_type, OrderType::Iceberg);
        assert_eq!(order.iceberg_qty, Some(dec!(1)));
    }

    #[test]
    fn test_process_fill_nonexistent_order() {
        let engine = ExecutionEngine::new(HashMap::new());
        let result = engine.process_fill(&OrderId::generate(), dec!(50000), dec!(1), dec!(5), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_position_flipping() {
        let engine = ExecutionEngine::new(HashMap::new());

        // Buy 1 BTC
        let mut order1 = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Buy,
            dec!(1),
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order1).unwrap();
        engine.process_fill(&order1.id, dec!(50000), dec!(1), dec!(5), false).unwrap();

        // Sell 2 BTC (flip to short)
        let mut order2 = engine.create_market_order(
            Symbol::new("BTC/USDT"),
            Side::Sell,
            dec!(2),
            ExchangeId::Paper,
        );
        engine.submit_order(&mut order2).unwrap();
        engine.process_fill(&order2.id, dec!(51000), dec!(2), dec!(5), false).unwrap();

        let positions = engine.get_positions();
        let pos = positions.get(&Symbol::new("BTC/USDT")).unwrap();
        assert_eq!(pos.side, Side::Sell);
        assert_eq!(pos.quantity, dec!(1));
    }

    #[test]
    fn test_fee_computation() {
        let mut configs = HashMap::new();
        configs.insert(ExchangeId::Binance, ExchangeConfig {
            exchange: ExchangeId::Binance,
            api_key: String::new(),
            api_secret: String::new(),
            passphrase: None,
            rest_url: String::new(),
            ws_url: String::new(),
            rate_limit_per_second: 10,
            maker_fee: dec!(0.001),
            taker_fee: dec!(0.001),
        });
        let engine = ExecutionEngine::new(configs);

        let taker_fee = engine.compute_fee(ExchangeId::Binance, dec!(50000), false);
        assert_eq!(taker_fee, dec!(50));

        let maker_fee = engine.compute_fee(ExchangeId::Binance, dec!(50000), true);
        assert_eq!(maker_fee, dec!(50));
    }

    #[test]
    fn test_fee_unknown_exchange() {
        let engine = ExecutionEngine::new(HashMap::new());
        let fee = engine.compute_fee(ExchangeId::Binance, dec!(50000), false);
        assert_eq!(fee, Decimal::ZERO);
    }

    #[test]
    fn test_get_open_orders() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order1 = engine.create_limit_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(1), dec!(50000),
            TimeInForce::GTC, ExchangeId::Paper,
        );
        let mut order2 = engine.create_limit_order(
            Symbol::new("ETH/USDT"), Side::Buy, dec!(10), dec!(3000),
            TimeInForce::GTC, ExchangeId::Paper,
        );
        engine.submit_order(&mut order1).unwrap();
        engine.submit_order(&mut order2).unwrap();

        let open = engine.get_open_orders();
        assert_eq!(open.len(), 2);

        // Fill one order
        engine.process_fill(&order1.id, dec!(50000), dec!(1), dec!(5), false).unwrap();
        let open = engine.get_open_orders();
        assert_eq!(open.len(), 1);
    }

    #[test]
    fn test_fixed_slippage_model() {
        let config = SlippageConfig {
            model: SlippageModel::Fixed,
            base_slippage_bps: dec!(5),
            ..SlippageConfig::default()
        };
        let engine = ExecutionEngine::new(HashMap::new()).with_slippage_config(config);

        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        let report = engine.process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false).unwrap();
        // Fixed: 50000 * 5/10000 = 25
        assert!(report.slippage > Decimal::ZERO);
    }

    #[test]
    fn test_sqrt_slippage_model() {
        let config = SlippageConfig {
            model: SlippageModel::SquareRoot,
            sigma: dec!(0.6),
            avg_daily_volume: dec!(1000000),
            ..SlippageConfig::default()
        };
        let engine = ExecutionEngine::new(HashMap::new()).with_slippage_config(config);

        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        let report = engine.process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false).unwrap();
        assert!(report.slippage > Decimal::ZERO);
    }

    #[test]
    fn test_multiple_partial_fills_avg_price() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_limit_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(3), dec!(50000),
            TimeInForce::GTC, ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();

        engine.process_fill(&order.id, dec!(49900), dec!(1), dec!(1), false).unwrap();
        engine.process_fill(&order.id, dec!(50100), dec!(1), dec!(1), false).unwrap();
        let report = engine.process_fill(&order.id, dec!(50000), dec!(1), dec!(1), false).unwrap();

        // Average: (49900*1 + 50100*1 + 50000*1) / 3 = 150000/3 = 50000
        assert_eq!(report.average_price, dec!(50000));
        assert_eq!(report.status, OrderStatus::Filled);
    }

    #[test]
    fn test_set_slippage_model_runtime() {
        let engine = ExecutionEngine::new(HashMap::new());
        assert_eq!(engine.get_slippage_config().model, SlippageModel::Percentage);

        engine.set_slippage_model(SlippageModel::Fixed);
        assert_eq!(engine.get_slippage_config().model, SlippageModel::Fixed);

        engine.set_slippage_model(SlippageModel::SquareRoot);
        assert_eq!(engine.get_slippage_config().model, SlippageModel::SquareRoot);
    }

    #[test]
    fn test_get_reports() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        engine.process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false).unwrap();

        let reports = engine.get_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].order_id, order.id);
    }

    #[test]
    fn test_orders_by_symbol() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order1 = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper,
        );
        let mut order2 = engine.create_market_order(
            Symbol::new("ETH/USDT"), Side::Buy, dec!(10), ExchangeId::Paper,
        );
        engine.submit_order(&mut order1).unwrap();
        engine.submit_order(&mut order2).unwrap();

        let btc_orders = engine.orders_by_symbol(&Symbol::new("BTC/USDT"));
        assert_eq!(btc_orders.len(), 1);

        let eth_orders = engine.orders_by_symbol(&Symbol::new("ETH/USDT"));
        assert_eq!(eth_orders.len(), 1);
    }

    #[test]
    fn test_total_pnl() {
        let engine = ExecutionEngine::new(HashMap::new());
        let mut order = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(1), ExchangeId::Paper,
        );
        engine.submit_order(&mut order).unwrap();
        engine.process_fill(&order.id, dec!(50000), dec!(1), dec!(5), false).unwrap();

        let unrealized = engine.total_unrealized_pnl();
        let realized = engine.total_realized_pnl();
        // Position opened at 50000, mark_price = 50000, so unrealized = 0
        assert_eq!(unrealized, Decimal::ZERO);
        assert_eq!(realized, Decimal::ZERO);
    }

    #[test]
    fn test_position_reduction() {
        let engine = ExecutionEngine::new(HashMap::new());
        
        // Buy 2 BTC at 50000
        let mut order1 = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Buy, dec!(2), ExchangeId::Paper,
        );
        engine.submit_order(&mut order1).unwrap();
        engine.process_fill(&order1.id, dec!(50000), dec!(2), dec!(5), false).unwrap();

        // Sell 1 BTC at 60000 (Reduction)
        let mut order2 = engine.create_market_order(
            Symbol::new("BTC/USDT"), Side::Sell, dec!(1), ExchangeId::Paper,
        );
        engine.submit_order(&mut order2).unwrap();
        engine.process_fill(&order2.id, dec!(60000), dec!(1), dec!(5), false).unwrap();

        let positions = engine.get_positions();
        let pos = positions.get(&Symbol::new("BTC/USDT")).unwrap();
        assert_eq!(pos.side, Side::Buy);
        assert_eq!(pos.quantity, dec!(1));
        assert_eq!(pos.entry_price, dec!(50000));
        assert_eq!(pos.realized_pnl, dec!(10000)); // (60000 - 50000) * 1
    }
}
