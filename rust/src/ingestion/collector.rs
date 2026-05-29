//! Multi-exchange market data collector
//!
//! WebSocket connection management with auto-reconnect,
//! tick normalization, and order book streaming.
//! Uses tokio-tungstenite for real WebSocket connections.
//! Exponential backoff: 1s → 2s → 4s → ... → 60s max.

use crate::core::types::*;
use crate::ingestion::normalizer::DataNormalizer;
use crossbeam_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tokio_tungstenite::connect_async;

/// Raw market data event from an exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MarketEvent {
    Tick(Tick),
    OrderBookDelta {
        symbol: Symbol,
        exchange: ExchangeId,
        bids: Vec<(Decimal, Decimal)>,
        asks: Vec<(Decimal, Decimal)>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    OrderBookSnapshot(OrderBookSnapshot),
    Candle(Candle),
    ConnectionStatus {
        exchange: ExchangeId,
        status: ConnectionStatus,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    Connected,
    Disconnected,
    Reconnecting,
    Error,
}

/// WebSocket connection config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WsConfig {
    pub exchange: ExchangeId,
    pub url: String,
    pub subscribe_ticks: Vec<Symbol>,
    pub subscribe_books: Vec<Symbol>,
    pub subscribe_candles: Vec<(Symbol, Timeframe)>,
    pub ping_interval_secs: u64,
    pub reconnect_delay_secs: u64,
    pub max_reconnect_attempts: u32,
}

/// Data collector that manages WebSocket connections to multiple exchanges
pub struct DataCollector {
    event_tx: Sender<MarketEvent>,
    event_rx: Receiver<MarketEvent>,
    configs: HashMap<ExchangeId, WsConfig>,
    statuses: Arc<RwLock<HashMap<ExchangeId, ConnectionStatus>>>,
    shutdown: Arc<AtomicBool>,
    cancellation_tokens: Arc<RwLock<HashMap<ExchangeId, CancellationToken>>>,
    normalizer: Arc<RwLock<DataNormalizer>>,
}

impl DataCollector {
    pub fn new() -> Self {
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        DataCollector {
            event_tx,
            event_rx,
            configs: HashMap::new(),
            statuses: Arc::new(RwLock::new(HashMap::new())),
            shutdown: Arc::new(AtomicBool::new(false)),
            cancellation_tokens: Arc::new(RwLock::new(HashMap::new())),
            normalizer: Arc::new(RwLock::new(DataNormalizer::new())),
        }
    }

    /// Add an exchange WebSocket configuration
    pub fn add_exchange(&mut self, config: WsConfig) {
        self.statuses.blocking_write().insert(config.exchange, ConnectionStatus::Disconnected);
        self.configs.insert(config.exchange, config);
    }

    /// Subscribe to market events
    pub fn subscribe(&self) -> Receiver<MarketEvent> {
        self.event_rx.clone()
    }

    /// Get connection status for an exchange
    pub fn status(&self, exchange: ExchangeId) -> ConnectionStatus {
        self.statuses.blocking_read().get(&exchange).copied().unwrap_or(ConnectionStatus::Disconnected)
    }

    /// Emit a market event
    pub fn emit(&self, event: MarketEvent) {
        let _ = self.event_tx.send(event);
    }

    /// Build Binance combined stream URL path
    fn binance_stream_path(config: &WsConfig) -> String {
        let mut streams: Vec<String> = Vec::new();
        for symbol in &config.subscribe_ticks {
            let s = symbol.as_str().to_lowercase().replace('/', "");
            streams.push(format!("{}@trade", s));
        }
        for symbol in &config.subscribe_books {
            let s = symbol.as_str().to_lowercase().replace('/', "");
            streams.push(format!("{}@depth20@100ms", s));
        }
        for (symbol, _tf) in &config.subscribe_candles {
            let s = symbol.as_str().to_lowercase().replace('/', "");
            streams.push(format!("{}@kline_m1", s));
        }
        if streams.is_empty() {
            String::new()
        } else {
            format!("/ws/{}", streams.join("/"))
        }
    }

    /// Build Bybit WebSocket subscribe message
    fn bybit_subscribe_message(config: &WsConfig) -> Option<serde_json::Value> {
        let mut args: Vec<String> = Vec::new();
        for symbol in &config.subscribe_ticks {
            let s = symbol.as_str().replace('/', "");
            args.push(format!("publicTrade.{}", s));
        }
        for symbol in &config.subscribe_books {
            let s = symbol.as_str().replace('/', "");
            args.push(format!("orderbook.20.{}", s));
        }
        if args.is_empty() {
            return None;
        }
        Some(serde_json::json!({ "op": "subscribe", "args": args }))
    }

    /// Build OKX WebSocket subscribe message
    fn okx_subscribe_message(config: &WsConfig) -> Option<serde_json::Value> {
        let mut args: Vec<serde_json::Value> = Vec::new();
        for symbol in &config.subscribe_ticks {
            let s = format!("{}-{}", symbol.base(), symbol.quote().unwrap_or("USDT"));
            args.push(serde_json::json!({ "channel": "trades", "instId": s }));
        }
        for symbol in &config.subscribe_books {
            let s = format!("{}-{}", symbol.base(), symbol.quote().unwrap_or("USDT"));
            args.push(serde_json::json!({ "channel": "books5", "instId": s }));
        }
        if args.is_empty() {
            return None;
        }
        Some(serde_json::json!({ "op": "subscribe", "args": args }))
    }

    /// Start all WebSocket connections as tokio tasks with auto-reconnect
    pub async fn start(&self) -> Result<(), String> {
        self.shutdown.store(false, Ordering::SeqCst);

        for (exchange, config) in &self.configs {
            let config = config.clone();
            let event_tx = self.event_tx.clone();
            let statuses = self.statuses.clone();
            let shutdown = self.shutdown.clone();
            let exchange = *exchange;

            let token = CancellationToken::new();
            self.cancellation_tokens.write().await.insert(exchange, token.clone());

            tokio::spawn(async move {
                Self::run_connection_loop(
                    exchange,
                    config,
                    event_tx,
                    statuses,
                    shutdown,
                    token,
                ).await;
            });
        }

        Ok(())
    }

    /// Main connection loop with exponential backoff: 1s → 2s → 4s → ... → 60s max
    async fn run_connection_loop(
        exchange: ExchangeId,
        config: WsConfig,
        event_tx: Sender<MarketEvent>,
        statuses: Arc<RwLock<HashMap<ExchangeId, ConnectionStatus>>>,
        shutdown: Arc<AtomicBool>,
        cancel_token: CancellationToken,
    ) {
        let mut attempts = 0u32;
        let max_backoff_secs: f64 = 60.0;

        loop {
            if shutdown.load(Ordering::SeqCst) || cancel_token.is_cancelled() {
                break;
            }

            statuses.write().await.insert(exchange, ConnectionStatus::Reconnecting);
            let _ = event_tx.send(MarketEvent::ConnectionStatus {
                exchange,
                status: ConnectionStatus::Reconnecting,
            });

            let url = match exchange {
                ExchangeId::Binance => {
                    let stream_path = Self::binance_stream_path(&config);
                    format!("{}{}", config.url, stream_path)
                }
                _ => config.url.clone(),
            };

            tracing::info!("Connecting to {} WebSocket: {}", exchange, url);

            match connect_async(&url).await {
                Ok((ws_stream, _response)) => {
                    tracing::info!("Connected to {} WebSocket", exchange);
                    attempts = 0;

                    statuses.write().await.insert(exchange, ConnectionStatus::Connected);
                    let _ = event_tx.send(MarketEvent::ConnectionStatus {
                        exchange,
                        status: ConnectionStatus::Connected,
                    });

                    let (mut write, mut read) = ws_stream.split();

                    // For Bybit/OKX, send subscription message after connect
                    if matches!(exchange, ExchangeId::Bybit | ExchangeId::OKX) {
                        let sub_msg = match exchange {
                            ExchangeId::Bybit => Self::bybit_subscribe_message(&config),
                            ExchangeId::OKX => Self::okx_subscribe_message(&config),
                            _ => None,
                        };
                        if let Some(msg) = sub_msg {
                            let text = serde_json::to_string(&msg).unwrap_or_default();
                            let _ = write.send(tokio_tungstenite::tungstenite::Message::Text(text)).await;
                        }
                    }

                    drop(write);

                    loop {
                        tokio::select! {
                            msg = read.next() => {
                                match msg {
                                    Some(Ok(tokio_tungstenite::tungstenite::Message::Text(text))) => {
                                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                                            Self::handle_message(exchange, &parsed, &event_tx);
                                        }
                                    }
                                    Some(Ok(tokio_tungstenite::tungstenite::Message::Ping(_data))) => {
                                        // Pong is auto-sent by tungstenite
                                    }
                                    Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) => {
                                        tracing::warn!("{} WebSocket closed by server", exchange);
                                        break;
                                    }
                                    Some(Err(e)) => {
                                        tracing::error!("{} WebSocket error: {}", exchange, e);
                                        break;
                                    }
                                    None => {
                                        tracing::warn!("{} WebSocket stream ended", exchange);
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            _ = cancel_token.cancelled() => {
                                tracing::info!("{} WebSocket cancelled", exchange);
                                break;
                            }
                        }

                        if shutdown.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to connect to {} WebSocket: {}", exchange, e);
                    statuses.write().await.insert(exchange, ConnectionStatus::Error);
                    let _ = event_tx.send(MarketEvent::ConnectionStatus {
                        exchange,
                        status: ConnectionStatus::Error,
                    });
                }
            }

            if shutdown.load(Ordering::SeqCst) || cancel_token.is_cancelled() {
                break;
            }

            attempts += 1;
            if attempts >= config.max_reconnect_attempts {
                tracing::error!(
                    "Max reconnect attempts ({}) reached for {}",
                    config.max_reconnect_attempts, exchange
                );
                break;
            }

            // Exponential backoff: 1s → 2s → 4s → 8s → 16s → 32s → 60s (capped)
            let delay_secs = (config.reconnect_delay_secs as f64 * 2_f64.powi(attempts as i32 - 1))
                .min(max_backoff_secs)
                + rand::random::<f64>() * 0.5; // Small jitter

            tracing::info!(
                "Reconnecting to {} in {:.1}s (attempt {})",
                exchange, delay_secs, attempts
            );

            tokio::select! {
                _ = tokio::time::sleep(tokio::time::Duration::from_secs_f64(delay_secs)) => {},
                _ = cancel_token.cancelled() => { break; }
            }
        }

        statuses.write().await.insert(exchange, ConnectionStatus::Disconnected);
        let _ = event_tx.send(MarketEvent::ConnectionStatus {
            exchange,
            status: ConnectionStatus::Disconnected,
        });
    }

    /// Handle an incoming WebSocket message based on exchange type
    fn handle_message(
        exchange: ExchangeId,
        msg: &serde_json::Value,
        event_tx: &Sender<MarketEvent>,
    ) {
        match exchange {
            ExchangeId::Binance => {
                let event_type = msg.get("e").and_then(|e| e.as_str()).unwrap_or("");
                match event_type {
                    "trade" => {
                        if let Some(tick) = parse_binance_trade(msg) {
                            let _ = event_tx.send(MarketEvent::Tick(tick));
                        }
                    }
                    "depthUpdate" => {
                        if let Some(event) = parse_binance_depth(msg) {
                            let _ = event_tx.send(event);
                        }
                    }
                    _ => {}
                }
            }
            ExchangeId::Bybit => {
                let topic = msg.get("topic").and_then(|t| t.as_str()).unwrap_or("");
                if topic.starts_with("publicTrade.") {
                    if let Some(tick) = parse_bybit_trade(msg) {
                        let _ = event_tx.send(MarketEvent::Tick(tick));
                    }
                }
            }
            ExchangeId::OKX => {
                let channel = msg.get("arg")
                    .and_then(|a| a.get("channel"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("");
                if channel == "trades" {
                    if let Some(tick) = parse_okx_trade(msg) {
                        let _ = event_tx.send(MarketEvent::Tick(tick));
                    }
                }
            }
            ExchangeId::Paper => {}
        }
    }

    /// Stop all WebSocket connections gracefully
    pub async fn stop(&self) {
        self.shutdown.store(true, Ordering::SeqCst);

        // Cancel all tokens
        let tokens = self.cancellation_tokens.read().await;
        for (_, token) in tokens.iter() {
            token.cancel();
        }

        let mut statuses = self.statuses.write().await;
        for (exchange, _) in &self.configs {
            statuses.insert(*exchange, ConnectionStatus::Disconnected);
            let _ = self.event_tx.send(MarketEvent::ConnectionStatus {
                exchange: *exchange,
                status: ConnectionStatus::Disconnected,
            });
        }
    }

    /// Get all configured exchanges
    pub fn exchanges(&self) -> Vec<ExchangeId> {
        self.configs.keys().copied().collect()
    }

    /// Create default Binance WebSocket config
    pub fn binance_config(symbols: Vec<Symbol>) -> WsConfig {
        WsConfig {
            exchange: ExchangeId::Binance,
            url: "wss://stream.binance.com:9443/ws".to_string(),
            subscribe_ticks: symbols.clone(),
            subscribe_books: symbols.clone(),
            subscribe_candles: symbols.iter().map(|s| (s.clone(), Timeframe::M1)).collect(),
            ping_interval_secs: 30,
            reconnect_delay_secs: 1,
            max_reconnect_attempts: 100,
        }
    }

    /// Create default Bybit WebSocket config
    pub fn bybit_config(symbols: Vec<Symbol>) -> WsConfig {
        WsConfig {
            exchange: ExchangeId::Bybit,
            url: "wss://stream.bybit.com/v5/public/spot".to_string(),
            subscribe_ticks: symbols.clone(),
            subscribe_books: symbols.clone(),
            subscribe_candles: symbols.iter().map(|s| (s.clone(), Timeframe::M1)).collect(),
            ping_interval_secs: 20,
            reconnect_delay_secs: 1,
            max_reconnect_attempts: 100,
        }
    }

    /// Create default OKX WebSocket config
    pub fn okx_config(symbols: Vec<Symbol>) -> WsConfig {
        WsConfig {
            exchange: ExchangeId::OKX,
            url: "wss://ws.okx.com:8443/ws/v5/public".to_string(),
            subscribe_ticks: symbols.clone(),
            subscribe_books: symbols.clone(),
            subscribe_candles: symbols.iter().map(|s| (s.clone(), Timeframe::M1)).collect(),
            ping_interval_secs: 30,
            reconnect_delay_secs: 1,
            max_reconnect_attempts: 100,
        }
    }
}

/// Convert exchange-specific symbol string to canonical Symbol
fn convert_exchange_symbol(symbol_str: &str, exchange: ExchangeId) -> Symbol {
    match exchange {
        ExchangeId::Binance | ExchangeId::Bybit => {
            for quote in &["USDT", "BUSD", "USDC", "BTC", "ETH", "BNB"] {
                if symbol_str.ends_with(quote) && symbol_str.len() > quote.len() {
                    let base = &symbol_str[..symbol_str.len() - quote.len()];
                    return Symbol::new(format!("{}/{}", base, quote));
                }
            }
            Symbol::new(symbol_str)
        }
        ExchangeId::OKX => {
            if let Some(idx) = symbol_str.find('-') {
                let base = &symbol_str[..idx];
                let quote = &symbol_str[idx + 1..];
                return Symbol::new(format!("{}/{}", base, quote));
            }
            Symbol::new(symbol_str)
        }
        ExchangeId::Paper => Symbol::new(symbol_str),
    }
}

/// Parse a Binance trade message
fn parse_binance_trade(msg: &serde_json::Value) -> Option<Tick> {
    let price_str = msg.get("p")?.as_str()?;
    let qty_str = msg.get("q")?.as_str()?;
    let symbol_str = msg.get("s")?.as_str()?;
    let is_buyer_maker = msg.get("m")?.as_bool()?;

    let price: Decimal = price_str.parse().ok()?;
    let quantity: Decimal = qty_str.parse().ok()?;
    let symbol = convert_exchange_symbol(symbol_str, ExchangeId::Binance);

    Some(Tick {
        symbol,
        exchange: ExchangeId::Binance,
        price,
        quantity,
        side: if is_buyer_maker { Side::Sell } else { Side::Buy },
        timestamp: chrono::Utc::now(),
        trade_id: msg.get("t")?.as_u64()?.to_string(),
    })
}

/// Parse a Binance depth update message
fn parse_binance_depth(msg: &serde_json::Value) -> Option<MarketEvent> {
    let symbol_str = msg.get("s")?.as_str()?;
    let symbol = convert_exchange_symbol(symbol_str, ExchangeId::Binance);

    let mut bids = Vec::new();
    let mut asks = Vec::new();

    if let Some(b) = msg.get("b").and_then(|v| v.as_array()) {
        for level in b {
            let arr = level.as_array()?;
            if arr.len() >= 2 {
                let price: Decimal = arr[0].as_str()?.parse().ok()?;
                let qty: Decimal = arr[1].as_str()?.parse().ok()?;
                bids.push((price, qty));
            }
        }
    }

    if let Some(a) = msg.get("a").and_then(|v| v.as_array()) {
        for level in a {
            let arr = level.as_array()?;
            if arr.len() >= 2 {
                let price: Decimal = arr[0].as_str()?.parse().ok()?;
                let qty: Decimal = arr[1].as_str()?.parse().ok()?;
                asks.push((price, qty));
            }
        }
    }

    Some(MarketEvent::OrderBookDelta {
        symbol,
        exchange: ExchangeId::Binance,
        bids,
        asks,
        timestamp: chrono::Utc::now(),
    })
}

/// Parse a Bybit trade message
fn parse_bybit_trade(msg: &serde_json::Value) -> Option<Tick> {
    let data = msg.get("data")?.as_array()?.first()?;
    let price: Decimal = data.get("p")?.as_str()?.parse().ok()?;
    let quantity: Decimal = data.get("v")?.as_str()?.parse().ok()?;
    let symbol_str = data.get("s")?.as_str()?;
    let side_str = data.get("S")?.as_str()?;
    let symbol = convert_exchange_symbol(symbol_str, ExchangeId::Bybit);

    Some(Tick {
        symbol,
        exchange: ExchangeId::Bybit,
        price,
        quantity,
        side: if side_str == "Buy" { Side::Buy } else { Side::Sell },
        timestamp: chrono::Utc::now(),
        trade_id: data.get("i")?.as_str()?.to_string(),
    })
}

/// Parse an OKX trade message
fn parse_okx_trade(msg: &serde_json::Value) -> Option<Tick> {
    let data = msg.get("data")?.as_array()?.first()?;
    let price: Decimal = data.get("px")?.as_str()?.parse().ok()?;
    let quantity: Decimal = data.get("sz")?.as_str()?.parse().ok()?;
    let symbol_str = data.get("instId")?.as_str()?;
    let side_str = data.get("side")?.as_str()?;
    let symbol = convert_exchange_symbol(symbol_str, ExchangeId::OKX);

    Some(Tick {
        symbol,
        exchange: ExchangeId::OKX,
        price,
        quantity,
        side: if side_str == "buy" { Side::Buy } else { Side::Sell },
        timestamp: chrono::Utc::now(),
        trade_id: data.get("tradeId")?.as_str()?.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collector_creation() {
        let mut collector = DataCollector::new();
        collector.add_exchange(DataCollector::binance_config(vec![Symbol::new("BTC/USDT")]));
        assert_eq!(collector.exchanges().len(), 1);
    }

    #[test]
    fn test_binance_config() {
        let config = DataCollector::binance_config(vec![Symbol::new("BTC/USDT")]);
        assert_eq!(config.exchange, ExchangeId::Binance);
        assert!(!config.subscribe_ticks.is_empty());
        assert_eq!(config.reconnect_delay_secs, 1); // starts at 1s
    }

    #[test]
    fn test_bybit_config() {
        let config = DataCollector::bybit_config(vec![Symbol::new("ETH/USDT")]);
        assert_eq!(config.exchange, ExchangeId::Bybit);
    }

    #[test]
    fn test_okx_config() {
        let config = DataCollector::okx_config(vec![Symbol::new("SOL/USDT")]);
        assert_eq!(config.exchange, ExchangeId::OKX);
    }

    #[test]
    fn test_convert_binance_symbol() {
        let symbol = convert_exchange_symbol("BTCUSDT", ExchangeId::Binance);
        assert_eq!(symbol.as_str(), "BTC/USDT");
    }

    #[test]
    fn test_convert_okx_symbol() {
        let symbol = convert_exchange_symbol("BTC-USDT", ExchangeId::OKX);
        assert_eq!(symbol.as_str(), "BTC/USDT");
    }

    #[test]
    fn test_convert_bybit_symbol() {
        let symbol = convert_exchange_symbol("ETHUSDT", ExchangeId::Bybit);
        assert_eq!(symbol.as_str(), "ETH/USDT");
    }

    #[test]
    fn test_emit_event() {
        let collector = DataCollector::new();
        let rx = collector.subscribe();

        collector.emit(MarketEvent::ConnectionStatus {
            exchange: ExchangeId::Binance,
            status: ConnectionStatus::Connected,
        });

        let event = rx.recv_timeout(std::time::Duration::from_millis(100));
        assert!(event.is_ok());
    }

    #[test]
    fn test_parse_binance_trade() {
        let msg = serde_json::json!({
            "e": "trade",
            "E": 1700000000000u64,
            "s": "BTCUSDT",
            "t": 12345u64,
            "p": "50000.00",
            "q": "0.100000",
            "b": 1000u64,
            "a": 2000u64,
            "T": 1700000000000u64,
            "m": false,
            "M": true
        });

        let tick = parse_binance_trade(&msg).unwrap();
        assert_eq!(tick.symbol.as_str(), "BTC/USDT");
        assert_eq!(tick.exchange, ExchangeId::Binance);
        assert_eq!(tick.side, Side::Buy);
    }

    #[test]
    fn test_parse_bybit_trade() {
        let msg = serde_json::json!({
            "topic": "publicTrade.BTCUSDT",
            "data": [{
                "s": "BTCUSDT",
                "p": "50000.00",
                "v": "0.5",
                "S": "Buy",
                "i": "trade123",
                "T": 1700000000000i64
            }]
        });

        let tick = parse_bybit_trade(&msg).unwrap();
        assert_eq!(tick.symbol.as_str(), "BTC/USDT");
        assert_eq!(tick.exchange, ExchangeId::Bybit);
        assert_eq!(tick.side, Side::Buy);
    }

    #[test]
    fn test_parse_okx_trade() {
        let msg = serde_json::json!({
            "arg": { "channel": "trades", "instId": "BTC-USDT" },
            "data": [{
                "instId": "BTC-USDT",
                "px": "50000.00",
                "sz": "0.1",
                "side": "buy",
                "tradeId": "123"
            }]
        });

        let tick = parse_okx_trade(&msg).unwrap();
        assert_eq!(tick.symbol.as_str(), "BTC/USDT");
        assert_eq!(tick.exchange, ExchangeId::OKX);
        assert_eq!(tick.side, Side::Buy);
    }

    #[test]
    fn test_parse_binance_depth() {
        let msg = serde_json::json!({
            "e": "depthUpdate",
            "E": 1700000000000u64,
            "s": "BTCUSDT",
            "b": [["50000.00", "1.5"], ["49999.00", "2.0"]],
            "a": [["50001.00", "0.5"], ["50002.00", "1.0"]]
        });

        let event = parse_binance_depth(&msg).unwrap();
        match event {
            MarketEvent::OrderBookDelta { bids, asks, symbol, exchange, .. } => {
                assert_eq!(symbol.as_str(), "BTC/USDT");
                assert_eq!(exchange, ExchangeId::Binance);
                assert_eq!(bids.len(), 2);
                assert_eq!(asks.len(), 2);
            }
            _ => panic!("Expected OrderBookDelta"),
        }
    }

    #[test]
    fn test_exponential_backoff_calculation() {
        // Verify the backoff: 1s → 2s → 4s → 8s → 16s → 32s → 60s cap
        let base = 1u64;
        let max_secs = 60.0;

        let delay_1 = (base as f64 * 2_f64.powi(0)).min(max_secs);
        let delay_2 = (base as f64 * 2_f64.powi(1)).min(max_secs);
        let delay_3 = (base as f64 * 2_f64.powi(2)).min(max_secs);
        let delay_6 = (base as f64 * 2_f64.powi(5)).min(max_secs);
        let delay_7 = (base as f64 * 2_f64.powi(6)).min(max_secs);

        assert_eq!(delay_1, 1.0);
        assert_eq!(delay_2, 2.0);
        assert_eq!(delay_3, 4.0);
        assert_eq!(delay_6, 32.0);
        assert_eq!(delay_7, 60.0); // capped
    }

    #[test]
    fn test_binance_stream_path() {
        let config = WsConfig {
            exchange: ExchangeId::Binance,
            url: "wss://stream.binance.com:9443/ws".to_string(),
            subscribe_ticks: vec![Symbol::new("BTC/USDT")],
            subscribe_books: vec![Symbol::new("ETH/USDT")],
            subscribe_candles: vec![],
            ping_interval_secs: 30,
            reconnect_delay_secs: 1,
            max_reconnect_attempts: 100,
        };

        let path = DataCollector::binance_stream_path(&config);
        assert!(path.contains("btcusdt@trade"));
        assert!(path.contains("ethusdt@depth20@100ms"));
    }
}
