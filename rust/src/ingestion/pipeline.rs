//! Data Pipeline
//!
//! Apache Arrow record batch construction, Parquet writing,
//! streaming pipeline with crossbeam channels, data quality checks,
//! file rotation (size-based and time-based), and compression (Snappy/Zstd).

use crate::core::types::*;
use arrow::array::*;
use arrow::datatypes::*;
use arrow::record_batch::RecordBatch;
use crossbeam_channel::{Receiver, Sender};
use parquet::arrow::ArrowWriter;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use std::fmt;
use std::sync::Arc;

/// Pipeline event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineEvent {
    TickBatch(Vec<Tick>),
    CandleBatch(Vec<Candle>),
    QualityAlert {
        symbol: Symbol,
        issue: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    GapDetected {
        symbol: Symbol,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    },
    OutlierDetected {
        symbol: Symbol,
        price: Decimal,
        expected_range: (Decimal, Decimal),
    },
}

/// Data quality checker
#[derive(Debug, Clone)]
pub struct QualityChecker {
    max_gap_secs: u64,
    max_price_change_pct: Decimal,
    min_volume: Decimal,
    last_ticks: std::collections::HashMap<Symbol, Tick>,
    total_ticks: u64,
    total_gaps: u64,
    total_outliers: u64,
}

impl QualityChecker {
    pub fn new(max_gap_secs: u64, max_price_change_pct: Decimal, min_volume: Decimal) -> Self {
        QualityChecker {
            max_gap_secs,
            max_price_change_pct,
            min_volume,
            last_ticks: std::collections::HashMap::new(),
            total_ticks: 0,
            total_gaps: 0,
            total_outliers: 0,
        }
    }

    /// Check a tick for quality issues
    pub fn check(&mut self, tick: &Tick) -> Vec<PipelineEvent> {
        let mut events = Vec::new();
        self.total_ticks += 1;

        if let Some(last) = self.last_ticks.get(&tick.symbol) {
            // Check for gap
            let gap = (tick.timestamp - last.timestamp).num_seconds().unsigned_abs();
            if gap > self.max_gap_secs {
                self.total_gaps += 1;
                events.push(PipelineEvent::GapDetected {
                    symbol: tick.symbol.clone(),
                    from: last.timestamp,
                    to: tick.timestamp,
                });
            }

            // Check for price outlier
            if last.price > Decimal::ZERO {
                let change = (tick.price - last.price).abs() / last.price
                    * rust_decimal_macros::dec!(100);
                if change > self.max_price_change_pct {
                    self.total_outliers += 1;
                    let expected_low = last.price * (Decimal::ONE - self.max_price_change_pct / rust_decimal_macros::dec!(100));
                    let expected_high = last.price * (Decimal::ONE + self.max_price_change_pct / rust_decimal_macros::dec!(100));
                    events.push(PipelineEvent::OutlierDetected {
                        symbol: tick.symbol.clone(),
                        price: tick.price,
                        expected_range: (expected_low, expected_high),
                    });
                }
            }
        }

        self.last_ticks.insert(tick.symbol.clone(), tick.clone());
        events
    }

    /// Get quality statistics
    pub fn stats(&self) -> QualityStats {
        QualityStats {
            total_ticks: self.total_ticks,
            total_gaps: self.total_gaps,
            total_outliers: self.total_outliers,
            gap_rate: if self.total_ticks > 0 {
                Decimal::from(self.total_gaps) / Decimal::from(self.total_ticks)
            } else {
                Decimal::ZERO
            },
            outlier_rate: if self.total_ticks > 0 {
                Decimal::from(self.total_outliers) / Decimal::from(self.total_ticks)
            } else {
                Decimal::ZERO
            },
        }
    }
}

/// Quality statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    pub total_ticks: u64,
    pub total_gaps: u64,
    pub total_outliers: u64,
    pub gap_rate: Decimal,
    pub outlier_rate: Decimal,
}

/// Tick batcher for efficient processing
pub struct TickBatcher {
    buffer: VecDeque<Tick>,
    batch_size: usize,
    flush_interval_ms: u64,
    last_flush: chrono::DateTime<chrono::Utc>,
    output_tx: Sender<PipelineEvent>,
}

impl TickBatcher {
    pub fn new(batch_size: usize, flush_interval_ms: u64, output_tx: Sender<PipelineEvent>) -> Self {
        TickBatcher {
            buffer: VecDeque::with_capacity(batch_size),
            batch_size,
            flush_interval_ms,
            last_flush: chrono::Utc::now(),
            output_tx,
        }
    }

    pub fn add(&mut self, tick: Tick) {
        self.buffer.push_back(tick);

        let should_flush = self.buffer.len() >= self.batch_size
            || (chrono::Utc::now() - self.last_flush).num_milliseconds() as u64 >= self.flush_interval_ms;

        if should_flush {
            self.flush();
        }
    }

    pub fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let batch: Vec<Tick> = self.buffer.drain(..).collect();
        let _ = self.output_tx.send(PipelineEvent::TickBatch(batch));
        self.last_flush = chrono::Utc::now();
    }
}

/// Compression type for Parquet output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    None,
    Snappy,
    Zstd,
}

impl fmt::Display for CompressionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionType::None => write!(f, "none"),
            CompressionType::Snappy => write!(f, "snappy"),
            CompressionType::Zstd => write!(f, "zstd"),
        }
    }
}

/// Build the Arrow schema for tick data
pub fn tick_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("timestamp", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("exchange", DataType::Utf8, false),
        Field::new("price", DataType::Float64, false),
        Field::new("quantity", DataType::Float64, false),
        Field::new("side", DataType::Utf8, false),
        Field::new("trade_id", DataType::Utf8, true),
    ]))
}

/// Build the Arrow schema for candle data
pub fn candle_arrow_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("open_time", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        Field::new("close_time", DataType::Timestamp(TimeUnit::Millisecond, None), false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("exchange", DataType::Utf8, false),
        Field::new("timeframe", DataType::Utf8, false),
        Field::new("open", DataType::Float64, false),
        Field::new("high", DataType::Float64, false),
        Field::new("low", DataType::Float64, false),
        Field::new("close", DataType::Float64, false),
        Field::new("volume", DataType::Float64, false),
        Field::new("quote_volume", DataType::Float64, false),
        Field::new("trades", DataType::Int64, false),
        Field::new("taker_buy_volume", DataType::Float64, false),
    ]))
}

/// Convert a slice of Ticks into an Arrow RecordBatch
pub fn ticks_to_record_batch(ticks: &[Tick]) -> Result<RecordBatch, String> {
    let schema = tick_arrow_schema();

    let mut timestamps = TimestampMillisecondBuilder::new();
    let mut symbols = StringBuilder::new();
    let mut exchanges = StringBuilder::new();
    let mut prices = Float64Builder::new();
    let mut quantities = Float64Builder::new();
    let mut sides = StringBuilder::new();
    let mut trade_ids = StringBuilder::new();

    for tick in ticks {
        timestamps.append_value(tick.timestamp.timestamp_millis());
        symbols.append_value(tick.symbol.as_str());
        exchanges.append_value(tick.exchange.to_string());
        prices.append_value(tick.price.to_string().parse::<f64>().unwrap_or(0.0));
        quantities.append_value(tick.quantity.to_string().parse::<f64>().unwrap_or(0.0));
        sides.append_value(tick.side.to_string());
        trade_ids.append_value(tick.trade_id.as_str());
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(timestamps.finish()),
            Arc::new(symbols.finish()),
            Arc::new(exchanges.finish()),
            Arc::new(prices.finish()),
            Arc::new(quantities.finish()),
            Arc::new(sides.finish()),
            Arc::new(trade_ids.finish()),
        ],
    ).map_err(|e| format!("Failed to create RecordBatch: {}", e))?;

    Ok(batch)
}

/// Convert a slice of Candles into an Arrow RecordBatch
pub fn candles_to_record_batch(candles: &[Candle]) -> Result<RecordBatch, String> {
    let schema = candle_arrow_schema();

    let mut open_times = TimestampMillisecondBuilder::new();
    let mut close_times = TimestampMillisecondBuilder::new();
    let mut symbols = StringBuilder::new();
    let mut exchanges = StringBuilder::new();
    let mut timeframes = StringBuilder::new();
    let mut opens = Float64Builder::new();
    let mut highs = Float64Builder::new();
    let mut lows = Float64Builder::new();
    let mut closes = Float64Builder::new();
    let mut volumes = Float64Builder::new();
    let mut quote_volumes = Float64Builder::new();
    let mut trades = Int64Builder::new();
    let mut taker_buy_volumes = Float64Builder::new();

    for candle in candles {
        open_times.append_value(candle.open_time.timestamp_millis());
        close_times.append_value(candle.close_time.timestamp_millis());
        symbols.append_value(candle.symbol.as_str());
        exchanges.append_value(candle.exchange.to_string());
        timeframes.append_value(candle.timeframe.to_string());
        opens.append_value(candle.open.to_string().parse::<f64>().unwrap_or(0.0));
        highs.append_value(candle.high.to_string().parse::<f64>().unwrap_or(0.0));
        lows.append_value(candle.low.to_string().parse::<f64>().unwrap_or(0.0));
        closes.append_value(candle.close.to_string().parse::<f64>().unwrap_or(0.0));
        volumes.append_value(candle.volume.to_string().parse::<f64>().unwrap_or(0.0));
        quote_volumes.append_value(candle.quote_volume.to_string().parse::<f64>().unwrap_or(0.0));
        trades.append_value(candle.trades as i64);
        taker_buy_volumes.append_value(candle.taker_buy_volume.to_string().parse::<f64>().unwrap_or(0.0));
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(open_times.finish()),
            Arc::new(close_times.finish()),
            Arc::new(symbols.finish()),
            Arc::new(exchanges.finish()),
            Arc::new(timeframes.finish()),
            Arc::new(opens.finish()),
            Arc::new(highs.finish()),
            Arc::new(lows.finish()),
            Arc::new(closes.finish()),
            Arc::new(volumes.finish()),
            Arc::new(quote_volumes.finish()),
            Arc::new(trades.finish()),
            Arc::new(taker_buy_volumes.finish()),
        ],
    ).map_err(|e| format!("Failed to create RecordBatch: {}", e))?;

    Ok(batch)
}

/// Write tick data to a Parquet file
pub fn write_ticks_to_parquet(ticks: &[Tick], path: &Path) -> Result<(), String> {
    let batch = ticks_to_record_batch(ticks)?;
    write_record_batch_to_parquet(&batch, path, CompressionType::Snappy)
}

/// Write tick data to a Parquet file with specific compression
pub fn write_ticks_to_parquet_with_compression(ticks: &[Tick], path: &Path, compression: CompressionType) -> Result<(), String> {
    let batch = ticks_to_record_batch(ticks)?;
    write_record_batch_to_parquet(&batch, path, compression)
}

/// Write candle data to a Parquet file
pub fn write_candles_to_parquet(candles: &[Candle], path: &Path) -> Result<(), String> {
    let batch = candles_to_record_batch(candles)?;
    write_record_batch_to_parquet(&batch, path, CompressionType::Snappy)
}

/// Write candle data with specific compression
pub fn write_candles_to_parquet_with_compression(candles: &[Candle], path: &Path, compression: CompressionType) -> Result<(), String> {
    let batch = candles_to_record_batch(candles)?;
    write_record_batch_to_parquet(&batch, path, compression)
}

/// Write an Arrow RecordBatch to a Parquet file with configurable compression
pub fn write_record_batch_to_parquet(batch: &RecordBatch, path: &Path, compression: CompressionType) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Failed to create file {:?}: {}", path, e))?;

    let compression_setting = match compression {
        CompressionType::None => parquet::basic::Compression::UNCOMPRESSED,
        CompressionType::Snappy => parquet::basic::Compression::SNAPPY,
        CompressionType::Zstd => parquet::basic::Compression::ZSTD(parquet::basic::ZstdLevel::default()),
    };

    let props = parquet::file::properties::WriterProperties::builder()
        .set_compression(compression_setting)
        .set_created_by("acms-core".to_string())
        .build();

    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
        .map_err(|e| format!("Failed to create Parquet writer: {}", e))?;

    writer.write(batch)
        .map_err(|e| format!("Failed to write RecordBatch: {}", e))?;

    writer.close()
        .map_err(|e| format!("Failed to close Parquet writer: {}", e))?;

    Ok(())
}

/// Data pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub quality_max_gap_secs: u64,
    pub quality_max_price_change_pct: Decimal,
    pub quality_min_volume: Decimal,
    pub parquet_output_dir: String,
    pub parquet_rotation_size_mb: u64,
    pub parquet_rotation_interval_secs: u64,
    pub compression: CompressionType,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            batch_size: 10000,
            flush_interval_ms: 1000,
            quality_max_gap_secs: 30,
            quality_max_price_change_pct: rust_decimal_macros::dec!(10),
            quality_min_volume: Decimal::ZERO,
            parquet_output_dir: "/data/acms/parquet".to_string(),
            parquet_rotation_size_mb: 100,
            parquet_rotation_interval_secs: 3600,
            compression: CompressionType::Snappy,
        }
    }
}

/// File rotation tracker
#[derive(Debug, Clone)]
pub struct FileRotator {
    current_file_index: u64,
    current_file_size_bytes: u64,
    max_file_size_bytes: u64,
    last_rotation: chrono::DateTime<chrono::Utc>,
    rotation_interval_secs: u64,
}

impl FileRotator {
    pub fn new(max_file_size_mb: u64, rotation_interval_secs: u64) -> Self {
        FileRotator {
            current_file_index: 0,
            current_file_size_bytes: 0,
            max_file_size_bytes: max_file_size_mb * 1024 * 1024,
            last_rotation: chrono::Utc::now(),
            rotation_interval_secs,
        }
    }

    /// Check if rotation is needed based on size or time
    pub fn should_rotate(&self, new_data_size: usize) -> bool {
        let size_exceeded = self.current_file_size_bytes + new_data_size as u64 > self.max_file_size_bytes;
        let time_exceeded = (chrono::Utc::now() - self.last_rotation).num_seconds() as u64 >= self.rotation_interval_secs;
        size_exceeded || time_exceeded
    }

    /// Perform rotation
    pub fn rotate(&mut self) {
        self.current_file_index += 1;
        self.current_file_size_bytes = 0;
        self.last_rotation = chrono::Utc::now();
    }

    /// Record data written
    pub fn record_write(&mut self, size: usize) {
        self.current_file_size_bytes += size as u64;
    }

    /// Get current file path
    pub fn current_path(&self, base_dir: &str, prefix: &str) -> String {
        format!("{}/{}_{}.parquet", base_dir, prefix, self.current_file_index)
    }

    /// Get current file index
    pub fn file_index(&self) -> u64 {
        self.current_file_index
    }
}

/// Main data pipeline
pub struct DataPipeline {
    config: PipelineConfig,
    quality_checker: QualityChecker,
    batcher: Option<TickBatcher>,
    event_rx: Receiver<PipelineEvent>,
    event_tx: Sender<PipelineEvent>,
    file_rotator: FileRotator,
    candles: Vec<Candle>,
}

impl DataPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let quality_checker = QualityChecker::new(
            config.quality_max_gap_secs,
            config.quality_max_price_change_pct,
            config.quality_min_volume,
        );
        let file_rotator = FileRotator::new(
            config.parquet_rotation_size_mb,
            config.parquet_rotation_interval_secs,
        );

        DataPipeline {
            config,
            quality_checker,
            batcher: None,
            event_rx,
            event_tx: event_tx.clone(),
            file_rotator,
            candles: Vec::new(),
        }
    }

    /// Process an incoming tick
    pub fn process_tick(&mut self, tick: Tick) {
        // Quality check
        let quality_events = self.quality_checker.check(&tick);
        for event in quality_events {
            let _ = self.event_tx.send(event);
        }

        // Batch
        if self.batcher.is_none() {
            self.batcher = Some(TickBatcher::new(
                self.config.batch_size,
                self.config.flush_interval_ms,
                self.event_tx.clone(),
            ));
        }
        if let Some(batcher) = &mut self.batcher {
            batcher.add(tick);
        }
    }

    /// Get candles from the pipeline
    pub fn get_candles(&self) -> Vec<Candle> {
        self.candles.clone()
    }

    /// Set candles (for external candle aggregation)
    pub fn set_candles(&mut self, candles: Vec<Candle>) {
        self.candles = candles;
    }

    /// Get quality statistics
    pub fn get_quality_stats(&self) -> QualityStats {
        self.quality_checker.stats()
    }

    /// Subscribe to pipeline events
    pub fn subscribe(&self) -> Receiver<PipelineEvent> {
        self.event_rx.clone()
    }

    /// Flush pending data
    pub fn flush(&mut self) {
        if let Some(batcher) = &mut self.batcher {
            batcher.flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use tempfile::TempDir;

    fn make_tick(price: Decimal, qty: Decimal, side: Side) -> Tick {
        Tick {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Binance,
            price,
            quantity: qty,
            side,
            timestamp: chrono::Utc::now(),
            trade_id: "1".to_string(),
        }
    }

    #[test]
    fn test_quality_checker() {
        let mut checker = QualityChecker::new(30, dec!(10), Decimal::ZERO);
        let tick = make_tick(dec!(50000), dec!(1), Side::Buy);
        let events = checker.check(&tick);
        assert!(events.is_empty());
    }

    #[test]
    fn test_data_pipeline() {
        let config = PipelineConfig::default();
        let mut pipeline = DataPipeline::new(config);
        let tick = make_tick(dec!(50000), dec!(1), Side::Buy);
        pipeline.process_tick(tick);
        pipeline.flush();
    }

    #[test]
    fn test_ticks_to_record_batch() {
        let ticks = vec![
            make_tick(dec!(50000), dec!(1), Side::Buy),
            make_tick(dec!(50001), dec!(0.5), Side::Sell),
        ];
        let batch = ticks_to_record_batch(&ticks).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 7);
    }

    #[test]
    fn test_candles_to_record_batch() {
        let now = chrono::Utc::now();
        let candles = vec![Candle {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Binance,
            timeframe: Timeframe::M1,
            open_time: now,
            close_time: now + chrono::Duration::seconds(60),
            open: dec!(50000),
            high: dec!(50100),
            low: dec!(49900),
            close: dec!(50050),
            volume: dec!(100),
            quote_volume: dec!(5000000),
            trades: 500,
            taker_buy_volume: dec!(50),
            taker_buy_quote_volume: dec!(2500000),
        }];
        let batch = candles_to_record_batch(&candles).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 13);
    }

    #[test]
    fn test_write_ticks_to_parquet() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ticks.parquet");
        let ticks = vec![
            make_tick(dec!(50000), dec!(1), Side::Buy),
            make_tick(dec!(50001), dec!(0.5), Side::Sell),
        ];
        write_ticks_to_parquet(&ticks, &path).unwrap();
        assert!(path.exists());

        let file = File::open(&path).unwrap();
        let reader = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap().build().unwrap();
        let mut total_rows = 0;
        for batch in reader {
            total_rows += batch.unwrap().num_rows();
        }
        assert_eq!(total_rows, 2);
    }

    #[test]
    fn test_write_candles_to_parquet() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("candles.parquet");
        let now = chrono::Utc::now();
        let candles = vec![Candle {
            symbol: Symbol::new("BTC/USDT"),
            exchange: ExchangeId::Binance,
            timeframe: Timeframe::M1,
            open_time: now,
            close_time: now + chrono::Duration::seconds(60),
            open: dec!(50000),
            high: dec!(50100),
            low: dec!(49900),
            close: dec!(50050),
            volume: dec!(100),
            quote_volume: dec!(5000000),
            trades: 500,
            taker_buy_volume: dec!(50),
            taker_buy_quote_volume: dec!(2500000),
        }];
        write_candles_to_parquet(&candles, &path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_write_parquet_with_zstd() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ticks_zstd.parquet");
        let ticks = vec![make_tick(dec!(50000), dec!(1), Side::Buy)];
        write_ticks_to_parquet_with_compression(&ticks, &path, CompressionType::Zstd).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_tick_arrow_schema() {
        let schema = tick_arrow_schema();
        assert_eq!(schema.fields().len(), 7);
    }

    #[test]
    fn test_candle_arrow_schema() {
        let schema = candle_arrow_schema();
        assert_eq!(schema.fields().len(), 13);
    }

    #[test]
    fn test_empty_ticks_record_batch() {
        let ticks: Vec<Tick> = vec![];
        let batch = ticks_to_record_batch(&ticks).unwrap();
        assert_eq!(batch.num_rows(), 0);
    }

    #[test]
    fn test_quality_checker_gap_detection() {
        let mut checker = QualityChecker::new(5, dec!(50), Decimal::ZERO);

        let mut tick1 = make_tick(dec!(50000), dec!(1), Side::Buy);
        tick1.timestamp = chrono::DateTime::from_timestamp(1000, 0).unwrap();

        let mut tick2 = make_tick(dec!(50000), dec!(1), Side::Buy);
        tick2.timestamp = chrono::DateTime::from_timestamp(2000, 0).unwrap();

        checker.check(&tick1);
        let events = checker.check(&tick2);
        assert!(events.iter().any(|e| matches!(e, PipelineEvent::GapDetected { .. })));
    }

    #[test]
    fn test_quality_checker_outlier_detection() {
        let mut checker = QualityChecker::new(30, dec!(10), Decimal::ZERO);

        let tick1 = make_tick(dec!(50000), dec!(1), Side::Buy);
        let mut tick2 = make_tick(dec!(60000), dec!(1), Side::Buy);

        let now = chrono::Utc::now();
        let mut t1 = tick1.clone();
        t1.timestamp = now;
        tick2.timestamp = now + chrono::Duration::seconds(1);

        checker.check(&t1);
        let events = checker.check(&tick2);
        assert!(events.iter().any(|e| matches!(e, PipelineEvent::OutlierDetected { .. })));
    }

    #[test]
    fn test_file_rotator() {
        let mut rotator = FileRotator::new(1, 3600); // 1MB max
        assert!(!rotator.should_rotate(100));
        rotator.record_write(1024 * 1024); // Write 1MB
        assert!(rotator.should_rotate(1)); // Should rotate now
        rotator.rotate();
        assert_eq!(rotator.file_index(), 1);
    }

    #[test]
    fn test_file_rotator_time_based() {
        let mut rotator = FileRotator::new(100, 0); // 0 second interval = always rotate
        assert!(rotator.should_rotate(0));
    }

    #[test]
    fn test_file_rotator_path() {
        let rotator = FileRotator::new(100, 3600);
        let path = rotator.current_path("/data", "ticks");
        assert_eq!(path, "/data/ticks_0.parquet");
    }

    #[test]
    fn test_quality_stats() {
        let mut checker = QualityChecker::new(5, dec!(10), Decimal::ZERO);
        let tick = make_tick(dec!(50000), dec!(1), Side::Buy);
        checker.check(&tick);
        let stats = checker.stats();
        assert_eq!(stats.total_ticks, 1);
        assert_eq!(stats.total_gaps, 0);
        assert_eq!(stats.total_outliers, 0);
    }

    #[test]
    fn test_compression_type_display() {
        assert_eq!(CompressionType::Snappy.to_string(), "snappy");
        assert_eq!(CompressionType::Zstd.to_string(), "zstd");
        assert_eq!(CompressionType::None.to_string(), "none");
    }

    #[test]
    fn test_pipeline_get_quality_stats() {
        let config = PipelineConfig::default();
        let mut pipeline = DataPipeline::new(config);
        let tick = make_tick(dec!(50000), dec!(1), Side::Buy);
        pipeline.process_tick(tick);
        let stats = pipeline.get_quality_stats();
        assert_eq!(stats.total_ticks, 1);
    }
}
