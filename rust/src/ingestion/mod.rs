//! Data Ingestion Module
//!
//! WebSocket collectors, data normalization, and Arrow pipeline.

pub mod collector;
pub mod normalizer;
pub mod pipeline;

pub use collector::DataCollector;
pub use normalizer::DataNormalizer;
pub use pipeline::DataPipeline;
pub use pipeline::ticks_to_record_batch;
pub use pipeline::candles_to_record_batch;
pub use pipeline::write_ticks_to_parquet;
pub use pipeline::write_candles_to_parquet;
pub use pipeline::write_ticks_to_parquet_with_compression;
pub use pipeline::write_candles_to_parquet_with_compression;
pub use pipeline::CompressionType;
pub use pipeline::QualityStats;
