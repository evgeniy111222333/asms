//! PyO3 Bridge Module
//!
//! Zero-copy Python integration via Apache Arrow.

pub mod types;
pub mod bridge;

pub use bridge::init_module;
