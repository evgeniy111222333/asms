"""Pipeline configuration."""

from dataclasses import dataclass


class PipelineConfig:
    """Data pipeline configuration."""
    data_dir: str = "/data/acms"
    parquet_dir: str = "/data/acms/parquet"
    default_exchange: str = "binance"
    download_batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    quality_check_enabled: bool = True
    outlier_std_threshold: float = 5.0
    gap_fill_method: str = "ffill"



__all__ = ["PipelineConfig"]
