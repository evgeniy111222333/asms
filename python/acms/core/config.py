"""Core configuration for ACMS."""

from dataclasses import dataclass, field


@dataclass
class ACMSConfig:
    """Main ACMS configuration."""
    # Database
    db_url: str = "postgresql://acms:acms@localhost:5432/acms"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Redpanda
    redpanda_brokers: list[str] = field(default_factory=lambda: ["localhost:9092"])

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_expiry_hours: int = 24
    api_key_length: int = 32

    # Risk
    max_position_per_symbol: float = 100000.0
    max_total_position: float = 1000000.0
    max_order_notional: float = 50000.0
    max_daily_drawdown: float = 0.05
    max_drawdown: float = 0.20
    max_orders_per_second: int = 10
    max_orders_per_minute: int = 100

    # Exchanges
    binance_api_key: str = ""
    binance_api_secret: str = ""
    bybit_api_key: str = ""
    bybit_api_secret: str = ""
    okx_api_key: str = ""
    okx_api_secret: str = ""
    okx_passphrase: str = ""

    # Data
    data_dir: str = "/data/acms"
    parquet_dir: str = "/data/acms/parquet"

    # ML
    ml_model_dir: str = "/data/acms/models"
    ml_training_enabled: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: str = "/data/acms/logs/acms.log"


__all__ = [
    "ACMSConfig",
]
