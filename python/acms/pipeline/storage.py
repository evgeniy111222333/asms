"""Parquet file storage for historical market data."""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Any

logger = logging.getLogger(__name__)


class ParquetStorage:
    """Parquet file storage for historical market data.

    Provides efficient read/write operations for large
    historical datasets using Parquet format.
    """

    def __init__(self, base_dir: str = "/data/acms/parquet"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, symbol: str, timeframe: str, exchange: str = "") -> Path:
        """Get the parquet file path for a symbol/timeframe/exchange."""
        parts = [exchange, symbol.replace("/", "_"), timeframe]
        return self.base_dir / "/".join(parts) / "data.parquet"

    def write(self, data: Any, symbol: str, timeframe: str,
              exchange: str = "") -> str:
        """Write data to a Parquet file.

        Args:
            data: DataFrame or dict to write.
            symbol: Trading pair symbol.
            timeframe: Data timeframe.
            exchange: Exchange name.

        Returns:
            Path to written file.
        """
        path = self._get_path(symbol, timeframe, exchange)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import polars as pl
            if isinstance(data, dict):
                df = pl.DataFrame(data)
            elif isinstance(data, pl.DataFrame):
                df = data
            else:
                df = pl.DataFrame(data)
            df.write_parquet(str(path))
            return str(path)
        except ImportError:
            # Fallback: save as CSV
            csv_path = path.with_suffix('.csv')
            if isinstance(data, dict):
                import csv
                keys = list(data.keys())
                rows = zip(*[data[k] for k in keys])
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(keys)
                    writer.writerows(rows)
            return str(csv_path)

    def read(self, symbol: str, timeframe: str, exchange: str = "",
             columns: Optional[List[str]] = None,
             start_date: Optional[str] = None,
             end_date: Optional[str] = None) -> Any:
        """Read data from a Parquet file.

        Args:
            symbol: Trading pair symbol.
            timeframe: Data timeframe.
            exchange: Exchange name.
            columns: Optional list of columns to read.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            Polars DataFrame or dict.
        """
        path = self._get_path(symbol, timeframe, exchange)

        try:
            import polars as pl
            if path.exists():
                df = pl.read_parquet(str(path), columns=columns)
                if start_date and "open_time" in df.columns:
                    df = df.filter(pl.col("open_time") >= start_date)
                if end_date and "open_time" in df.columns:
                    df = df.filter(pl.col("open_time") <= end_date)
                return df
            else:
                return pl.DataFrame()
        except ImportError:
            # Fallback: read CSV
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                data = {}
                import csv
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for k, v in row.items():
                            if k not in data:
                                data[k] = []
                            data[k].append(v)
                return data
            return {}

    def exists(self, symbol: str, timeframe: str, exchange: str = "") -> bool:
        """Check if data exists for a symbol/timeframe."""
        path = self._get_path(symbol, timeframe, exchange)
        return path.exists() or path.with_suffix('.csv').exists()

    def list_symbols(self, exchange: str = "") -> List[str]:
        """List all symbols with stored data."""
        exchange_dir = self.base_dir / exchange if exchange else self.base_dir
        if not exchange_dir.exists():
            return []
        symbols = []
        for d in exchange_dir.iterdir():
            if d.is_dir():
                symbols.append(d.name.replace("_", "/"))
        return sorted(symbols)


__all__ = ["ParquetStorage"]
