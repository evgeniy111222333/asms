"""Complete data pipeline engine."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import numpy as np

from acms.pipeline.config import PipelineConfig
from acms.pipeline.quality import DataQualityChecker
from acms.pipeline.resampler import DataResampler
from acms.pipeline.windowing import DataWindowing
from acms.pipeline.storage import ParquetStorage

logger = logging.getLogger(__name__)


class DataPipeline:
    """Complete data pipeline: fetch → validate → transform → store.

    Orchestrates the full data lifecycle from exchange download
    through quality checks to persistent storage.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.quality_checker = DataQualityChecker(
            outlier_std_threshold=self.config.outlier_std_threshold,
            gap_fill_method=self.config.gap_fill_method,
        )
        self.resampler = DataResampler()
        self.windowing = DataWindowing()
        self.storage = ParquetStorage(base_dir=self.config.parquet_dir)
        self._exchange_adapter = None

    def set_exchange(self, adapter: Any) -> None:
        """Set the exchange adapter for data fetching.

        Args:
            adapter: ExchangeAdapter instance with get_candles method.
        """
        self._exchange_adapter = adapter

    async def download_historical(self, symbol: str, timeframe: str,
                                   start_date: str, end_date: str,
                                   exchange: Optional[str] = None) -> List[Dict]:
        """Download historical candle data from exchange.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            exchange: Exchange name override.

        Returns:
            List of candle dicts.
        """
        if not self._exchange_adapter:
            logger.error("No exchange adapter configured")
            return []

        all_candles = []
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start

        while current < end:
            try:
                candles = await self._exchange_adapter.get_candles(
                    symbol, timeframe, limit=self.config.download_batch_size,
                )
                for c in candles:
                    all_candles.append({
                        "open_time": c.open_time.isoformat() if hasattr(c, 'open_time') else str(c.open_time),
                        "close_time": c.close_time.isoformat() if hasattr(c, 'close_time') else str(c.close_time),
                        "open": c.open, "high": c.high, "low": c.low,
                        "close": c.close, "volume": c.volume,
                        "quote_volume": c.quote_volume if hasattr(c, 'quote_volume') else 0,
                        "trades": c.trades if hasattr(c, 'trades') else 0,
                    })

                if not candles:
                    break

                # Move forward by the time range of fetched candles
                last_time = candles[-1].close_time if candles else current
                if hasattr(last_time, '__add__'):
                    current = last_time + timedelta(minutes=1)
                else:
                    current = end

                await asyncio.sleep(0.5)  # Rate limit

            except Exception as e:
                logger.error("Download error: %s", e)
                if self.config.max_retries > 0:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                break

        return all_candles

    async def run_pipeline(self, symbol: str, timeframe: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           quality_check: bool = True,
                           store: bool = True) -> Dict:
        """Run the full data pipeline.

        Args:
            symbol: Trading pair symbol.
            timeframe: Candle timeframe.
            start_date: Optional start date for download.
            end_date: Optional end date for download.
            quality_check: Whether to run quality checks.
            store: Whether to store results.

        Returns:
            Dict with pipeline results and quality report.
        """
        result = {"symbol": symbol, "timeframe": timeframe, "status": "pending"}

        # Step 1: Fetch data
        if start_date and end_date:
            candles = await self.download_historical(symbol, timeframe, start_date, end_date)
        elif self._exchange_adapter:
            try:
                raw_candles = await self._exchange_adapter.get_candles(symbol, timeframe)
                candles = [
                    {"open": c.open, "high": c.high, "low": c.low,
                     "close": c.close, "volume": c.volume}
                    for c in raw_candles
                ]
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                return result
        else:
            result["status"] = "error"
            result["error"] = "No data source"
            return result

        result["raw_count"] = len(candles)

        if not candles:
            result["status"] = "no_data"
            return result

        # Step 2: Convert to arrays for quality checking
        data = self._candles_to_arrays(candles)

        # Step 3: Quality checks
        quality_report = {}
        if quality_check and self.config.quality_check_enabled:
            # Check missing data
            quality_report["missing"] = self.quality_checker.check_missing(data)

            # Filter outliers
            data = self.quality_checker.filter_outliers(data)

            # Fill gaps
            data = self.quality_checker.fill_gaps(data)

        result["quality_report"] = quality_report

        # Step 4: Store
        if store:
            try:
                path = self.storage.write(data, symbol, timeframe)
                result["storage_path"] = path
            except Exception as e:
                logger.warning("Storage error: %s", e)

        result["status"] = "success"
        result["final_count"] = len(candles)
        return result

    @staticmethod
    def _candles_to_arrays(candles: List[Dict]) -> Dict[str, np.ndarray]:
        """Convert list of candle dicts to columnar arrays.

        Args:
            candles: List of candle dicts.

        Returns:
            Dict mapping column names to numpy arrays.
        """
        if not candles:
            return {}

        columns = {}
        for key in ["open", "high", "low", "close", "volume"]:
            values = []
            for c in candles:
                val = c.get(key, 0)
                values.append(float(val) if val is not None else np.nan)
            columns[key] = np.array(values, dtype=np.float64)

        return columns


__all__ = ["DataPipeline"]
