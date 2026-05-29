"""Data resampling between timeframes."""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class DataResampler:
    """Resample market data between timeframes.

    Converts OHLCV data from one timeframe to another,
    properly aggregating OHLCV fields.
    """

    @staticmethod
    def resample_candles(candles: List[Dict], source_tf: str,
                          target_tf: str) -> List[Dict]:
        """Resample candle data to a different timeframe.

        Args:
            candles: List of candle dicts with OHLCV fields.
            source_tf: Source timeframe (e.g., '1m', '5m').
            target_tf: Target timeframe (e.g., '5m', '1h').

        Returns:
            List of resampled candle dicts.
        """
        if not candles:
            return []

        # Parse timeframe to minutes
        source_minutes = DataResampler._parse_timeframe(source_tf)
        target_minutes = DataResampler._parse_timeframe(target_tf)

        if source_minutes <= 0 or target_minutes <= 0:
            logger.warning("Invalid timeframe: %s -> %s", source_tf, target_tf)
            return candles

        if target_minutes <= source_minutes:
            logger.warning("Target timeframe must be larger than source")
            return candles

        ratio = target_minutes // source_minutes
        if ratio < 2:
            return candles

        resampled = []
        for i in range(0, len(candles), ratio):
            batch = candles[i:i + ratio]
            if not batch:
                break

            resampled.append({
                "open_time": batch[0].get("open_time"),
                "close_time": batch[-1].get("close_time"),
                "open": batch[0].get("open", 0),
                "high": max(c.get("high", 0) for c in batch),
                "low": min(c.get("low", 0) for c in batch),
                "close": batch[-1].get("close", 0),
                "volume": sum(c.get("volume", 0) for c in batch),
                "quote_volume": sum(c.get("quote_volume", 0) for c in batch),
                "trades": sum(c.get("trades", 0) for c in batch),
            })

        return resampled

    @staticmethod
    def _parse_timeframe(tf: str) -> int:
        """Parse timeframe string to minutes.

        Args:
            tf: Timeframe string (e.g., '1m', '5m', '1h', '4h', '1d').

        Returns:
            Number of minutes.
        """
        tf = tf.lower().strip()
        if tf.endswith('m'):
            return int(tf[:-1])
        elif tf.endswith('h'):
            return int(tf[:-1]) * 60
        elif tf.endswith('d'):
            return int(tf[:-1]) * 1440
        elif tf.endswith('w'):
            return int(tf[:-1]) * 10080
        return 0



__all__ = ["DataResampler"]
