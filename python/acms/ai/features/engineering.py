"""
ACMS AI Advanced Feature Engineering
=====================================

Sophisticated feature engineering for crypto market data in the
Algorithmic Crypto Management System. Covers market microstructure,
cross-asset relationships, temporal patterns, regime detection,
sentiment integration, and on-chain analytics.

Components
----------
AdvancedFeatureEngineer : Orchestrates all feature engineering modules
MarketMicrostructureFeatures : Orderbook, trade flow, VWAP features
CrossAssetFeatures : Pair correlation, lead-lag, relative strength
TemporalFeatures : Time-of-day, day-of-week, session effects
RegimeFeatures : Volatility, trend, liquidity regime classification
SentimentFeatures : NLP-derived sentiment feature integration
OnChainFeatures : Blockchain metrics (when available)
FeatureStabilityScorer : Feature stability over time
FeatureInteractionDetector : Detects feature interactions
"""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Feature Module
# ---------------------------------------------------------------------------

class FeatureModule(ABC):
    """Abstract base class for feature engineering modules.

    Each module computes a specific category of features from
    raw market data and returns a dict of feature_name → value.
    """

    @abstractmethod
    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute features from raw data.

        Parameters
        ----------
        data : dict
            Raw market data (prices, volumes, orderbook, etc.).

        Returns
        -------
        dict
            Mapping of feature name to computed value.
        """
        ...

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return the list of feature names this module computes."""
        ...


# ---------------------------------------------------------------------------
# Market Microstructure Features
# ---------------------------------------------------------------------------

class MarketMicrostructureFeatures(FeatureModule):
    """Features derived from order book and trade flow data.

    Computes orderbook imbalance, weighted mid-price, trade flow
    imbalance, VWAP deviation, bid-ask spread, and depth metrics.

    Expected input data keys
    ------------------------
    - ``bid_prices`` : array of bid price levels
    - ``bid_sizes`` : array of bid sizes at each level
    - ``ask_prices`` : array of ask price levels
    - ``ask_sizes`` : array of ask sizes at each level
    - ``trades`` : list of recent trades (price, size, side)
    - ``vwap`` : current VWAP value
    - ``last_price`` : current last traded price
    - ``volume`` : cumulative volume
    """

    def __init__(self, max_depth_levels: int = 10) -> None:
        self._max_depth = max_depth_levels
        logger.info("MarketMicrostructureFeatures initialized (depth=%d)", max_depth_levels)

    @property
    def feature_names(self) -> List[str]:
        return [
            "ob_imbalance", "ob_weighted_mid", "ob_spread_bps",
            "ob_bid_depth", "ob_ask_depth", "ob_depth_ratio",
            "trade_flow_imbalance", "trade_flow_velocity",
            "vwap_deviation", "vwap_ratio",
            "effective_spread", "realized_spread",
            "kyle_lambda", "amihud_illiquidity",
        ]

    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute market microstructure features."""
        features: Dict[str, float] = {}

        bid_prices = np.asarray(data.get("bid_prices", []), dtype=np.float64)
        bid_sizes = np.asarray(data.get("bid_sizes", []), dtype=np.float64)
        ask_prices = np.asarray(data.get("ask_prices", []), dtype=np.float64)
        ask_sizes = np.asarray(data.get("ask_sizes", []), dtype=np.float64)
        last_price = float(data.get("last_price", 0.0))
        vwap = float(data.get("vwap", 0.0))
        volume = float(data.get("volume", 1.0))

        # Orderbook imbalance
        if len(bid_sizes) > 0 and len(ask_sizes) > 0:
            total_bid = np.sum(bid_sizes[:self._max_depth])
            total_ask = np.sum(ask_sizes[:self._max_depth])
            total = total_bid + total_ask
            features["ob_imbalance"] = float((total_bid - total_ask) / total) if total > 0 else 0.0
        else:
            features["ob_imbalance"] = 0.0

        # Weighted mid-price
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            best_bid = bid_prices[0]
            best_ask = ask_prices[0]
            bid_size_0 = bid_sizes[0] if len(bid_sizes) > 0 else 1.0
            ask_size_0 = ask_sizes[0] if len(ask_sizes) > 0 else 1.0
            total_imb = bid_size_0 + ask_size_0
            features["ob_weighted_mid"] = float(
                (best_bid * ask_size_0 + best_ask * bid_size_0) / total_imb
            ) if total_imb > 0 else float((best_bid + best_ask) / 2)

            # Spread in basis points
            mid = (best_bid + best_ask) / 2
            features["ob_spread_bps"] = float((best_ask - best_bid) / mid * 10000) if mid > 0 else 0.0
        else:
            features["ob_weighted_mid"] = last_price
            features["ob_spread_bps"] = 0.0

        # Depth metrics
        features["ob_bid_depth"] = float(np.sum(bid_sizes[:self._max_depth])) if len(bid_sizes) > 0 else 0.0
        features["ob_ask_depth"] = float(np.sum(ask_sizes[:self._max_depth])) if len(ask_sizes) > 0 else 0.0
        ask_depth = features["ob_ask_depth"]
        features["ob_depth_ratio"] = (
            features["ob_bid_depth"] / ask_depth if ask_depth > 0 else 1.0
        )

        # Trade flow features
        trades = data.get("trades", [])
        if trades:
            buy_vol = sum(t.get("size", 0) for t in trades if t.get("side") == "buy")
            sell_vol = sum(t.get("size", 0) for t in trades if t.get("side") == "sell")
            total_vol = buy_vol + sell_vol
            features["trade_flow_imbalance"] = float((buy_vol - sell_vol) / total_vol) if total_vol > 0 else 0.0
            features["trade_flow_velocity"] = float(total_vol / max(len(trades), 1))
        else:
            features["trade_flow_imbalance"] = 0.0
            features["trade_flow_velocity"] = 0.0

        # VWAP features
        if vwap > 0 and last_price > 0:
            features["vwap_deviation"] = float((last_price - vwap) / vwap)
            features["vwap_ratio"] = float(last_price / vwap)
        else:
            features["vwap_deviation"] = 0.0
            features["vwap_ratio"] = 1.0

        # Effective spread (approximation)
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            mid = (bid_prices[0] + ask_prices[0]) / 2
            features["effective_spread"] = float(2 * abs(last_price - mid) / mid) if mid > 0 else 0.0
        else:
            features["effective_spread"] = 0.0

        # Realized spread (placeholder - needs trade data with timestamps)
        features["realized_spread"] = features.get("effective_spread", 0.0) * 0.8

        # Kyle's lambda (price impact coefficient)
        if len(trades) > 5 and last_price > 0:
            trade_sizes = np.array([t.get("size", 0) for t in trades])
            trade_sides = np.array([1 if t.get("side") == "buy" else -1 for t in trades])
            net_flow = np.sum(trade_sizes * trade_sides)
            price_change = float(data.get("price_change", 0.0))
            features["kyle_lambda"] = float(abs(price_change) / (abs(net_flow) + 1e-8))
        else:
            features["kyle_lambda"] = 0.0

        # Amihud illiquidity
        if volume > 0 and last_price > 0:
            price_change_abs = abs(float(data.get("price_change", 0.0)))
            features["amihud_illiquidity"] = float(price_change_abs / (volume * last_price + 1e-12))
        else:
            features["amihud_illiquidity"] = 0.0

        return features


# ---------------------------------------------------------------------------
# Cross-Asset Features
# ---------------------------------------------------------------------------

class CrossAssetFeatures(FeatureModule):
    """Features capturing relationships between crypto assets.

    Computes pair correlations, lead-lag indicators, relative
    strength, and beta coefficients between the target asset
    and reference assets.

    Expected input data keys
    ------------------------
    - ``target_returns`` : array of target asset returns
    - ``reference_returns`` : dict of asset_name → return array
    - ``target_price`` : current target price
    - ``reference_prices`` : dict of asset_name → price
    """

    def __init__(self, lookback: int = 60, reference_assets: Optional[List[str]] = None) -> None:
        self._lookback = lookback
        self._reference_assets = reference_assets or ["BTC", "ETH", "SOL"]
        logger.info("CrossAssetFeatures initialized (lookback=%d, refs=%s)", lookback, reference_assets)

    @property
    def feature_names(self) -> List[str]:
        names = ["cross_asset_avg_correlation", "cross_asset_max_correlation",
                 "cross_asset_min_correlation", "cross_asset_beta",
                 "relative_strength_index"]
        for asset in self._reference_assets:
            names.extend([
                f"corr_{asset.lower()}",
                f"lead_lag_{asset.lower()}",
                f"relative_strength_{asset.lower()}",
                f"beta_{asset.lower()}",
            ])
        return names

    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute cross-asset features."""
        features: Dict[str, float] = {}

        target_returns = np.asarray(data.get("target_returns", []), dtype=np.float64)
        target_price = float(data.get("target_price", 0.0))
        reference_returns = data.get("reference_returns", {})
        reference_prices = data.get("reference_prices", {})

        correlations: List[float] = []
        betas: List[float] = []

        for asset in self._reference_assets:
            ref_returns = np.asarray(reference_returns.get(asset, []), dtype=np.float64)
            ref_price = float(reference_prices.get(asset, 0.0))

            if len(target_returns) < 5 or len(ref_returns) < 5:
                features[f"corr_{asset.lower()}"] = 0.0
                features[f"lead_lag_{asset.lower()}"] = 0.0
                features[f"relative_strength_{asset.lower()}"] = 0.0
                features[f"beta_{asset.lower()}"] = 0.0
                continue

            # Align lengths
            min_len = min(len(target_returns), len(ref_returns))
            t_ret = target_returns[-min_len:]
            r_ret = ref_returns[-min_len:]

            # Pearson correlation
            corr = self._safe_corr(t_ret, r_ret)
            features[f"corr_{asset.lower()}"] = corr
            correlations.append(corr)

            # Lead-lag (cross-correlation at lag 1)
            if min_len > 5:
                lead_lag = self._compute_lead_lag(t_ret, r_ret)
                features[f"lead_lag_{asset.lower()}"] = lead_lag
            else:
                features[f"lead_lag_{asset.lower()}"] = 0.0

            # Relative strength
            if target_price > 0 and ref_price > 0:
                target_cum = float(np.prod(1 + t_ret[-self._lookback:]))
                ref_cum = float(np.prod(1 + r_ret[-self._lookback:]))
                features[f"relative_strength_{asset.lower()}"] = float(
                    target_cum / ref_cum - 1.0 if ref_cum > 0 else 0.0
                )
            else:
                features[f"relative_strength_{asset.lower()}"] = 0.0

            # Beta (regression coefficient)
            beta = self._compute_beta(t_ret, r_ret)
            features[f"beta_{asset.lower()}"] = beta
            betas.append(beta)

        # Aggregate features
        features["cross_asset_avg_correlation"] = float(np.mean(correlations)) if correlations else 0.0
        features["cross_asset_max_correlation"] = float(np.max(correlations)) if correlations else 0.0
        features["cross_asset_min_correlation"] = float(np.min(correlations)) if correlations else 0.0
        features["cross_asset_beta"] = float(np.mean(betas)) if betas else 0.0

        # Overall relative strength
        if target_price > 0:
            ref_avg = np.mean([float(reference_prices.get(a, target_price)) for a in self._reference_assets])
            features["relative_strength_index"] = float(target_price / ref_avg - 1.0) if ref_avg > 0 else 0.0
        else:
            features["relative_strength_index"] = 0.0

        return features

    @staticmethod
    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Pearson correlation safely."""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a < 1e-10 or std_b < 1e-10:
            return 0.0
        corr = np.corrcoef(a, b)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def _compute_lead_lag(target: np.ndarray, reference: np.ndarray) -> float:
        """Compute lead-lag indicator (cross-correlation at lag ±1)."""
        if len(target) < 5:
            return 0.0
        # Reference leads target by 1 period
        if len(reference) > 1 and len(target) > 1:
            ref_lead = reference[:-1]
            target_lag = target[1:]
            min_len = min(len(ref_lead), len(target_lag))
            corr_lead = np.corrcoef(ref_lead[:min_len], target_lag[:min_len])[0, 1]
            # Target leads reference
            target_lead = target[:-1]
            ref_lag = reference[1:]
            min_len2 = min(len(target_lead), len(ref_lag))
            corr_lag = np.corrcoef(target_lead[:min_len2], ref_lag[:min_len2])[0, 1]
            return float(corr_lead - corr_lag) if not np.isnan(corr_lead) and not np.isnan(corr_lag) else 0.0
        return 0.0

    @staticmethod
    def _compute_beta(target: np.ndarray, reference: np.ndarray) -> float:
        """Compute beta (OLS regression coefficient)."""
        if len(target) < 5 or len(reference) < 5:
            return 0.0
        ref_var = np.var(reference)
        if ref_var < 1e-10:
            return 0.0
        cov = np.cov(target, reference)[0, 1]
        return float(cov / ref_var) if not np.isnan(cov) else 0.0


# ---------------------------------------------------------------------------
# Temporal Features
# ---------------------------------------------------------------------------

class TemporalFeatures(FeatureModule):
    """Features capturing time-of-day, day-of-week, and session effects.

    Crypto markets trade 24/7, but distinct patterns exist around
    traditional market sessions and time periods.

    Expected input data keys
    ------------------------
    - ``timestamp`` : Unix timestamp of the current bar
    - ``hourly_volumes`` : array of average volumes per hour (24 values)
    """

    def __init__(self, timezone_offset: int = 0) -> None:
        self._tz_offset = timezone_offset
        logger.info("TemporalFeatures initialized (tz_offset=%d)", timezone_offset)

    @property
    def feature_names(self) -> List[str]:
        return [
            "time_of_day_sin", "time_of_day_cos",
            "day_of_week_sin", "day_of_week_cos",
            "is_weekend", "is_month_end",
            "session_asian", "session_european", "session_us",
            "session_overlap_eu_us", "session_overlap_as_eu",
            "hourly_volume_zscore", "bar_progress",
        ]

    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute temporal features."""
        features: Dict[str, float] = {}
        ts = float(data.get("timestamp", time.time()))

        # Apply timezone offset
        local_hour = (int(ts / 3600) + self._tz_offset) % 24
        day_of_week = int(ts / 86400) % 7  # 0=Thursday epoch, but consistent

        # Cyclical time-of-day encoding
        features["time_of_day_sin"] = float(math.sin(2 * math.pi * local_hour / 24))
        features["time_of_day_cos"] = float(math.cos(2 * math.pi * local_hour / 24))

        # Cyclical day-of-week encoding
        features["day_of_week_sin"] = float(math.sin(2 * math.pi * day_of_week / 7))
        features["day_of_week_cos"] = float(math.cos(2 * math.pi * day_of_week / 7))

        # Binary features
        features["is_weekend"] = 1.0 if day_of_week >= 5 else 0.0
        # Month end approximation (every ~30 days)
        day_of_month = int((ts % (86400 * 30)) / 86400)
        features["is_month_end"] = 1.0 if day_of_month >= 28 else 0.0

        # Session indicators (UTC-based)
        features["session_asian"] = 1.0 if 0 <= local_hour < 8 else 0.0
        features["session_european"] = 1.0 if 7 <= local_hour < 16 else 0.0
        features["session_us"] = 1.0 if 13 <= local_hour < 22 else 0.0
        features["session_overlap_eu_us"] = 1.0 if 13 <= local_hour < 16 else 0.0
        features["session_overlap_as_eu"] = 1.0 if 7 <= local_hour < 8 else 0.0

        # Hourly volume z-score
        hourly_volumes = np.asarray(data.get("hourly_volumes", []), dtype=np.float64)
        if len(hourly_volumes) == 24 and local_hour < 24:
            vol = hourly_volumes[local_hour]
            mean_vol = np.mean(hourly_volumes)
            std_vol = np.std(hourly_volumes)
            features["hourly_volume_zscore"] = float((vol - mean_vol) / std_vol) if std_vol > 0 else 0.0
        else:
            features["hourly_volume_zscore"] = 0.0

        # Bar progress (how far into the current hour)
        features["bar_progress"] = float((ts % 3600) / 3600)

        return features


# ---------------------------------------------------------------------------
# Regime Features
# ---------------------------------------------------------------------------

class RegimeFeatures(FeatureModule):
    """Features for detecting market regimes (volatility, trend, liquidity).

    Uses simple statistical thresholds and Hidden Markov Model-like
    state classification to identify the current market regime.

    Expected input data keys
    ------------------------
    - ``returns`` : array of recent returns
    - ``volumes`` : array of recent volumes
    - ``prices`` : array of recent prices
    - ``atr`` : current Average True Range value
    """

    def __init__(
        self,
        vol_lookback: int = 20,
        trend_lookback: int = 50,
        vol_regime_thresholds: Tuple[float, float] = (0.3, 0.7),
    ) -> None:
        self._vol_lookback = vol_lookback
        self._trend_lookback = trend_lookback
        self._vol_thresholds = vol_regime_thresholds
        logger.info("RegimeFeatures initialized (vol_lb=%d, trend_lb=%d)", vol_lookback, trend_lookback)

    @property
    def feature_names(self) -> List[str]:
        return [
            "volatility_regime", "volatility_percentile",
            "realized_volatility", "parkinson_volatility",
            "trend_regime", "trend_strength",
            "ad_score",  # AD line score
            "liquidity_regime", "volume_percentile",
            "regime_stability",
        ]

    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute regime features."""
        features: Dict[str, float] = {}

        returns = np.asarray(data.get("returns", []), dtype=np.float64)
        volumes = np.asarray(data.get("volumes", []), dtype=np.float64)
        prices = np.asarray(data.get("prices", []), dtype=np.float64)
        atr = float(data.get("atr", 0.0))

        # --- Volatility regime ---
        if len(returns) >= self._vol_lookback:
            recent_returns = returns[-self._vol_lookback:]
            realized_vol = float(np.std(recent_returns) * np.sqrt(252))  # Annualized
            features["realized_volatility"] = realized_vol

            # Parkinson volatility (using high-low range)
            if len(prices) >= self._vol_lookback:
                # Approximate using price range
                price_window = prices[-self._vol_lookback:]
                highs = np.maximum.accumulate(price_window)
                lows = np.minimum.accumulate(price_window)
                hl_ratio = np.log(highs / (lows + 1e-8))
                parkinson_vol = float(np.sqrt(np.mean(hl_ratio ** 2) / (4 * np.log(2))) * np.sqrt(252))
                features["parkinson_volatility"] = parkinson_vol
            else:
                features["parkinson_volatility"] = realized_vol

            # Volatility percentile (relative to longer history)
            if len(returns) >= 100:
                long_vol = float(np.std(returns) * np.sqrt(252))
                features["volatility_percentile"] = float(realized_vol / (long_vol + 1e-8))
            else:
                features["volatility_percentile"] = 1.0

            # Classify regime
            pct = features["volatility_percentile"]
            if pct < self._vol_thresholds[0]:
                features["volatility_regime"] = 0.0  # Low vol
            elif pct > self._vol_thresholds[1]:
                features["volatility_regime"] = 2.0  # High vol
            else:
                features["volatility_regime"] = 1.0  # Normal vol
        else:
            features["realized_volatility"] = 0.0
            features["parkinson_volatility"] = 0.0
            features["volatility_percentile"] = 1.0
            features["volatility_regime"] = 1.0

        # --- Trend regime ---
        if len(prices) >= self._trend_lookback:
            price_window = prices[-self._trend_lookback:]
            # Simple linear regression slope
            x = np.arange(len(price_window))
            slope, _ = np.polyfit(x, price_window, 1)
            # Normalize by price level
            trend_strength = float(slope / (np.mean(price_window) + 1e-8))
            features["trend_strength"] = trend_strength

            # AD score (accumulation/distribution)
            if len(returns) > 0 and len(volumes) > 0:
                min_len = min(len(returns), len(volumes))
                ad_flow = returns[-min_len:] * volumes[-min_len:]
                features["ad_score"] = float(np.sum(ad_flow) / (np.sum(volumes[-min_len:]) + 1e-8))
            else:
                features["ad_score"] = 0.0

            # Classify trend
            if abs(trend_strength) < 1e-5:
                features["trend_regime"] = 0.0  # Range-bound
            elif trend_strength > 0:
                features["trend_regime"] = 1.0  # Uptrend
            else:
                features["trend_regime"] = -1.0  # Downtrend
        else:
            features["trend_strength"] = 0.0
            features["ad_score"] = 0.0
            features["trend_regime"] = 0.0

        # --- Liquidity regime ---
        if len(volumes) >= 20:
            recent_vol = volumes[-20:]
            long_vol = volumes[-min(100, len(volumes)):] if len(volumes) >= 20 else recent_vol
            vol_pct = float(np.mean(recent_vol) / (np.mean(long_vol) + 1e-8))
            features["volume_percentile"] = vol_pct
            features["liquidity_regime"] = 0.0 if vol_pct < 0.5 else (2.0 if vol_pct > 1.5 else 1.0)
        else:
            features["volume_percentile"] = 1.0
            features["liquidity_regime"] = 1.0

        # --- Regime stability ---
        features["regime_stability"] = self._compute_regime_stability(returns, volumes)

        return features

    @staticmethod
    def _compute_regime_stability(returns: np.ndarray, volumes: np.ndarray) -> float:
        """Compute how stable the current regime is (0-1, higher = more stable)."""
        if len(returns) < 20:
            return 0.5
        # Split recent history into chunks and check volatility consistency
        chunk_size = 10
        n_chunks = len(returns) // chunk_size
        if n_chunks < 2:
            return 0.5
        chunk_vols = []
        for i in range(n_chunks):
            chunk = returns[i * chunk_size : (i + 1) * chunk_size]
            chunk_vols.append(float(np.std(chunk)))
        vol_cv = float(np.std(chunk_vols) / (np.mean(chunk_vols) + 1e-8))
        return float(max(0.0, 1.0 - vol_cv))


# ---------------------------------------------------------------------------
# Sentiment Features
# ---------------------------------------------------------------------------

class SentimentFeatures(FeatureModule):
    """Features derived from NLP sentiment analysis output.

    Integrates sentiment scores from news, social media, and
    other text sources into trading features.

    Expected input data keys
    ------------------------
    - ``news_sentiment`` : float, aggregated news sentiment score
    - ``social_sentiment`` : float, aggregated social media sentiment
    - ``fear_greed_index`` : float, fear & greed index value
    - ``sentiment_momentum`` : float, change in sentiment
    - ``mention_count`` : int, number of recent mentions
    - ``sentiment_distribution`` : dict, distribution of sentiment labels
    """

    def __init__(self, decay_factor: float = 0.95) -> None:
        self._decay = decay_factor
        self._sentiment_ema: float = 0.0
        self._momentum_ema: float = 0.0
        logger.info("SentimentFeatures initialized (decay=%.2f)", decay_factor)

    @property
    def feature_names(self) -> List[str]:
        return [
            "news_sentiment", "social_sentiment",
            "fear_greed_index", "combined_sentiment",
            "sentiment_momentum", "sentiment_divergence",
            "mention_intensity", "sentiment_volatility",
            "bullish_ratio", "sentiment_ema",
        ]

    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute sentiment features."""
        features: Dict[str, float] = {}

        news_sent = float(data.get("news_sentiment", 0.0))
        social_sent = float(data.get("social_sentiment", 0.0))
        fear_greed = float(data.get("fear_greed_index", 50.0))
        sent_momentum = float(data.get("sentiment_momentum", 0.0))
        mention_count = int(data.get("mention_count", 0))
        sent_dist = data.get("sentiment_distribution", {})

        features["news_sentiment"] = news_sent
        features["social_sentiment"] = social_sent
        features["fear_greed_index"] = (fear_greed - 50.0) / 50.0  # Normalize to [-1, 1]

        # Combined sentiment (weighted average)
        features["combined_sentiment"] = float(0.5 * news_sent + 0.3 * social_sent + 0.2 * features["fear_greed_index"])

        # Momentum
        features["sentiment_momentum"] = sent_momentum

        # Divergence between news and social
        features["sentiment_divergence"] = float(abs(news_sent - social_sent))

        # Mention intensity (log-scaled)
        features["mention_intensity"] = float(math.log1p(mention_count))

        # Sentiment volatility (using distribution)
        pos = float(sent_dist.get("positive", 0.0))
        neg = float(sent_dist.get("negative", 0.0))
        neutral = float(sent_dist.get("neutral", 0.0))
        total = pos + neg + neutral
        features["bullish_ratio"] = float(pos / total) if total > 0 else 0.5
        # Entropy-based volatility
        probs = np.array([pos, neg, neutral]) / total if total > 0 else np.array([0.33, 0.33, 0.34])
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        features["sentiment_volatility"] = float(1.0 - entropy / math.log(3))  # Normalized

        # EMA of combined sentiment
        self._sentiment_ema = self._decay * self._sentiment_ema + (1 - self._decay) * features["combined_sentiment"]
        features["sentiment_ema"] = self._sentiment_ema

        return features


# ---------------------------------------------------------------------------
# On-Chain Features
# ---------------------------------------------------------------------------

class OnChainFeatures(FeatureModule):
    """Features derived from blockchain on-chain data.

    Computes metrics from blockchain activity such as active addresses,
    transaction volumes, hash rate, and supply dynamics.

    Expected input data keys
    ------------------------
    - ``active_addresses`` : int, recent active address count
    - ``transaction_count`` : int, recent transaction count
    - ``hash_rate`` : float, current network hash rate
    - ``exchange_inflow`` : float, recent exchange inflow volume
    - ``exchange_outflow`` : float, recent exchange outflow volume
    - ``supply_rate`` : float, current inflation/supply growth rate
    - ``nvt_ratio`` : float, Network Value to Transaction ratio
    - ``mempool_size`` : int, current mempool transaction count
    """

    def __init__(self, lookback: int = 7) -> None:
        self._lookback = lookback
        logger.info("OnChainFeatures initialized (lookback=%d days)", lookback)

    @property
    def feature_names(self) -> List[str]:
        return [
            "active_addresses_zscore", "transaction_count_zscore",
            "hash_rate_change", "exchange_net_flow",
            "exchange_flow_ratio", "supply_growth_rate",
            "nvt_ratio_normalized", "mempool_congestion",
            "on_chain_momentum", "whale_activity_score",
        ]

    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute on-chain features."""
        features: Dict[str, float] = {}

        active_addr = float(data.get("active_addresses", 0))
        tx_count = float(data.get("transaction_count", 0))
        hash_rate = float(data.get("hash_rate", 0))
        ex_inflow = float(data.get("exchange_inflow", 0))
        ex_outflow = float(data.get("exchange_outflow", 0))
        supply_rate = float(data.get("supply_rate", 0))
        nvt = float(data.get("nvt_ratio", 0))
        mempool = int(data.get("mempool_size", 0))

        # Historical context for z-scores (using provided baselines or defaults)
        addr_mean = float(data.get("active_addresses_mean", active_addr))
        addr_std = float(data.get("active_addresses_std", 1))
        tx_mean = float(data.get("transaction_count_mean", tx_count))
        tx_std = float(data.get("transaction_count_std", 1))

        features["active_addresses_zscore"] = float((active_addr - addr_mean) / (addr_std + 1e-8))
        features["transaction_count_zscore"] = float((tx_count - tx_mean) / (tx_std + 1e-8))

        # Hash rate change
        hash_rate_prev = float(data.get("hash_rate_prev", hash_rate))
        features["hash_rate_change"] = float((hash_rate - hash_rate_prev) / (hash_rate_prev + 1e-8))

        # Exchange flow
        net_flow = ex_outflow - ex_inflow
        total_flow = ex_inflow + ex_outflow
        features["exchange_net_flow"] = float(net_flow / (total_flow + 1e-8))
        features["exchange_flow_ratio"] = float(ex_outflow / (ex_inflow + 1e-8))

        # Supply growth
        features["supply_growth_rate"] = supply_rate

        # NVT ratio (normalize)
        features["nvt_ratio_normalized"] = float(math.log1p(max(0, nvt)))

        # Mempool congestion
        mempool_baseline = float(data.get("mempool_baseline", 5000))
        features["mempool_congestion"] = float(min(2.0, mempool / (mempool_baseline + 1e-8)))

        # On-chain momentum (composite)
        features["on_chain_momentum"] = float(
            0.3 * features["active_addresses_zscore"]
            + 0.2 * features["transaction_count_zscore"]
            + 0.2 * features["hash_rate_change"]
            + 0.3 * features["exchange_net_flow"]
        )

        # Whale activity (placeholder - requires whale-specific data)
        whale_inflow = float(data.get("whale_exchange_inflow", 0))
        whale_baseline = float(data.get("whale_exchange_inflow_mean", 1))
        features["whale_activity_score"] = float(whale_inflow / (whale_baseline + 1e-8))

        return features


# ---------------------------------------------------------------------------
# Feature Stability Scorer
# ---------------------------------------------------------------------------

class FeatureStabilityScorer:
    """Scores feature stability over time windows.

    Measures how consistent a feature's distribution remains across
    successive time windows using Population Stability Index (PSI)
    and Kolmogorov-Smirnov-like statistics.

    Parameters
    ----------
    window_size : int
        Number of observations per window.
    psi_threshold : float
        PSI value above which a feature is considered unstable.
    """

    def __init__(self, window_size: int = 500, psi_threshold: float = 0.2) -> None:
        self._window_size = window_size
        self._psi_threshold = psi_threshold
        self._windows: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._scores: Dict[str, float] = {}
        logger.info("FeatureStabilityScorer initialized (window=%d, psi_threshold=%.2f)", window_size, psi_threshold)

    def add_observations(self, feature_name: str, values: np.ndarray) -> None:
        """Add observations for a feature."""
        self._windows[feature_name].append(values.copy())
        # Keep only recent windows
        if len(self._windows[feature_name]) > 10:
            self._windows[feature_name] = self._windows[feature_name][-10:]

    def compute_stability(self, feature_name: str) -> Dict[str, float]:
        """Compute stability metrics for a feature.

        Returns
        -------
        dict
            Stability metrics including PSI, KS statistic, and stability score.
        """
        windows = self._windows.get(feature_name, [])
        if len(windows) < 2:
            return {"psi": 0.0, "ks_stat": 0.0, "stability_score": 1.0}

        reference = np.concatenate(windows[:-1])
        current = windows[-1]

        psi = self._compute_psi(reference, current)
        ks = self._compute_ks(reference, current)

        stability = max(0.0, 1.0 - psi / self._psi_threshold)
        self._scores[feature_name] = stability

        return {
            "psi": float(psi),
            "ks_stat": float(ks),
            "stability_score": float(stability),
            "is_stable": psi < self._psi_threshold,
        }

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index."""
        # Create bins from reference distribution
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        # Normalize to probabilities
        ref_pct = ref_hist / (len(reference) + 1e-8)
        cur_pct = cur_hist / (len(current) + 1e-8)

        # PSI formula
        psi = 0.0
        for p, q in zip(ref_pct, cur_pct):
            if p > 0 and q > 0:
                psi += (p - q) * np.log(p / q)
        return psi

    @staticmethod
    def _compute_ks(reference: np.ndarray, current: np.ndarray) -> float:
        """Compute Kolmogorov-Smirnov statistic (simplified)."""
        ref_sorted = np.sort(reference)
        cur_sorted = np.sort(current)
        all_vals = np.sort(np.concatenate([ref_sorted, cur_sorted]))

        n_ref = len(ref_sorted)
        n_cur = len(cur_sorted)

        max_diff = 0.0
        for val in all_vals[::max(1, len(all_vals) // 100)]:
            ref_cdf = np.searchsorted(ref_sorted, val, side="right") / n_ref
            cur_cdf = np.searchsorted(cur_sorted, val, side="right") / n_cur
            diff = abs(ref_cdf - cur_cdf)
            max_diff = max(max_diff, diff)

        return max_diff

    def get_all_scores(self) -> Dict[str, float]:
        """Return stability scores for all tracked features."""
        return dict(self._scores)


# ---------------------------------------------------------------------------
# Feature Interaction Detector
# ---------------------------------------------------------------------------

class FeatureInteractionDetector:
    """Detects significant feature interactions.

    Uses pairwise correlation, mutual information, and interaction
    strength metrics to identify features that have meaningful
    joint effects on the target.

    Parameters
    ----------
    max_interactions : int
        Maximum number of interactions to track.
    correlation_threshold : float
        Minimum absolute correlation to consider.
    """

    def __init__(self, max_interactions: int = 50, correlation_threshold: float = 0.3) -> None:
        self._max_interactions = max_interactions
        self._corr_threshold = correlation_threshold
        self._interactions: List[Dict[str, Any]] = []
        logger.info("FeatureInteractionDetector initialized (max=%d)", max_interactions)

    def detect(self, feature_matrix: np.ndarray, feature_names: List[str],
               target: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Detect feature interactions.

        Parameters
        ----------
        feature_matrix : np.ndarray
            Shape (n_samples, n_features).
        feature_names : list of str
            Column names corresponding to feature_matrix.
        target : np.ndarray, optional
            Target values for supervised interaction detection.

        Returns
        -------
        list of dict
            Detected interactions with strength metrics.
        """
        n_features = feature_matrix.shape[1]
        interactions: List[Dict[str, Any]] = []

        # Pairwise correlations
        for i in range(n_features):
            for j in range(i + 1, n_features):
                col_i = feature_matrix[:, i]
                col_j = feature_matrix[:, j]

                # Skip if either column has zero variance
                if np.std(col_i) < 1e-10 or np.std(col_j) < 1e-10:
                    continue

                corr = float(np.corrcoef(col_i, col_j)[0, 1])
                if np.isnan(corr):
                    continue

                if abs(corr) >= self._corr_threshold:
                    interaction: Dict[str, Any] = {
                        "feature_a": feature_names[i],
                        "feature_b": feature_names[j],
                        "correlation": corr,
                        "interaction_type": "positive" if corr > 0 else "negative",
                    }

                    # If target is available, compute interaction effect
                    if target is not None:
                        product = col_i * col_j
                        if np.std(product) > 1e-10 and np.std(target) > 1e-10:
                            interaction_corr = float(np.corrcoef(product, target)[0, 1])
                            if not np.isnan(interaction_corr):
                                interaction["interaction_effect"] = interaction_corr

                    interactions.append(interaction)

        # Sort by absolute correlation
        interactions.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        self._interactions = interactions[: self._max_interactions]
        return self._interactions

    @property
    def top_interactions(self) -> List[Dict[str, Any]]:
        """Return the top detected interactions."""
        return list(self._interactions)


# ---------------------------------------------------------------------------
# Advanced Feature Engineer
# ---------------------------------------------------------------------------

class AdvancedFeatureEngineer:
    """Orchestrates all feature engineering modules.

    Provides a unified interface for computing all feature categories
    and combining them into a feature vector for model input.

    Parameters
    ----------
    enable_microstructure : bool
        Whether to compute market microstructure features.
    enable_cross_asset : bool
        Whether to compute cross-asset features.
    enable_temporal : bool
        Whether to compute temporal features.
    enable_regime : bool
        Whether to compute regime features.
    enable_sentiment : bool
        Whether to compute sentiment features.
    enable_onchain : bool
        Whether to compute on-chain features.
    timezone_offset : int
        UTC offset for temporal features.

    Examples
    --------
    >>> engineer = AdvancedFeatureEngineer()
    >>> features = engineer.compute(market_data)
    >>> vector = engineer.compute_vector(market_data)
    """

    def __init__(
        self,
        enable_microstructure: bool = True,
        enable_cross_asset: bool = True,
        enable_temporal: bool = True,
        enable_regime: bool = True,
        enable_sentiment: bool = True,
        enable_onchain: bool = False,
        timezone_offset: int = 0,
    ) -> None:
        self._modules: Dict[str, FeatureModule] = {}

        if enable_microstructure:
            self._modules["microstructure"] = MarketMicrostructureFeatures()
        if enable_cross_asset:
            self._modules["cross_asset"] = CrossAssetFeatures()
        if enable_temporal:
            self._modules["temporal"] = TemporalFeatures(timezone_offset=timezone_offset)
        if enable_regime:
            self._modules["regime"] = RegimeFeatures()
        if enable_sentiment:
            self._modules["sentiment"] = SentimentFeatures()
        if enable_onchain:
            self._modules["onchain"] = OnChainFeatures()

        self._stability_scorer = FeatureStabilityScorer()
        self._interaction_detector = FeatureInteractionDetector()
        self._feature_names: Optional[List[str]] = None

        logger.info(
            "AdvancedFeatureEngineer initialized with modules: %s",
            list(self._modules.keys()),
        )

    def compute(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Compute all features from raw market data.

        Parameters
        ----------
        data : dict
            Raw market data (must contain relevant keys for each module).

        Returns
        -------
        dict
            Flat mapping of feature name to computed value.
        """
        all_features: Dict[str, float] = {}

        for module_name, module in self._modules.items():
            try:
                features = module.compute(data)
                all_features.update(features)
            except Exception as exc:
                logger.error("Feature module '%s' error: %s", module_name, exc)
                # Fill with zeros on failure
                for fname in module.feature_names:
                    if fname not in all_features:
                        all_features[fname] = 0.0

        return all_features

    def compute_vector(self, data: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Compute features and return as an ordered vector.

        Returns
        -------
        tuple of (np.ndarray, list of str)
            Feature vector and corresponding feature names.
        """
        features = self.compute(data)
        names = self.get_feature_names()
        vector = np.array([features.get(n, 0.0) for n in names], dtype=np.float32)
        return vector, names

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of all feature names."""
        if self._feature_names is not None:
            return self._feature_names

        names: List[str] = []
        for module in self._modules.values():
            names.extend(module.feature_names)
        self._feature_names = names
        return names

    @property
    def feature_count(self) -> int:
        """Total number of features computed."""
        return len(self.get_feature_names())

    def get_module(self, name: str) -> Optional[FeatureModule]:
        """Get a specific feature module by name."""
        return self._modules.get(name)

    @property
    def stability_scorer(self) -> FeatureStabilityScorer:
        return self._stability_scorer

    @property
    def interaction_detector(self) -> FeatureInteractionDetector:
        return self._interaction_detector
