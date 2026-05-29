"""Technical Indicators for ACMS."""

import numpy as np
from typing import Optional, Tuple, List, Dict


class CandlestickPatterns:
    """Comprehensive candlestick pattern recognition.

    Detects 35+ single, double, and triple candlestick patterns.
    All methods return a dict with pattern name -> bool mapping.
    """

    def __init__(self, body_ratio: float = 0.1, doji_ratio: float = 0.05,
                 shadow_ratio: float = 0.1):
        """Initialize CandlestickPatterns.

        Args:
            body_ratio: Minimum body/range ratio for significant candles.
            doji_ratio: Maximum body/range ratio for doji candles.
            shadow_ratio: Minimum shadow/range ratio for significance.
        """
        self.body_ratio = body_ratio
        self.doji_ratio = doji_ratio
        self.shadow_ratio = shadow_ratio

    def _body(self, o: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute candle body sizes."""
        return np.abs(c - o)

    def _range(self, h: np.ndarray, l: np.ndarray) -> np.ndarray:
        """Compute candle ranges."""
        return h - l

    def _upper_shadow(self, h: np.ndarray, o: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute upper shadows."""
        return h - np.maximum(o, c)

    def _lower_shadow(self, l: np.ndarray, o: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Compute lower shadows."""
        return np.minimum(o, c) - l

    def _is_bullish(self, o: float, c: float) -> bool:
        """Check if candle is bullish (close > open)."""
        return c > o

    def _is_bearish(self, o: float, c: float) -> bool:
        """Check if candle is bearish (close < open)."""
        return c < o

    def detect_all(self, opens: np.ndarray, highs: np.ndarray,
                   lows: np.ndarray, closes: np.ndarray) -> Dict[str, bool]:
        """Detect all candlestick patterns at the current bar.

        Args:
            opens: Open prices array (at least 5 bars recommended).
            highs: High prices array.
            lows: Low prices array.
            closes: Close prices array.

        Returns:
            Dict mapping pattern name -> detected (bool).
        """
        if len(closes) < 5:
            return {}
        o, h, l, c = opens, highs, lows, closes
        results: Dict[str, bool] = {}
        # Doji variants
        results["doji"] = self._detect_doji(o, h, l, c)
        results["dragonfly_doji"] = self._detect_dragonfly_doji(o, h, l, c)
        results["gravestone_doji"] = self._detect_gravestone_doji(o, h, l, c)
        results["long_legged_doji"] = self._detect_long_legged_doji(o, h, l, c)
        # Single candle
        results["hammer"] = self._detect_hammer(o, h, l, c)
        results["inverted_hammer"] = self._detect_inverted_hammer(o, h, l, c)
        results["hanging_man"] = self._detect_hanging_man(o, h, l, c)
        results["shooting_star"] = self._detect_shooting_star(o, h, l, c)
        results["bullish_marubozu"] = self._detect_bullish_marubozu(o, h, l, c)
        results["bearish_marubozu"] = self._detect_bearish_marubozu(o, h, l, c)
        results["spinning_top"] = self._detect_spinning_top(o, h, l, c)
        # Double candle
        results["bullish_engulfing"] = self._detect_bullish_engulfing(o, h, l, c)
        results["bearish_engulfing"] = self._detect_bearish_engulfing(o, h, l, c)
        results["tweezer_top"] = self._detect_tweezer_top(o, h, l, c)
        results["tweezer_bottom"] = self._detect_tweezer_bottom(o, h, l, c)
        results["piercing_line"] = self._detect_piercing_line(o, h, l, c)
        results["dark_cloud_cover"] = self._detect_dark_cloud_cover(o, h, l, c)
        results["bullish_harami"] = self._detect_bullish_harami(o, h, l, c)
        results["bearish_harami"] = self._detect_bearish_harami(o, h, l, c)
        # Triple candle
        results["morning_star"] = self._detect_morning_star(o, h, l, c)
        results["evening_star"] = self._detect_evening_star(o, h, l, c)
        results["three_white_soldiers"] = self._detect_three_white_soldiers(o, h, l, c)
        results["three_black_crows"] = self._detect_three_black_crows(o, h, l, c)
        results["three_inside_up"] = self._detect_three_inside_up(o, h, l, c)
        results["three_inside_down"] = self._detect_three_inside_down(o, h, l, c)
        results["three_outside_up"] = self._detect_three_outside_up(o, h, l, c)
        results["three_outside_down"] = self._detect_three_outside_down(o, h, l, c)
        results["bullish_abandoned_baby"] = self._detect_bullish_abandoned_baby(o, h, l, c)
        results["bearish_abandoned_baby"] = self._detect_bearish_abandoned_baby(o, h, l, c)
        # Complex patterns
        results["rising_three_methods"] = self._detect_rising_three_methods(o, h, l, c)
        results["falling_three_methods"] = self._detect_falling_three_methods(o, h, l, c)
        results["bullish_belt_hold"] = self._detect_bullish_belt_hold(o, h, l, c)
        results["bearish_belt_hold"] = self._detect_bearish_belt_hold(o, h, l, c)
        results["bullish_counterattack"] = self._detect_bullish_counterattack(o, h, l, c)
        results["bearish_counterattack"] = self._detect_bearish_counterattack(o, h, l, c)
        results["bullish_breakaway"] = self._detect_bullish_breakaway(o, h, l, c)
        results["bearish_breakaway"] = self._detect_bearish_breakaway(o, h, l, c)
        return results

    # --- Doji Variants ---

    def _detect_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Standard doji: body is very small relative to range."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        return rng > 0 and body / rng <= self.doji_ratio

    def _detect_dragonfly_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Dragonfly doji: open=close=high, long lower shadow."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= self.doji_ratio and us / rng <= self.doji_ratio and ls / rng >= 0.6

    def _detect_gravestone_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Gravestone doji: open=close=low, long upper shadow."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= self.doji_ratio and ls / rng <= self.doji_ratio and us / rng >= 0.6

    def _detect_long_legged_doji(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Long-legged doji: small body, long shadows on both sides."""
        rng = self._range(h, l)[-1]
        body = self._body(o, c)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= self.doji_ratio and us / rng >= 0.3 and ls / rng >= 0.3

    # --- Single Candle Patterns ---

    def _detect_hammer(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Hammer: small body at top, long lower shadow (bullish reversal in downtrend)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and ls / rng >= 0.6 and us / rng <= 0.1 and self._is_bullish(o[-1], c[-1])

    def _detect_inverted_hammer(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Inverted hammer: small body at bottom, long upper shadow (bullish reversal)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and us / rng >= 0.6 and ls / rng <= 0.1

    def _detect_hanging_man(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Hanging man: same shape as hammer but in uptrend (bearish reversal)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and ls / rng >= 0.6 and us / rng <= 0.1 and self._is_bearish(o[-1], c[-1])

    def _detect_shooting_star(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Shooting star: small body at bottom, long upper shadow (bearish reversal)."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.33 and us / rng >= 0.6 and ls / rng <= 0.1 and self._is_bearish(o[-1], c[-1])

    def _detect_bullish_marubozu(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish marubozu: no shadows, close > open."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return self._is_bullish(o[-1], c[-1]) and us / rng <= 0.05 and ls / rng <= 0.05

    def _detect_bearish_marubozu(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish marubozu: no shadows, close < open."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return self._is_bearish(o[-1], c[-1]) and us / rng <= 0.05 and ls / rng <= 0.05

    def _detect_spinning_top(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Spinning top: small body, roughly equal shadows."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        body = self._body(o, c)[-1]
        us = self._upper_shadow(h, o, c)[-1]
        ls = self._lower_shadow(l, o, c)[-1]
        return body / rng <= 0.25 and us / rng >= 0.25 and ls / rng >= 0.25

    # --- Double Candle Patterns ---

    def _detect_bullish_engulfing(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish engulfing: current bullish candle engulfs previous bearish."""
        if len(c) < 2:
            return False
        prev_bearish = self._is_bearish(o[-2], c[-2])
        curr_bullish = self._is_bullish(o[-1], c[-1])
        return prev_bearish and curr_bullish and c[-1] >= o[-2] and o[-1] <= c[-2]

    def _detect_bearish_engulfing(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish engulfing: current bearish candle engulfs previous bullish."""
        if len(c) < 2:
            return False
        prev_bullish = self._is_bullish(o[-2], c[-2])
        curr_bearish = self._is_bearish(o[-1], c[-1])
        return prev_bullish and curr_bearish and o[-1] >= c[-2] and c[-1] <= o[-2]

    def _detect_tweezer_top(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Tweezer top: two candles with same high, bearish reversal."""
        if len(c) < 2:
            return False
        same_high = abs(h[-1] - h[-2]) / max(h[-1], h[-2], 1e-10) < 0.001
        return same_high and self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])

    def _detect_tweezer_bottom(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Tweezer bottom: two candles with same low, bullish reversal."""
        if len(c) < 2:
            return False
        same_low = abs(l[-1] - l[-2]) / max(l[-1], l[-2], 1e-10) < 0.001
        return same_low and self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])

    def _detect_piercing_line(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Piercing line: bullish reversal with close above midpoint of prev bearish."""
        if len(c) < 2:
            return False
        prev_mid = (o[-2] + c[-2]) / 2.0
        return (self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and o[-1] < l[-2] and c[-1] > prev_mid and c[-1] < o[-2])

    def _detect_dark_cloud_cover(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Dark cloud cover: bearish reversal with close below midpoint of prev bullish."""
        if len(c) < 2:
            return False
        prev_mid = (o[-2] + c[-2]) / 2.0
        return (self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and o[-1] > h[-2] and c[-1] < prev_mid and c[-1] > o[-2])

    def _detect_bullish_harami(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish harami: small bullish candle inside prev large bearish."""
        if len(c) < 2:
            return False
        return (self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and o[-1] > c[-2] and c[-1] < o[-2] and o[-1] < o[-2] and c[-1] > c[-2])

    def _detect_bearish_harami(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish harami: small bearish candle inside prev large bullish."""
        if len(c) < 2:
            return False
        return (self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and c[-1] > o[-2] and o[-1] < c[-2] and c[-1] < c[-2] and o[-1] > o[-2])

    # --- Triple Candle Patterns ---

    def _detect_morning_star(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Morning star: bearish + small body + bullish reversal."""
        if len(c) < 3:
            return False
        b1_bearish = self._is_bearish(o[-3], c[-3])
        b2_body = self._body(o, c)[-2]
        b2_rng = self._range(h, l)[-2]
        b2_small = b2_rng > 0 and b2_body / b2_rng <= 0.33 if b2_rng > 0 else False
        b3_bullish = self._is_bullish(o[-1], c[-1])
        return b1_bearish and b2_small and b3_bullish and c[-1] > (o[-3] + c[-3]) / 2.0

    def _detect_evening_star(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Evening star: bullish + small body + bearish reversal."""
        if len(c) < 3:
            return False
        b1_bullish = self._is_bullish(o[-3], c[-3])
        b2_body = self._body(o, c)[-2]
        b2_rng = self._range(h, l)[-2]
        b2_small = b2_rng > 0 and b2_body / b2_rng <= 0.33 if b2_rng > 0 else False
        b3_bearish = self._is_bearish(o[-1], c[-1])
        return b1_bullish and b2_small and b3_bearish and c[-1] < (o[-3] + c[-3]) / 2.0

    def _detect_three_white_soldiers(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three white soldiers: three consecutive bullish candles, each opening within prev body."""
        if len(c) < 3:
            return False
        return (self._is_bullish(o[-3], c[-3]) and self._is_bullish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and c[-1] > c[-2] > c[-3] and o[-2] > o[-3] and o[-2] < c[-3]
                and o[-1] > o[-2] and o[-1] < c[-2])

    def _detect_three_black_crows(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three black crows: three consecutive bearish candles, each opening within prev body."""
        if len(c) < 3:
            return False
        return (self._is_bearish(o[-3], c[-3]) and self._is_bearish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and c[-1] < c[-2] < c[-3] and o[-2] < o[-3] and o[-2] > c[-3]
                and o[-1] < o[-2] and o[-1] > c[-2])

    def _detect_three_inside_up(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three inside up: bullish harami followed by bullish close."""
        if len(c) < 3:
            return False
        harami = (self._is_bearish(o[-3], c[-3]) and self._is_bullish(o[-2], c[-2])
                  and o[-2] > c[-3] and c[-2] < o[-3])
        return harami and self._is_bullish(o[-1], c[-1]) and c[-1] > c[-2]

    def _detect_three_inside_down(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three inside down: bearish harami followed by bearish close."""
        if len(c) < 3:
            return False
        harami = (self._is_bullish(o[-3], c[-3]) and self._is_bearish(o[-2], c[-2])
                  and c[-2] > o[-3] and o[-2] < c[-3])
        return harami and self._is_bearish(o[-1], c[-1]) and c[-1] < c[-2]

    def _detect_three_outside_up(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three outside up: bullish engulfing followed by higher close."""
        if len(c) < 3:
            return False
        engulfing = (self._is_bearish(o[-3], c[-3]) and self._is_bullish(o[-2], c[-2])
                     and c[-2] >= o[-3] and o[-2] <= c[-3])
        return engulfing and self._is_bullish(o[-1], c[-1]) and c[-1] > c[-2]

    def _detect_three_outside_down(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Three outside down: bearish engulfing followed by lower close."""
        if len(c) < 3:
            return False
        engulfing = (self._is_bullish(o[-3], c[-3]) and self._is_bearish(o[-2], c[-2])
                     and o[-2] >= c[-3] and c[-2] <= o[-3])
        return engulfing and self._is_bearish(o[-1], c[-1]) and c[-1] < c[-2]

    def _detect_bullish_abandoned_baby(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish abandoned baby: bearish + doji gap + bullish."""
        if len(c) < 3:
            return False
        b1_bearish = self._is_bearish(o[-3], c[-3])
        b2_doji = self._range(h, l)[-2] > 0 and self._body(o, c)[-2] / self._range(h, l)[-2] <= self.doji_ratio
        b3_bullish = self._is_bullish(o[-1], c[-1])
        gap_down = l[-2] > h[-3] if h[-3] > 0 else False
        gap_up = l[-1] > h[-2] if h[-2] > 0 else False
        return b1_bearish and b2_doji and b3_bullish and gap_down and gap_up

    def _detect_bearish_abandoned_baby(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish abandoned baby: bullish + doji gap + bearish."""
        if len(c) < 3:
            return False
        b1_bullish = self._is_bullish(o[-3], c[-3])
        b2_doji = self._range(h, l)[-2] > 0 and self._body(o, c)[-2] / self._range(h, l)[-2] <= self.doji_ratio
        b3_bearish = self._is_bearish(o[-1], c[-1])
        gap_up = h[-2] < l[-3] if l[-3] > 0 else False
        gap_down = h[-1] < l[-2] if l[-2] > 0 else False
        return b1_bullish and b2_doji and b3_bearish and gap_up and gap_down

    # --- Complex Patterns ---

    def _detect_rising_three_methods(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Rising three methods: large bullish + 3 small bearish + large bullish."""
        if len(c) < 5:
            return False
        b1_bullish = self._is_bullish(o[-5], c[-5]) and self._body(o, c)[-5] / self._range(h, l)[-5] > 0.6 if self._range(h, l)[-5] > 0 else False
        small_bears = all(self._is_bearish(o[-4 + i], c[-4 + i]) for i in range(3))
        within_range = all(l[-4 + i] > l[-5] and h[-4 + i] < h[-5] for i in range(3))
        b5_bullish = self._is_bullish(o[-1], c[-1]) and c[-1] > c[-5]
        return b1_bullish and small_bears and within_range and b5_bullish

    def _detect_falling_three_methods(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Falling three methods: large bearish + 3 small bullish + large bearish."""
        if len(c) < 5:
            return False
        b1_bearish = self._is_bearish(o[-5], c[-5]) and self._body(o, c)[-5] / self._range(h, l)[-5] > 0.6 if self._range(h, l)[-5] > 0 else False
        small_bulls = all(self._is_bullish(o[-4 + i], c[-4 + i]) for i in range(3))
        within_range = all(l[-4 + i] > l[-5] and h[-4 + i] < h[-5] for i in range(3))
        b5_bearish = self._is_bearish(o[-1], c[-1]) and c[-1] < c[-5]
        return b1_bearish and small_bulls and within_range and b5_bearish

    def _detect_bullish_belt_hold(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish belt hold: opens at low, closes near high in downtrend."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        return self._is_bullish(o[-1], c[-1]) and abs(o[-1] - l[-1]) / rng < 0.01 and self._body(o, c)[-1] / rng > 0.5

    def _detect_bearish_belt_hold(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish belt hold: opens at high, closes near low in uptrend."""
        rng = self._range(h, l)[-1]
        if rng == 0:
            return False
        return self._is_bearish(o[-1], c[-1]) and abs(o[-1] - h[-1]) / rng < 0.01 and self._body(o, c)[-1] / rng > 0.5

    def _detect_bullish_counterattack(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish counterattack: bearish candle followed by bullish closing at same level."""
        if len(c) < 2:
            return False
        return (self._is_bearish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
                and abs(c[-1] - c[-2]) / max(abs(c[-1]), abs(c[-2]), 1e-10) < 0.01)

    def _detect_bearish_counterattack(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish counterattack: bullish candle followed by bearish closing at same level."""
        if len(c) < 2:
            return False
        return (self._is_bullish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
                and abs(c[-1] - c[-2]) / max(abs(c[-1]), abs(c[-2]), 1e-10) < 0.01)

    def _detect_bullish_breakaway(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bullish breakaway: 5-candle bullish reversal pattern."""
        if len(c) < 5:
            return False
        long_bear = self._is_bearish(o[-5], c[-5]) and self._body(o, c)[-5] / max(self._range(h, l)[-5], 1e-10) > 0.5
        continues_down = c[-4] < c[-5] and c[-3] < c[-4]
        reversal = self._is_bullish(o[-2], c[-2]) and self._is_bullish(o[-1], c[-1])
        close_above = c[-1] > (o[-5] + c[-5]) / 2.0
        return long_bear and continues_down and reversal and close_above

    def _detect_bearish_breakaway(self, o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> bool:
        """Bearish breakaway: 5-candle bearish reversal pattern."""
        if len(c) < 5:
            return False
        long_bull = self._is_bullish(o[-5], c[-5]) and self._body(o, c)[-5] / max(self._range(h, l)[-5], 1e-10) > 0.5
        continues_up = c[-4] > c[-5] and c[-3] > c[-4]
        reversal = self._is_bearish(o[-2], c[-2]) and self._is_bearish(o[-1], c[-1])
        close_below = c[-1] < (o[-5] + c[-5]) / 2.0
        return long_bull and continues_up and reversal and close_below


# ============================================================================
# Statistical Utility Functions
# ============================================================================

__all__ = ['CandlestickPatterns']
