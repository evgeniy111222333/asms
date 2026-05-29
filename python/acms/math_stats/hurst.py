"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class HurstExponent:
    """Hurst exponent estimation using R/S analysis with Anis-Lloyd correction.

    The Hurst exponent characterizes the long-range dependence of a time series:
    - H < 0.5: Mean-reverting (anti-persistent)
    - H = 0.5: Random walk
    - H > 0.5: Trending (persistent)

    Uses the corrected R/S method from Anis & Lloyd (1976) and provides
    bootstrapped confidence intervals.
    """

    @staticmethod
    def _rs_analysis(data: np.ndarray, min_subsample: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Perform R/S analysis on a time series.

        Args:
            data: Time series data.
            min_subsample: Minimum subsample size.

        Returns:
            Tuple of (log_sizes, log_rs) arrays for regression.
        """
        n = len(data)
        max_size = n // 2
        sizes = []
        rs_values = []

        size = min_subsample
        while size <= max_size:
            n_subsamples = n // size
            rs_subsample = []

            for i in range(n_subsamples):
                subsample = data[i * size:(i + 1) * size]
                mean_val = np.mean(subsample)
                deviations = np.cumsum(subsample - mean_val)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(subsample, ddof=1)
                if S > 0:
                    rs_subsample.append(R / S)

            if rs_subsample:
                sizes.append(size)
                rs_values.append(np.mean(rs_subsample))

            size = int(size * 1.5) if size < max_size else max_size + 1

        if len(sizes) < 3:
            return np.array([]), np.array([])

        log_sizes = np.log(sizes)
        log_rs = np.log(rs_values)

        # Anis-Lloyd correction
        corrected_rs = []
        for s in sizes:
            # Expected R/S for random walk
            al_correction = 0.5 * np.pi * s if s > 1 else 1.0
            expected_rs = al_correction / np.sqrt(np.pi * s / 2) if s > 2 else 1.0
            # Simpler Anis-Lloyd: E[R/S] = (n/2)^H for H=0.5 -> sqrt(pi*n/2) / Gamma(0.5*(n+1)/n)
            # Using the approximation from Weron (2002)
            expected_rs_al = np.sqrt(np.pi * s / 2) * (1.0 / (np.math.gamma(0.5 * (s + 1) / s)))
            corrected_rs.append(np.log(rs_values[sizes.index(s)]) - np.log(expected_rs_al) + np.log(np.sqrt(s / 2)))

        return np.array(log_sizes), np.array(corrected_rs)

    @staticmethod
    def estimate(data: np.ndarray, min_subsample: int = 10) -> Dict:
        """Estimate Hurst exponent using corrected R/S analysis.

        Args:
            data: Time series data (prices or values).
            min_subsample: Minimum subsample size for R/S computation.

        Returns:
            Dict with Hurst exponent, R-squared, and interpretation.
        """
        if len(data) < 100:
            return {"hurst": float('nan'), "r_squared": float('nan'),
                    "interpretation": "insufficient_data"}

        # Use log returns if data appears to be prices
        returns = np.diff(np.log(data)) if np.all(data > 0) and np.std(data) > np.mean(data) * 0.1 else np.diff(data)

        log_sizes, log_rs = HurstExponent._rs_analysis(returns, min_subsample)
        if len(log_sizes) < 3:
            return {"hurst": float('nan'), "r_squared": float('nan'),
                    "interpretation": "insufficient_data"}

        # Linear regression: log(R/S) = H * log(n) + c
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_rs)
        hurst = float(slope)

        if hurst < 0.45:
            interpretation = "mean_reverting"
        elif hurst > 0.55:
            interpretation = "trending"
        else:
            interpretation = "random_walk"

        return {
            "hurst": hurst,
            "r_squared": float(r_value ** 2),
            "std_error": float(std_err),
            "p_value": float(p_value),
            "interpretation": interpretation,
            "n_points": len(log_sizes),
        }

    @staticmethod
    def estimate_with_bootstrap(data: np.ndarray, n_bootstrap: int = 200,
                                 min_subsample: int = 10,
                                 confidence: float = 0.95) -> Dict:
        """Estimate Hurst exponent with bootstrapped confidence intervals.

        Resamples the data with replacement and re-estimates H for each
        bootstrap sample to construct a confidence interval.

        Args:
            data: Time series data.
            n_bootstrap: Number of bootstrap iterations.
            min_subsample: Minimum subsample size.
            confidence: Confidence level for interval.

        Returns:
            Dict with Hurst exponent, CI, and interpretation.
        """
        point_estimate = HurstExponent.estimate(data, min_subsample)
        if np.isnan(point_estimate["hurst"]):
            return point_estimate

        n = len(data)
        boot_hursts = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # Bootstrap resample
            indices = np.random.choice(n, size=n, replace=True)
            boot_data = data[indices]
            boot_result = HurstExponent.estimate(boot_data, min_subsample)
            boot_hursts[i] = boot_result["hurst"]

        boot_hursts = boot_hursts[~np.isnan(boot_hursts)]
        if len(boot_hursts) == 0:
            return point_estimate

        alpha = 1 - confidence
        ci_lower = float(np.percentile(boot_hursts, alpha / 2 * 100))
        ci_upper = float(np.percentile(boot_hursts, (1 - alpha / 2) * 100))

        point_estimate["ci_lower"] = ci_lower
        point_estimate["ci_upper"] = ci_upper
        point_estimate["boot_std_error"] = float(np.std(boot_hursts))
        point_estimate["n_bootstrap"] = n_bootstrap

        # Check if CI contains 0.5
        if ci_lower > 0.55:
            point_estimate["interpretation"] = "trending_significant"
        elif ci_upper < 0.45:
            point_estimate["interpretation"] = "mean_reverting_significant"
        else:
            point_estimate["interpretation"] = "random_walk_not_rejected"

        return point_estimate

__all__ = ['HurstExponent']
