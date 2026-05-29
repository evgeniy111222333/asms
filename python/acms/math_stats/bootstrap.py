"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class Bootstrap:
    """Bootstrap methods for confidence intervals.

    Implements percentile bootstrap, BCa (bias-corrected and accelerated)
    bootstrap, and block bootstrap for time series.
    """

    @staticmethod
    def confidence_interval(data: np.ndarray, statistic: Callable = np.mean,
                           n_bootstrap: int = 1000, alpha: float = 0.05) -> Dict:
        """Compute bootstrap confidence interval (percentile method).

        Args:
            data: Input data.
            statistic: Function to compute statistic.
            n_bootstrap: Number of bootstrap samples.
            alpha: Significance level.

        Returns:
            Dict with CI bounds and bootstrap distribution.
        """
        if len(data) < 5:
            return {"lower": float('nan'), "upper": float('nan'), "estimate": float('nan')}

        n = len(data)
        boot_stats = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            boot_stats[i] = statistic(sample)

        lower = float(np.percentile(boot_stats, alpha / 2 * 100))
        upper = float(np.percentile(boot_stats, (1 - alpha / 2) * 100))
        estimate = float(statistic(data))

        return {
            "lower": lower, "upper": upper, "estimate": estimate,
            "std_error": float(np.std(boot_stats)),
            "bootstrap_distribution": boot_stats,
        }

    @staticmethod
    def bca_confidence_interval(data: np.ndarray, statistic: Callable = np.mean,
                                n_bootstrap: int = 2000,
                                alpha: float = 0.05) -> Dict:
        """Compute BCa (bias-corrected and accelerated) bootstrap CI.

        The BCa method corrects for both bias and skewness in the
        bootstrap distribution, producing more accurate intervals.

        Args:
            data: Input data.
            statistic: Function to compute statistic.
            n_bootstrap: Number of bootstrap samples.
            alpha: Significance level.

        Returns:
            Dict with BCa-corrected CI bounds.
        """
        if len(data) < 5:
            return {"lower": float('nan'), "upper": float('nan'), "estimate": float('nan')}

        n = len(data)
        theta_hat = float(statistic(data))
        boot_stats = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            boot_stats[i] = statistic(sample)

        # Bias correction: z0
        prop_below = np.mean(boot_stats < theta_hat)
        z0 = stats.norm.ppf(prop_below) if 0 < prop_below < 1 else 0.0

        # Acceleration: a (using jackknife)
        jackknife_stats = np.zeros(n)
        for i in range(n):
            jack_sample = np.delete(data, i)
            jackknife_stats[i] = statistic(jack_sample)

        jack_mean = np.mean(jackknife_stats)
        a_numerator = np.sum((jack_mean - jackknife_stats) ** 3)
        a_denominator = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
        a = a_numerator / a_denominator if a_denominator > 0 else 0.0

        # Adjusted percentiles
        z_alpha_low = stats.norm.ppf(alpha / 2)
        z_alpha_high = stats.norm.ppf(1 - alpha / 2)

        def adjust_z(z_alpha):
            denom = 1 - a * (z0 + z_alpha)
            if abs(denom) < 1e-10:
                return z_alpha
            return z0 + (z0 + z_alpha) / denom

        adjusted_low = adjust_z(z_alpha_low)
        adjusted_high = adjust_z(z_alpha_high)

        p_low = stats.norm.cdf(adjusted_low)
        p_high = stats.norm.cdf(adjusted_high)

        lower = float(np.percentile(boot_stats, p_low * 100))
        upper = float(np.percentile(boot_stats, p_high * 100))

        return {
            "lower": lower, "upper": upper, "estimate": theta_hat,
            "std_error": float(np.std(boot_stats)),
            "bias_correction_z0": float(z0),
            "acceleration_a": float(a),
            "bootstrap_distribution": boot_stats,
        }

    @staticmethod
    def block_bootstrap(data: np.ndarray, block_size: int = 10,
                        n_bootstrap: int = 1000, alpha: float = 0.05) -> Dict:
        """Block bootstrap for time series (preserves autocorrelation).

        Args:
            data: Time series data.
            block_size: Size of blocks.
            n_bootstrap: Number of bootstrap samples.
            alpha: Significance level.

        Returns:
            Dict with CI bounds.
        """
        n = len(data)
        if n < block_size * 2:
            return {"lower": float('nan'), "upper": float('nan'), "estimate": float(np.mean(data))}

        boot_stats = np.zeros(n_bootstrap)
        n_blocks = n // block_size

        for i in range(n_bootstrap):
            start_indices = np.random.randint(0, n - block_size + 1, size=n_blocks)
            sample = np.concatenate([data[s:s + block_size] for s in start_indices])
            boot_stats[i] = np.mean(sample[:n])

        return {
            "lower": float(np.percentile(boot_stats, alpha / 2 * 100)),
            "upper": float(np.percentile(boot_stats, (1 - alpha / 2) * 100)),
            "estimate": float(np.mean(data)),
        }



__all__ = ['Bootstrap']
