"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class VarianceRatioTest:
    """Variance Ratio test for random walk hypothesis (Lo-MacKinlay).

    If a series is a random walk, the variance of q-period returns
    should be q times the variance of one-period returns.
    VR(q) = Var(r_q) / (q * Var(r_1))
    VR = 1: random walk. VR > 1: trending. VR < 1: mean-reverting.
    """

    @staticmethod
    def test(data: np.ndarray, q: int = 2) -> Dict:
        """Perform Variance Ratio test for a single holding period.

        Args:
            data: Price series.
            q: Holding period for variance ratio calculation.

        Returns:
            Dict with VR statistic, z-score, and p-value.
        """
        if len(data) < q * 10:
            return {"vr": float('nan'), "z_stat": float('nan'), "p_value": float('nan'),
                    "is_random_walk": True}

        returns = np.diff(np.log(data))
        n = len(returns)

        var_1 = np.var(returns, ddof=1)
        q_returns = np.diff(np.log(data), n=q)
        var_q = np.var(q_returns, ddof=1)

        if var_1 == 0:
            return {"vr": float('nan'), "z_stat": float('nan'), "p_value": float('nan'),
                    "is_random_walk": True}

        vr = var_q / (q * var_1)

        theta = 0.0
        for j in range(1, q):
            delta_j = np.sum((returns[j:] - np.mean(returns)) * (returns[:-j] - np.mean(returns))) / n
            theta += 2 * (q - j) / q * delta_j / var_1

        se = np.sqrt((2 * (2*q - 1) * (q - 1) / (3 * q * n)))
        z_stat = (vr - 1) / se if se > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            "vr": float(vr), "z_stat": float(z_stat), "p_value": float(p_value),
            "is_random_walk": p_value > 0.05,
            "interpretation": "random_walk" if p_value > 0.05 else ("trending" if vr > 1 else "mean_reverting"),
        }

    @staticmethod
    def multiple_holding_periods(data: np.ndarray,
                                  periods: Optional[List[int]] = None) -> Dict:
        """Test variance ratio across multiple holding periods.

        Args:
            data: Price series.
            periods: List of holding periods to test.

        Returns:
            Dict with results for each period and joint interpretation.
        """
        if periods is None:
            periods = [2, 4, 8, 16, 32]

        results = {}
        trending_count = 0
        mean_reverting_count = 0

        for q in periods:
            results[q] = VarianceRatioTest.test(data, q)
            interp = results[q].get("interpretation", "random_walk")
            if interp == "trending":
                trending_count += 1
            elif interp == "mean_reverting":
                mean_reverting_count += 1

        if trending_count > len(periods) / 2:
            joint_interpretation = "trending"
        elif mean_reverting_count > len(periods) / 2:
            joint_interpretation = "mean_reverting"
        else:
            joint_interpretation = "random_walk"

        return {
            "periods": results,
            "joint_interpretation": joint_interpretation,
            "trending_count": trending_count,
            "mean_reverting_count": mean_reverting_count,
        }


# ============================================================================
# Phillips-Perron Test

class PhillipsPerronTest:
    """Phillips-Perron unit root test.

    Non-parametric test for stationarity that corrects for
    serial correlation and heteroscedasticity using Newey-West
    standard errors.
    """

    @staticmethod
    def test(data: np.ndarray, lags: Optional[int] = None) -> Dict:
        """Perform Phillips-Perron test.

        Args:
            data: Time series to test.
            lags: Number of lags for long-run variance (auto if None).

        Returns:
            Dict with test statistic and p-value.
        """
        if len(data) < 30:
            return {"pp_statistic": float('nan'), "p_value": float('nan'), "is_stationary": True}

        y = np.diff(data)
        x = data[:-1]
        n = len(y)

        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        sigma2 = np.sum(residuals ** 2) / n

        if lags is None:
            lags = int(4 * (n / 100) ** (2/9))

        gamma0 = np.sum(residuals ** 2) / n
        s2 = gamma0
        for j in range(1, lags + 1):
            w = 1 - j / (lags + 1)  # Bartlett kernel
            gamma_j = np.sum(residuals[j:] * residuals[:-j]) / n
            s2 += 2 * w * gamma_j

        t_stat = beta[1] / (np.sqrt(sigma2) / np.sqrt(np.sum(x ** 2)))
        pp_stat = t_stat * np.sqrt(sigma2 / s2) - 0.5 * (s2 - sigma2) * n * beta[1] / (np.sqrt(s2) * np.sqrt(np.sum(x ** 2)))

        is_stationary = pp_stat < -2.86

        return {
            "pp_statistic": float(pp_stat),
            "p_value": float(max(0.001, stats.norm.cdf(pp_stat))),
            "is_stationary": is_stationary,
            "lags_used": lags,
        }

__all__ = ['VarianceRatioTest', 'PhillipsPerronTest']
