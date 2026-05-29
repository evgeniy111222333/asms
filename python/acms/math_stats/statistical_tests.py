"""Math & Statistics Library for ACMS."""

import logging

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable

logger = logging.getLogger(__name__)


class GrangerCausalityTest:
    """Granger causality test with automatic lag selection.

    Tests whether one time series helps predict another,
    using F-tests for nested model comparison.
    """

    @staticmethod
    def test(y: np.ndarray, x: np.ndarray, max_lag: int = 5) -> Dict:
        """Perform Granger causality test.

        Tests whether x Granger-causes y.

        Args:
            y: Dependent variable series.
            x: Potential causal variable series.
            max_lag: Maximum lag to test.

        Returns:
            Dict with F-statistic, p-value, and causality result.
        """
        if len(y) != len(x) or len(y) < max_lag + 10:
            return {"f_statistic": float('nan'), "p_value": float('nan'),
                    "causes": False, "best_lag": 0}

        T = len(y)
        best_lag = 1
        best_f = 0
        best_p = 1.0

        for lag in range(1, max_lag + 1):
            Y = y[lag:]
            X_restricted = np.column_stack([np.ones(T - lag)] +
                                           [y[lag - i:T - i] for i in range(1, lag + 1)])
            beta_r = np.linalg.lstsq(X_restricted, Y, rcond=None)[0]
            resid_r = Y - X_restricted @ beta_r
            ssr_r = np.sum(resid_r ** 2)

            X_unrestricted = np.column_stack([X_restricted] +
                                              [x[lag - i:T - i] for i in range(1, lag + 1)])
            beta_u = np.linalg.lstsq(X_unrestricted, Y, rcond=None)[0]
            resid_u = Y - X_unrestricted @ beta_u
            ssr_u = np.sum(resid_u ** 2)

            n_params = lag
            df1 = n_params
            df2 = T - 2 * lag - 1

            if df2 > 0 and ssr_u > 0:
                f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)

                if p_value < best_p:
                    best_f = f_stat
                    best_p = p_value
                    best_lag = lag

        return {
            "f_statistic": float(best_f),
            "p_value": float(best_p),
            "causes": bool(best_p < 0.05),
            "best_lag": best_lag,
        }


# ============================================================================
# Cointegration

class EngleGranger:
    """Engle-Granger cointegration test.

    Two-step procedure: OLS regression then ADF test on residuals.
    """

    @staticmethod
    def test(y: np.ndarray, x: np.ndarray, significance: float = 0.05) -> dict:
        """Two-step Engle-Granger cointegration test.

        Args:
            y: Dependent variable.
            x: Independent variable.
            significance: Significance level.

        Returns:
            Dict with test results.
        """
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        hedge_ratio = beta[1]
        intercept = beta[0]
        residuals = y - X @ beta

        adf_stat = 0.0
        p_value = 0.1
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(residuals, maxlag=1)
            adf_stat = adf_result[0]
            p_value = adf_result[1]
        except ImportError:
            logger.debug("statsmodels not available for ADF test in cointegration")

        is_cointegrated = p_value < significance
        return {
            "adf_statistic": float(adf_stat),
            "p_value": float(p_value),
            "hedge_ratio": float(hedge_ratio),
            "intercept": float(intercept),
            "is_cointegrated": is_cointegrated,
            "residuals": residuals,
        }


class Johansen:
    """Johansen cointegration test.

    Uses statsmodels implementation if available.
    """

    @staticmethod
    def test(data: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> dict:
        """Perform Johansen cointegration test.

        Args:
            data: Multivariate time series (T x N).
            det_order: Deterministic trend assumption.
            k_ar_diff: Lag order for VAR differences.

        Returns:
            Dict with eigenvalues, trace statistics, and eigenvectors.
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            result = coint_johansen(data, det_order, k_ar_diff)
            return {
                "eig": result.eig, "lr1": result.lr1, "lr2": result.lr2,
                "cvt": result.cvt, "cvm": result.cvm, "evec": result.evec,
            }
        except ImportError:
            return {"error": "statsmodels not available for Johansen test"}



__all__ = ['GrangerCausalityTest', 'EngleGranger', 'Johansen']
