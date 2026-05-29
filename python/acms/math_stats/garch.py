"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class GARCH11:
    """GARCH(1,1) volatility model.

    sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2

    Estimates parameters using maximum likelihood and provides
    volatility forecasting and standardized residuals.
    """

    def __init__(self, omega: float = 0.1, alpha: float = 0.1, beta: float = 0.8):
        """Initialize GARCH(1,1) model.

        Args:
            omega: Constant term.
            alpha: ARCH coefficient (lagged squared residual).
            beta: GARCH coefficient (lagged conditional variance).
        """
        self.omega = omega
        self.alpha = alpha
        self.beta = beta

    def fit(self, returns: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> dict:
        """Fit GARCH(1,1) parameters using maximum likelihood.

        Uses a perturbation-based optimization to find parameters
        that maximize the Gaussian log-likelihood.

        Args:
            returns: Array of log returns.
            max_iter: Maximum iterations for optimization.
            tol: Convergence tolerance.

        Returns:
            Dict with fitted parameters, conditional variances, and diagnostics.
        """
        if len(returns) < 50:
            return {"omega": self.omega, "alpha": self.alpha, "beta": self.beta,
                    "conditional_variance": np.full(len(returns), np.var(returns)),
                    "standardized_residuals": returns.copy()}

        T = len(returns)
        var_target = np.var(returns)

        omega = var_target * 0.05
        alpha = 0.08
        beta = 0.87

        def neg_log_likelihood(params):
            w, a, b = params
            if w < 0 or a < 0 or b < 0 or a + b >= 1:
                return 1e10
            h = np.zeros(T)
            h[0] = var_target
            for t in range(1, T):
                h[t] = w + a * returns[t-1]**2 + b * h[t-1]
                if h[t] <= 0:
                    return 1e10
            ll = -0.5 * np.sum(np.log(h) + returns**2 / h)
            return -ll

        best_ll = neg_log_likelihood([omega, alpha, beta])
        best_params = [omega, alpha, beta]

        for _ in range(max_iter):
            improved = False
            for idx in range(3):
                for delta in [0.01, -0.01, 0.001, -0.001]:
                    params = best_params.copy()
                    params[idx] += delta
                    if params[0] > 0 and params[1] > 0 and params[2] > 0 and params[1] + params[2] < 1:
                        ll = neg_log_likelihood(params)
                        if ll < best_ll:
                            best_ll = ll
                            best_params = params
                            improved = True
            if not improved:
                break

        self.omega, self.alpha, self.beta = best_params
        conditional_var = self._compute_variance(returns)
        standardized_residuals = returns / np.sqrt(conditional_var)

        return {
            "omega": self.omega, "alpha": self.alpha, "beta": self.beta,
            "conditional_variance": conditional_var,
            "standardized_residuals": standardized_residuals,
            "persistence": self.alpha + self.beta,
            "long_run_variance": self.omega / (1 - self.alpha - self.beta) if self.alpha + self.beta < 1 else float('inf'),
        }

    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast volatility for `horizon` steps ahead.

        Args:
            returns: Historical returns.
            horizon: Number of steps to forecast.

        Returns:
            Array of forecasted variances.
        """
        h = self._compute_variance(returns)
        if len(h) == 0:
            return np.array([])

        forecasts = np.zeros(horizon)
        last_h = h[-1]
        last_e2 = returns[-1] ** 2

        for i in range(horizon):
            forecasts[i] = self.omega + self.alpha * last_e2 + self.beta * last_h
            last_e2 = forecasts[i]
            last_h = forecasts[i]

        return forecasts

    def _compute_variance(self, returns: np.ndarray) -> np.ndarray:
        """Compute conditional variance series.

        Args:
            returns: Return series.

        Returns:
            Conditional variance series.
        """
        T = len(returns)
        h = np.zeros(T)
        h[0] = np.var(returns)
        for t in range(1, T):
            h[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * h[t-1]
            h[t] = max(h[t], 1e-10)
        return h

__all__ = ['GARCH11']
