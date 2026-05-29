"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class GaussianCopula:
    """Gaussian Copula for dependency structure modeling.

    Separates marginal distributions from the dependency structure.
    Gaussian copula has zero tail dependence.
    """

    def __init__(self):
        self.correlation: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> Dict:
        """Fit Gaussian copula to multivariate data.

        Args:
            data: Data matrix (T x N).

        Returns:
            Dict with correlation matrix and fit statistics.
        """
        T, N = data.shape
        u = np.zeros_like(data)
        for j in range(N):
            ranks = stats.rankdata(data[:, j]) / (T + 1)
            u[:, j] = ranks

        z = stats.norm.ppf(u)
        z = np.nan_to_num(z, nan=0.0, posinf=3.0, neginf=-3.0)

        self.correlation = np.corrcoef(z.T)

        return {"correlation": self.correlation, "pseudo_observations": u}

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """Generate samples from the fitted copula.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Matrix of samples in uniform space (n_samples x N).
        """
        if self.correlation is None:
            raise RuntimeError("Copula not fitted yet")
        N = self.correlation.shape[0]
        z = np.random.multivariate_normal(np.zeros(N), self.correlation, n_samples)
        u = stats.norm.cdf(z)
        return u

    def tail_dependence(self) -> Dict[str, float]:
        """Compute tail dependence coefficients.

        Returns:
            Dict with upper and lower tail dependence (both 0 for Gaussian).
        """
        return {
            "upper_tail": 0.0,
            "lower_tail": 0.0,
            "note": "Gaussian copula has zero tail dependence by definition",
        }


class StudentTCopula:
    """Student-t Copula for dependency structure modeling.

    Captures symmetric tail dependence, making it more appropriate
    than Gaussian for financial returns with joint extreme events.
    """

    def __init__(self):
        self.correlation: Optional[np.ndarray] = None
        self.df: float = 5.0

    def fit(self, data: np.ndarray) -> Dict:
        """Fit Student-t copula to multivariate data.

        Estimates the correlation matrix and degrees of freedom
        using method-of-moments for df.

        Args:
            data: Data matrix (T x N).

        Returns:
            Dict with correlation matrix, degrees of freedom, and diagnostics.
        """
        T, N = data.shape

        # Transform to uniform margins
        u = np.zeros_like(data)
        for j in range(N):
            ranks = stats.rankdata(data[:, j]) / (T + 1)
            u[:, j] = ranks

        # Transform to standard normal for correlation estimation
        z = stats.norm.ppf(u)
        z = np.nan_to_num(z, nan=0.0, posinf=3.0, neginf=-3.0)

        self.correlation = np.corrcoef(z.T)

        # Estimate degrees of freedom using method-of-moments
        # Based on the kurtosis of the transformed data
        if N >= 2:
            portfolio = np.mean(z, axis=1)
            kurt = float(stats.kurtosis(portfolio, fisher=True))
            if kurt > 0:
                self.df = max(2.5, 6.0 / kurt + 4.0)
            else:
                self.df = 10.0

        return {
            "correlation": self.correlation,
            "degrees_of_freedom": self.df,
            "pseudo_observations": u,
        }

    def sample(self, n_samples: int = 1000) -> np.ndarray:
        """Generate samples from the fitted Student-t copula.

        Args:
            n_samples: Number of samples to generate.

        Returns:
            Matrix of samples in uniform space (n_samples x N).
        """
        if self.correlation is None:
            raise RuntimeError("Copula not fitted yet")
        N = self.correlation.shape[0]

        # Generate multivariate t samples
        z = np.random.multivariate_normal(np.zeros(N), self.correlation, n_samples)
        chi2 = np.random.chisquare(self.df, size=n_samples)
        t_samples = z / np.sqrt(chi2[:, np.newaxis] / self.df)

        # Transform to uniform via t CDF
        u = stats.t.cdf(t_samples, df=self.df)
        return u

    def tail_dependence(self, quantile: float = 0.05) -> Dict[str, float]:
        """Compute empirical tail dependence coefficients.

        Args:
            quantile: Quantile for tail dependence estimation.

        Returns:
            Dict with estimated upper and lower tail dependence.
        """
        if self.correlation is None or self.correlation.shape[0] < 2:
            return {"upper_tail": 0.0, "lower_tail": 0.0}

        rho = self.correlation[0, 1]
        # Approximate tail dependence for Student-t copula
        # lambda_L = 2 * t_{nu+1}(-sqrt(nu+1) * sqrt((1-rho)/(1+rho)))
        if rho > -1 and rho < 1:
            factor = np.sqrt((1 - rho) / (1 + rho))
            t_val = -np.sqrt(self.df + 1) * factor
            lower_tail = 2 * stats.t.cdf(t_val, df=self.df + 1)
            upper_tail = lower_tail  # Symmetric
        else:
            lower_tail = 0.0
            upper_tail = 0.0

        return {
            "upper_tail": float(upper_tail),
            "lower_tail": float(lower_tail),
            "degrees_of_freedom": self.df,
            "correlation_01": float(rho),
        }



__all__ = ['GaussianCopula', 'StudentTCopula']
