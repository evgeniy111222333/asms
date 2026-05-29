"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class KMeansClustering:
    """K-means clustering for regime/market structure detection.

    Implements K-means with multiple random initializations
    and selects the best solution by inertia.
    """

    def fit(self, data: np.ndarray, k: int = 3, max_iter: int = 100,
            n_init: int = 10) -> dict:
        """Fit K-means clustering.

        Args:
            data: Data matrix (n_samples x n_features).
            k: Number of clusters.
            max_iter: Maximum iterations per initialization.
            n_init: Number of random initializations.

        Returns:
            Dict with labels, centroids, and inertia.
        """
        best_labels = None
        best_inertia = float('inf')
        best_centroids = None

        for _ in range(n_init):
            idx = np.random.choice(len(data), k, replace=False)
            centroids = data[idx].copy()

            for _ in range(max_iter):
                distances = np.array([[np.linalg.norm(x - c) for c in centroids] for x in data])
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([
                    data[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
                    for i in range(k)
                ])
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            inertia = sum(np.sum((data[labels == i] - centroids[i]) ** 2) for i in range(k))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        return {"labels": best_labels, "centroids": best_centroids,
                "inertia": float(best_inertia), "k": k}


class PCA:
    """Principal Component Analysis.

    Decomposes data into orthogonal components ordered by
    variance explained.
    """

    def fit(self, data: np.ndarray, n_components: Optional[int] = None) -> dict:
        """Fit PCA model.

        Args:
            data: Data matrix (n_samples x n_features).
            n_components: Number of components to retain.

        Returns:
            Dict with components, explained variance, and mean.
        """
        mean = np.mean(data, axis=0)
        centered = data - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        total_var = np.sum(eigenvalues)
        explained_variance_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues
        n = n_components or len(eigenvalues)
        return {
            "components": eigenvectors[:, :n].T,
            "explained_variance": eigenvalues[:n],
            "explained_variance_ratio": explained_variance_ratio[:n],
            "mean": mean, "n_components": n,
            "cumulative_variance": np.cumsum(explained_variance_ratio[:n]),
        }

    def transform(self, data: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
        """Project data onto principal components.

        Args:
            data: Data matrix.
            components: Principal components (from fit).
            mean: Mean used for centering.

        Returns:
            Transformed data matrix.
        """
        return (data - mean) @ components.T


# ============================================================================
# Statistical Utilities

def autocorrelation(data: np.ndarray, max_lag: int = 40) -> np.ndarray:
    """Compute autocorrelation function.

    Args:
        data: Input time series.
        max_lag: Maximum lag to compute.

    Returns:
        Array of autocorrelation values for lags 0 to max_lag-1.
    """
    mean = np.mean(data)
    var = np.var(data)
    if var == 0:
        return np.zeros(max_lag)
    return np.array([
        np.mean((data[:len(data) - lag] - mean) * (data[lag:] - mean)) / var
        for lag in range(max_lag)
    ])


def partial_autocorrelation(data: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Compute partial autocorrelation using Durbin-Levinson algorithm.

    Args:
        data: Input time series.
        max_lag: Maximum lag to compute.

    Returns:
        Array of partial autocorrelation values.
    """
    acf = autocorrelation(data, max_lag)
    pacf = np.zeros(max_lag)
    pacf[0] = 1.0
    if max_lag > 1:
        pacf[1] = acf[1]

    phi = np.zeros((max_lag, max_lag))
    phi[0, 0] = acf[1]
    for k in range(2, max_lag):
        phi[k - 1, k - 1] = (acf[k] - np.sum(phi[:k - 1, :k - 1].diagonal() * acf[1:k][::-1])) / (
            1 - np.sum(phi[:k - 1, :k - 1].diagonal() * acf[1:k])
        )
        for j in range(k - 1):
            phi[k, j] = phi[k - 1, j] - phi[k - 1, k - 1] * phi[k - 1, k - 1 - j]
        pacf[k] = phi[k - 1, k - 1]
    return pacf


def jarque_bera(data: np.ndarray) -> dict:
    """Jarque-Bera test for normality.

    Args:
        data: Input data.

    Returns:
        Dict with test statistic, p-value, and normality decision.
    """
    n = len(data)
    if n < 3:
        return {"statistic": 0.0, "p_value": 1.0, "is_normal": True}
    s = stats.skew(data)
    k = stats.kurtosis(data)
    jb = n * (s ** 2 / 6 + (k - 3) ** 2 / 24)
    p_value = 1 - stats.chi2.cdf(jb, 2)
    return {"statistic": float(jb), "p_value": float(p_value), "is_normal": p_value > 0.05}


def augmented_dickey_fuller(data: np.ndarray, maxlag: int = 1) -> dict:
    """Augmented Dickey-Fuller test for stationarity.

    Args:
        data: Time series data.
        maxlag: Maximum lag order.

    Returns:
        Dict with ADF statistic, p-value, and stationarity decision.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(data, maxlag=maxlag)
        return {
            "adf_statistic": result[0], "p_value": result[1],
            "used_lag": result[2], "critical_values": result[4],
            "is_stationary": result[1] < 0.05,
        }
    except ImportError:
        return {"error": "statsmodels not available"}

__all__ = ['KMeansClustering', 'PCA', 'autocorrelation', 'partial_autocorrelation', 'jarque_bera', 'augmented_dickey_fuller']
