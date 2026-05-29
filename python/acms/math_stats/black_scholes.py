"""Math & Statistics Library for ACMS."""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


class BlackScholes:
    """Black-Scholes option pricing model.

    Provides European option pricing, implied volatility computation,
    and full Greeks calculation.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Compute d1 in the Black-Scholes formula.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to expiry in years.
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            d1 value.
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Compute d2 in the Black-Scholes formula."""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T) if T > 0 else 0.0

    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Price a European call option.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to expiry in years.
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Call option price.
        """
        if T <= 0:
            return max(S - K, 0.0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return float(S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2))

    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Price a European put option.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to expiry in years.
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Put option price.
        """
        if T <= 0:
            return max(K - S, 0.0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return float(K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1))

    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float,
                           r: float, option_type: str = "call", tol: float = 1e-6,
                           max_iter: int = 100) -> float:
        """Compute implied volatility using Newton-Raphson method.

        Args:
            market_price: Observed option price.
            S: Spot price.
            K: Strike price.
            T: Time to expiry.
            r: Risk-free rate.
            option_type: "call" or "put".
            tol: Convergence tolerance.
            max_iter: Maximum iterations.

        Returns:
            Implied volatility.
        """
        sigma = 0.3
        for _ in range(max_iter):
            if option_type == "call":
                price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                price = BlackScholes.put_price(S, K, T, r, sigma)
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            d1 = BlackScholes.d1(S, K, T, r, sigma)
            vega = S * stats.norm.pdf(d1) * np.sqrt(T)
            if vega < 1e-10:
                break
            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 5.0))
        return sigma

    @staticmethod
    def greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
        """Compute all Greeks for a European option.

        Args:
            S: Spot price.
            K: Strike price.
            T: Time to expiry.
            r: Risk-free rate.
            sigma: Volatility.

        Returns:
            Dict with delta, gamma, theta, vega, rho.
        """
        if T <= 0:
            return {"delta": 1.0 if S > K else 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return {
            "delta": float(stats.norm.cdf(d1)),
            "gamma": float(stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))),
            "theta": float(-(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2)),
            "vega": float(S * stats.norm.pdf(d1) * np.sqrt(T)),
            "rho": float(K * T * np.exp(-r * T) * stats.norm.cdf(d2)),
        }

__all__ = ['BlackScholes']
