"""Math & Statistics Library for ACMS.

Implements:
- Black-Scholes option pricing with Greeks
- Almgren-Chriss optimal execution
- Hidden Markov Model with BIC/AIC model selection (RegimeDetection)
- Cointegration tests (Engle-Granger, Johansen)
- K-means clustering
- PCA
- GARCH(1,1) volatility model with MLE, forecasting, standardized residuals
- Kalman Filter for dynamic state estimation, hedge ratio, trend extraction
- Variance Ratio Test (Lo-MacKinlay) with multiple holding periods
- Phillips-Perron unit root test with Newey-West standard errors
- Copula models: Gaussian copula, Student-t copula
- Wavelet decomposition: Haar and DB4 wavelets
- Bootstrap methods: percentile, BCa, block bootstrap
- Granger causality test with lag selection
- Hurst exponent: R/S analysis with Anis-Lloyd correction, bootstrapped CI
- Statistical utilities: autocorrelation, partial autocorrelation, Jarque-Bera, ADF
"""

import numpy as np
from scipy import stats, linalg
from typing import Optional, Dict, List, Tuple, Callable


# ============================================================================
# Options Pricing
# ============================================================================

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


# ============================================================================
# Optimal Execution
# ============================================================================

class AlmgrenChriss:
    """Almgren-Chriss optimal execution model.

    Computes the optimal trading trajectory that minimizes
    expected cost plus risk (variance of cost).
    """

    def __init__(self, total_shares: float, total_time: int, sigma: float,
                 eta: float = 0.1, gamma: float = 0.1, lambd: float = 0.1):
        """Initialize Almgren-Chriss model.

        Args:
            total_shares: Total shares to execute.
            total_time: Total execution time in periods.
            sigma: Asset volatility.
            eta: Temporary impact coefficient.
            gamma: Permanent impact coefficient.
            lambd: Risk aversion parameter.
        """
        self.X = total_shares
        self.T = total_time
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lambd = lambd

    def optimal_trajectory(self, num_steps: int = 100) -> dict:
        """Compute optimal trading trajectory.

        Args:
            num_steps: Number of time steps.

        Returns:
            Dict with trajectory, trades, costs, and kappa.
        """
        kappa = np.sqrt(self.lambd * self.sigma ** 2 / self.eta)
        t = np.linspace(0, self.T, num_steps)
        x = self.X * (np.sinh(kappa * (self.T - t)) / np.sinh(kappa * self.T)) if np.sinh(kappa * self.T) != 0 else np.linspace(self.X, 0, num_steps)
        n = np.diff(x, prepend=self.X)
        n[0] = self.X - x[0]
        permanent_cost = 0.5 * self.gamma * self.X ** 2
        temporary_cost = self.eta * np.sum(n ** 2)
        expected_cost = permanent_cost + temporary_cost
        cost_variance = self.sigma ** 2 * np.sum(x[:-1] ** 2 * np.diff(t))
        return {
            "trajectory": x, "trades": n, "times": t,
            "expected_cost": float(expected_cost),
            "cost_variance": float(cost_variance), "kappa": float(kappa),
        }


# ============================================================================
# Hurst Exponent
# ============================================================================

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


# ============================================================================
# GARCH(1,1) Volatility Model
# ============================================================================

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


# ============================================================================
# Kalman Filter
# ============================================================================

class KalmanFilter:
    """Kalman Filter for dynamic state estimation.

    Implements the standard linear Kalman filter for:
    - Dynamic hedge ratio estimation
    - Price level tracking with noise
    - Trend extraction
    Uses Joseph form for numerical stability.
    """

    def __init__(self, state_dim: int = 1, observation_dim: int = 1,
                 process_noise: float = 1e-5, measurement_noise: float = 1e-2):
        """Initialize Kalman Filter.

        Args:
            state_dim: Dimension of state vector.
            observation_dim: Dimension of observation vector.
            process_noise: Process noise variance.
            measurement_noise: Measurement noise variance.
        """
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(observation_dim) * measurement_noise
        self.H = np.eye(observation_dim, state_dim)
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        """Initialize state and covariance.

        Args:
            x0: Initial state estimate.
            P0: Initial state covariance.
        """
        self.x = x0.copy()
        self.P = P0 if P0 is not None else np.eye(self.state_dim)

    def predict(self, F: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step: project state forward.

        Args:
            F: State transition matrix (default: identity).

        Returns:
            Tuple of (predicted_state, predicted_covariance).
        """
        if F is None:
            F = np.eye(self.state_dim)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update step: incorporate observation.

        Uses Joseph form for numerical stability.

        Args:
            z: Observation vector.

        Returns:
            Tuple of (updated_state, updated_covariance).
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.state_dim) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy(), self.P.copy()

    def filter_series(self, observations: np.ndarray,
                      F: Optional[np.ndarray] = None) -> Dict:
        """Run Kalman filter on a series of observations.

        Args:
            observations: Matrix of observations (T x obs_dim).
            F: State transition matrix.

        Returns:
            Dict with filtered states and covariances.
        """
        T = len(observations)
        states = np.zeros((T, self.state_dim))
        covs = np.zeros((T, self.state_dim, self.state_dim))

        for t in range(T):
            self.predict(F)
            self.update(observations[t])
            states[t] = self.x
            covs[t] = self.P

        return {"states": states, "covariances": covs}

    @staticmethod
    def dynamic_hedge_ratio(y: np.ndarray, x: np.ndarray) -> Dict:
        """Estimate dynamic hedge ratio using Kalman filter.

        Models the relationship y_t = alpha_t + beta_t * x_t + epsilon_t
        where alpha and beta follow random walks.

        Args:
            y: Dependent variable series.
            x: Independent variable series.

        Returns:
            Dict with hedge_ratio and intercept series.
        """
        T = len(y)
        if T < 10 or len(x) != T:
            return {"hedge_ratio": np.ones(T), "intercept": np.zeros(T)}

        kf = KalmanFilter(state_dim=2, observation_dim=1,
                          process_noise=1e-5, measurement_noise=1e-2)
        kf.H = np.array([[1.0, x[0]]])
        kf.initialize(np.array([0.0, 1.0]))

        ratios = np.zeros(T)
        intercepts = np.zeros(T)

        for t in range(T):
            kf.predict()
            kf.H = np.array([[1.0, x[t]]])
            kf.update(np.array([y[t]]))
            intercepts[t] = kf.x[0]
            ratios[t] = kf.x[1]

        return {"hedge_ratio": ratios, "intercept": intercepts}

    @staticmethod
    def trend_extraction(data: np.ndarray, noise_var: float = 1e-4,
                         process_var: float = 1e-6) -> Dict:
        """Extract trend from noisy data using Kalman filter.

        Models observation as trend + noise where trend follows
        a local level model (random walk).

        Args:
            data: Input time series.
            noise_var: Observation noise variance.
            process_var: Process (trend) noise variance.

        Returns:
            Dict with trend and innovation series.
        """
        T = len(data)
        if T < 5:
            return {"trend": data.copy(), "innovations": np.zeros(T)}

        kf = KalmanFilter(state_dim=1, observation_dim=1,
                          process_noise=process_var, measurement_noise=noise_var)
        kf.initialize(np.array([data[0]]))

        trend = np.zeros(T)
        innovations = np.zeros(T)

        for t in range(T):
            kf.predict()
            innovations[t] = data[t] - kf.x[0]
            kf.update(np.array([data[t]]))
            trend[t] = kf.x[0]

        return {"trend": trend, "innovations": innovations}


# ============================================================================
# Hidden Markov Model
# ============================================================================

class HMM:
    """Hidden Markov Model with Viterbi decoding and BIC selection.

    Implements the Baum-Welch (EM) algorithm for parameter estimation,
    Viterbi algorithm for state decoding, and BIC for model selection.
    """

    def __init__(self, n_states: int = 3):
        """Initialize HMM.

        Args:
            n_states: Number of hidden states.
        """
        self.n_states = n_states
        self.transition: Optional[np.ndarray] = None
        self.emission_params: Optional[List[Tuple[float, float]]] = None
        self.initial_probs: Optional[np.ndarray] = None
        self.log_likelihood: Optional[float] = None

    def fit(self, observations: np.ndarray, max_iter: int = 100, tol: float = 1e-6):
        """Fit HMM using Baum-Welch (EM) algorithm.

        Args:
            observations: 1-D observation sequence.
            max_iter: Maximum EM iterations.
            tol: Log-likelihood convergence tolerance.

        Returns:
            self
        """
        T = len(observations)
        N = self.n_states

        self.transition = np.ones((N, N)) / N + np.random.randn(N, N) * 0.01
        self.transition = np.abs(self.transition)
        self.transition /= self.transition.sum(axis=1, keepdims=True)

        self.initial_probs = np.ones(N) / N
        obs_sorted = np.sort(observations)
        quantiles = np.array_split(obs_sorted, N)
        self.emission_params = [(np.mean(q), np.std(q) + 1e-8) for q in quantiles]

        prev_ll = -np.inf
        ll = prev_ll
        for iteration in range(max_iter):
            alpha, beta, gamma, xi, ll = self._forward_backward(observations)

            self.initial_probs = gamma[0] / (gamma[0].sum() + 1e-300)

            for i in range(N):
                for j in range(N):
                    self.transition[i, j] = xi[:, i, j].sum() / (gamma[:, i].sum() + 1e-300)

            for i in range(N):
                weights = gamma[:, i]
                w_sum = weights.sum()
                if w_sum > 0:
                    new_mean = np.sum(weights * observations) / w_sum
                    new_std = np.sqrt(np.sum(weights * (observations - new_mean) ** 2) / w_sum) + 1e-8
                    self.emission_params[i] = (new_mean, new_std)

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        self.log_likelihood = ll
        return self

    def _forward_backward(self, observations: np.ndarray) -> Tuple:
        """Forward-backward algorithm for computing posterior state probabilities."""
        T = len(observations)
        N = self.n_states

        B = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                mu, sigma = self.emission_params[i]
                B[t, i] = stats.norm.pdf(observations[t], mu, sigma)

        alpha = np.zeros((T, N))
        c = np.zeros(T)

        alpha[0] = self.initial_probs * B[0]
        c[0] = alpha[0].sum()
        if c[0] > 0:
            alpha[0] /= c[0] + 1e-300
        else:
            alpha[0] = np.ones(N) / N
            c[0] = 1.0

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = B[t, j] * np.sum(alpha[t - 1] * self.transition[:, j])
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t] + 1e-300
            else:
                alpha[t] = np.ones(N) / N
                c[t] = 1.0

        beta = np.zeros((T, N))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(self.transition[i, :] * B[t + 1, :] * beta[t + 1, :])
            if c[t + 1] > 0:
                beta[t] /= c[t + 1] + 1e-300

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = alpha[t, i] * self.transition[i, j] * B[t + 1, j] * beta[t + 1, j]
            xi[t] /= xi[t].sum() + 1e-300

        ll = np.sum(np.log(c + 1e-300))
        return alpha, beta, gamma, xi, ll

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Viterbi algorithm for most likely state sequence.

        Args:
            observations: 1-D observation sequence.

        Returns:
            Array of most likely states.
        """
        T = len(observations)
        N = self.n_states

        B = np.zeros((T, N))
        for t in range(T):
            for i in range(N):
                mu, sigma = self.emission_params[i]
                B[t, i] = stats.norm.pdf(observations[t], mu, sigma)

        V = np.zeros((T, N))
        backpointer = np.zeros((T, N), dtype=int)
        V[0] = np.log(self.initial_probs + 1e-300) + np.log(B[0] + 1e-300)

        for t in range(1, T):
            for j in range(N):
                prob = V[t - 1] + np.log(self.transition[:, j] + 1e-300)
                backpointer[t, j] = np.argmax(prob)
                V[t, j] = prob[backpointer[t, j]] + np.log(B[t, j] + 1e-300)

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            states[t] = backpointer[t + 1, states[t + 1]]
        return states

    @staticmethod
    def select_n_states(observations: np.ndarray, max_states: int = 6) -> Dict:
        """Select optimal number of states using BIC.

        BIC = -2 * log_likelihood + k * ln(T)

        Args:
            observations: 1-D observation sequence.
            max_states: Maximum number of states to test.

        Returns:
            Dict with optimal n_states and BIC values.
        """
        T = len(observations)
        bic_values = {}

        for n in range(2, max_states + 1):
            hmm = HMM(n_states=n)
            hmm.fit(observations)
            k = n * (n - 1) + n * 2 + (n - 1)
            bic = -2 * hmm.log_likelihood + k * np.log(T)
            bic_values[n] = bic

        best_n = min(bic_values, key=bic_values.get)
        return {"optimal_n_states": best_n, "bic_values": bic_values}


class RegimeDetection:
    """Regime detection using HMM with BIC/AIC model selection.

    Provides a high-level interface for detecting market regimes
    with automatic model selection.
    """

    @staticmethod
    def detect(returns: np.ndarray, max_states: int = 6,
               method: str = "bic") -> Dict:
        """Detect market regimes with automatic state selection.

        Args:
            returns: Return series.
            max_states: Maximum number of regimes to test.
            method: Model selection criterion ("bic" or "aic").

        Returns:
            Dict with optimal model, states, and model selection results.
        """
        if len(returns) < 100:
            return {"states": np.zeros(len(returns), dtype=int),
                    "optimal_n_states": 1, "model": None}

        best_score = float('inf')
        best_n = 2
        best_hmm = None
        scores = {}

        for n in range(2, max_states + 1):
            hmm = HMM(n_states=n)
            hmm.fit(returns)

            k = n * (n - 1) + n * 2 + (n - 1)
            T = len(returns)

            if method == "aic":
                score = -2 * hmm.log_likelihood + 2 * k
            else:  # BIC
                score = -2 * hmm.log_likelihood + k * np.log(T)

            scores[n] = score
            if score < best_score:
                best_score = score
                best_n = n
                best_hmm = hmm

        states = best_hmm.viterbi(returns) if best_hmm is not None else np.zeros(len(returns), dtype=int)

        return {
            "states": states,
            "optimal_n_states": best_n,
            "model": best_hmm,
            "scores": scores,
            "method": method,
        }


# ============================================================================
# Variance Ratio Test
# ============================================================================

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
# ============================================================================

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


# ============================================================================
# Copula Models
# ============================================================================

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


# ============================================================================
# Wavelet Decomposition
# ============================================================================

class WaveletDecomposition:
    """Wavelet decomposition for multi-scale analysis.

    Implements discrete wavelet transforms:
    - Haar: Simplest wavelet, good for piecewise constant signals
    - DB4: Daubechies 4-coefficient wavelet, smoother decomposition
    """

    @staticmethod
    def haar_transform(data: np.ndarray) -> Dict:
        """Compute Haar wavelet decomposition.

        Args:
            data: 1-D signal to decompose.

        Returns:
            Dict with approximation and detail coefficients at each level.
        """
        n = len(data)
        if n < 4:
            return {"approximations": [data], "details": [np.array([])], "levels": 0}

        next_pow2 = int(2 ** np.ceil(np.log2(n)))
        padded = np.zeros(next_pow2)
        padded[:n] = data

        approximations = []
        details = []
        current = padded.copy()
        level = 0

        while len(current) >= 4:
            half = len(current) // 2
            approx = np.zeros(half)
            detail = np.zeros(half)

            for i in range(half):
                approx[i] = (current[2*i] + current[2*i + 1]) / np.sqrt(2)
                detail[i] = (current[2*i] - current[2*i + 1]) / np.sqrt(2)

            approximations.append(approx[:min(half, n)])
            details.append(detail[:min(half, n)])
            current = approx
            level += 1

        return {"approximations": approximations, "details": details, "levels": level, "wavelet": "haar"}

    @staticmethod
    def db4_transform(data: np.ndarray) -> Dict:
        """Compute DB4 (Daubechies 4) wavelet decomposition.

        Uses the 4-coefficient Daubechies filter bank for smoother
        decomposition than Haar.

        Args:
            data: 1-D signal to decompose.

        Returns:
            Dict with approximation and detail coefficients.
        """
        n = len(data)
        if n < 8:
            return WaveletDecomposition.haar_transform(data)

        # DB4 filter coefficients
        h = np.array([
            0.2303778133088964,
            0.7148465705529154,
            0.6308807679398587,
            -0.0279837694168599,
            -0.1870348117190931,
            0.0308413818355607,
            0.0328830116668852,
            -0.0105974017850690,
        ])
        g = np.array([
            -0.0105974017850690,
            -0.0328830116668852,
            0.0308413818355607,
            0.1870348117190931,
            -0.0279837694168599,
            -0.6308807679398587,
            0.7148465705529154,
            -0.2303778133088964,
        ])

        # Pad to even length
        pad_len = len(h)
        padded = np.zeros(n + pad_len)
        padded[:n] = data

        approximations = []
        details = []
        current = padded.copy()
        level = 0

        while len(current) >= 2 * len(h):
            n_curr = len(current)
            half = n_curr // 2

            # Periodic extension
            extended = np.concatenate([current, current[:pad_len]])

            approx = np.zeros(half)
            detail = np.zeros(half)

            for i in range(half):
                for k in range(len(h)):
                    idx = 2 * i + k
                    if idx < len(extended):
                        approx[i] += h[k] * extended[idx]
                        detail[i] += g[k] * extended[idx]

            approximations.append(approx)
            details.append(detail)
            current = approx
            level += 1

            if level >= 10:  # Safety limit
                break

        return {
            "approximations": approximations,
            "details": details,
            "levels": level,
            "wavelet": "db4",
        }

    @staticmethod
    def denoise(data: np.ndarray, threshold: float = 1.0,
                wavelet: str = "haar") -> np.ndarray:
        """Denoise signal using wavelet thresholding.

        Applies soft thresholding to detail coefficients and
        reconstructs the signal.

        Args:
            data: Input signal.
            threshold: Threshold for detail coefficients.
            wavelet: Wavelet type ("haar" or "db4").

        Returns:
            Denoised signal.
        """
        if wavelet == "db4":
            result = WaveletDecomposition.db4_transform(data)
        else:
            result = WaveletDecomposition.haar_transform(data)

        if not result["details"] or not result["approximations"]:
            return data

        # Soft thresholding
        denoised_details = []
        for detail in result["details"]:
            denoised = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            denoised_details.append(denoised)

        # Reconstruct using approximation at coarsest level
        approx = result["approximations"][-1]
        reconstructed = np.repeat(approx, max(1, len(data) // len(approx)))

        if len(reconstructed) < len(data):
            reconstructed = np.pad(reconstructed, (0, len(data) - len(reconstructed)), mode='edge')
        return reconstructed[:len(data)]


# ============================================================================
# Bootstrap Methods
# ============================================================================

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


# ============================================================================
# Granger Causality Test
# ============================================================================

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
            "causes": best_p < 0.05,
            "best_lag": best_lag,
        }


# ============================================================================
# Cointegration
# ============================================================================

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
            pass

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


# ============================================================================
# Clustering & PCA
# ============================================================================

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
# ============================================================================

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
